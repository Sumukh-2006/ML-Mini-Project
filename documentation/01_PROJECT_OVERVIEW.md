# 01. Project Overview & Master Study Summary

---

## 1. Project Title
**End-to-End Statistical Parametric Text-to-Speech (TTS) Synthesis System using Hidden Markov Models (HMM) and Maximum Likelihood Parameter Generation (MLPG)**

---

## 2. One-Paragraph Executive Summary
This project implements a complete, classical **Statistical Parametric Speech Synthesis (SPSS)** system from scratch, mapping written English text directly to synthetic speech audio without relying on modern deep neural network black boxes. Built on the single-speaker **LJ Speech Dataset (v1.1)**, the system extracts 80-dimensional log-Mel spectrograms, performs deterministic text normalization and Grapheme-to-Phoneme conversion, aligns phonemes to acoustic frames via Montreal Forced Aligner, and estimates supervised parameters for a 3-state left-to-right monophone **Hidden Markov Model (HMM)**. In **Version 1 (V1)**, the system generates acoustic trajectories by directly emitting Gaussian mean vectors, which produces an intelligible baseline but causes severe piecewise-constant "stair-step" discontinuities at state boundaries. In **Version 2 (V2)**, we augment the acoustic representation with utterance-wide velocity ($\Delta$) and acceleration ($\Delta\Delta$) dynamic features (240 dimensions total) and formulate speech synthesis as a **Maximum Likelihood Parameter Generation (MLPG)** trajectory optimization solved in linear time via a banded Cholesky solver. Evaluated on a strictly isolated 655-utterance held-out test split, V2 achieves a **31.68 percentage-point reduction in Whisper-tiny Word Error Rate (WER)** (78.78% $\rightarrow$ 47.10%), a **1.64 dB reduction in Mel-Cepstral Distortion (MCD)** (40.95 dB $\rightarrow$ 39.31 dB), and a **67.8% drop in boundary discontinuity**, demonstrating the power of mathematical smoothness constraints in statistical machine learning.

---

## 3. Introduction

### What is Text-to-Speech (TTS)?
**Text-to-Speech (TTS)**, also known as speech synthesis, is the automated process of converting natural language text into intelligible, human-like speech waveforms.

### Inputs and Outputs
* **Input:** A sequence of characters or words (graphemes), e.g., `"The examination begins tomorrow morning."`
* **Output:** A continuous, one-dimensional time-domain acoustic pressure waveform $x(t)$, digitized as discrete pulse-code modulation (PCM) audio samples at 22,050 Hz.

### Why is TTS a Difficult Machine Learning Problem?
1. **Extreme One-to-Many Mapping:** A single sentence can be spoken in infinite ways depending on speaker pitch, emotion, cadence, stress, and accent. The machine learning model must find a coherent, natural acoustic trajectory among an infinite set of valid waveforms.
2. **Loss of High-Level Structure in Raw Text:** Text contains semantic and grammatical information, but it completely omits physical acoustic instructions such as vocal tract shaping, fundamental frequency ($F_0$), breathing pauses, and phoneme durations.
3. **Sequential and Temporal Dependency:** Speech is not a set of independent frames; every millisecond of speech depends heavily on physical inertia—the tongue, lips, and vocal cords cannot instantaneously jump from one configuration to another.
4. **Different from Standard Classification and Regression:** In standard classification (e.g., predicting sentiment from text), many input tokens collapse into a single discrete class. In regression (e.g., house price prediction), multiple tabular features map to a scalar value. In TTS, a low-dimensional, discrete input sequence (a few dozen letters) must map into a high-dimensional, continuous temporal signal (hundreds of thousands of audio samples) with strict temporal and spectral coherence.

---

## 4. Problem Statement

### Formal Definition
Given an arbitrary string of normalized English text $W = (w_1, w_2, \dots, w_N)$, synthesize a discrete audio waveform $\mathbf{x} = [x_1, x_2, \dots, x_M]$ that is:
1. **Intelligible:** Accurately conveys the exact words spoken.
2. **Acoustically Plausible:** Preserves realistic spectral envelopes and formant transitions.
3. **Temporally Consistent:** Respects human articulatory rates without jarring acoustic jumps.

### What is Given?
* A raw training dataset of audio clips paired with textual transcriptions (LJ Speech v1.1).
* A fixed lexicon: the CMU Pronouncing Dictionary.
* Deterministic text-processing rules.

### What are We Trying to Learn?
* The mapping between phonetic symbols and sub-phonetic acoustic distributions.
* The statistical mean and variance of spectral envelopes for each phoneme sub-state.
* The typical duration (frame count) spent in each phonetic sub-state.
* In V2: The temporal rate of change (velocity and acceleration) associated with each phonetic sub-state.

### What Makes the Mapping Difficult?
The physical human vocal tract is a continuous mechanical resonator. When speaking, the tongue and vocal tract begin shaping the acoustics of the *next* sound while still pronouncing the *current* sound (co-articulation). Capturing this continuous physical movement using statistical discrete state models requires careful mathematical formulation.

---

## 5. What Are We Actually Trying to Solve? (Deconstructed ML View)

Rather than treating speech synthesis as an intractable direct mapping from text characters to individual raw audio pressure samples ($W \rightarrow \mathbf{x}$), we decompose the pipeline into four distinct stages:

```
[Raw Text]
    ↓  (Deterministic Rule-Based Text Processing)
[Token Sequence & Punctuation]
    ↓  (Lexicon Lookup & G2P Fallback)
[Phonetic Symbol Sequence]
    ↓  (Learned Duration Model / Lookup)
[HMM Temporal State Sequence]
    ↓  (Statistical Acoustic Emission / Trajectory Optimization)
[Acoustic Log-Mel Spectrogram Trajectory]
    ↓  (Deterministic Pseudo-Inverse & Iterative Phase Estimation)
[Synthesized Audio Waveform]
```

### Statistical Learning vs. Deterministic Processing
To clearly understand our machine learning system, we must distinguish what is learned statistically from data versus what is fixed or computed deterministically:

| Pipeline Stage | Nature | Component Description |
| :--- | :--- | :--- |
| **Text Normalization** | Deterministic | Regex cleaning, NFKD ASCII conversion, acronym spelling. |
| **G2P Conversion** | Deterministic / Rule-based | CMU Pronouncing Dictionary lookup with neural fallback. |
| **Phoneme Alignment** | Supervised Extraction | Montreal Forced Aligner (MFA) extracts phone time boundaries. |
| **Acoustic Model (HMM)** | **Statistically Learned** | Gaussian emission means $\boldsymbol{\mu}$ and diagonal variances $\boldsymbol{\Sigma}$. |
| **Duration Model** | **Statistically Learned** | Empirical mean and variance of frame counts per state. |
| **Dynamic Features (V2)** | Deterministic Transform | Utterance-wide discrete convolutions (finite difference operators). |
| **Trajectory Solver (V2)** | **Analytical Optimization** | MLPG linear system solution: $(W^T U^{-1} W) C = W^T U^{-1} \boldsymbol{\mu}$. |
| **Vocoder (Waveform)** | Deterministic DSP | Mel pseudo-inverse filterbank + Fast Griffin-Lim algorithm. |

---

## 6. What Does Our Model Learn?

It is vital in an academic examination to know exactly which numerical parameters are stored in the trained model JSON file (`v1_hmm_model.json` and `v2_hmm_model.json`):

### 1. Statistically Learned from Data
* **State Emission Means ($\boldsymbol{\mu}$):**
  * **V1:** Shape `[40, 3, 80]` $\rightarrow$ 40 phonemes $\times$ 3 temporal states $\times$ 80 log-Mel static dimensions = 9,600 floating-point parameters.
  * **V2:** Shape `[40, 3, 240]` $\rightarrow$ 40 phonemes $\times$ 3 temporal states $\times$ 240 dimensions (80 static + 80 delta + 80 delta-delta) = 28,800 floating-point parameters.
* **State Emission Variances ($\boldsymbol{\Sigma}$ or $\boldsymbol{\sigma}^2$):**
  * **V1:** Shape `[40, 3, 80]` diagonal variance elements floored at $10^{-6}$.
  * **V2:** Shape `[40, 3, 240]` diagonal variance elements floored at $10^{-6}$.
* **State Observation Counts:**
  * Shape `[40, 3]` recording exactly how many acoustic frames were mapped to each state across the 11,790 training utterances.
* **Duration Statistics:**
  * `duration_mean`: Shape `[40, 3]` (average frames per state).
  * `duration_std`: Shape `[40, 3]` (standard deviation of frame duration).
  * `duration_min` and `duration_max`: Shape `[40, 3]`.
  * `phone_total_duration_mean`: Shape `[40]` (sum of average state durations).

### 2. Fixed by Design (Architectural Constants)
* Number of phonetic classes: 40 (39 ARPAbet phonemes + 1 `SIL` silence symbol).
* Number of states per phone: 3 left-to-right temporal states (State 0: beginning, State 1: middle, State 2: end).
* Topology: Strictly left-to-right with no skip transitions ($S_0 \rightarrow S_1 \rightarrow S_2$).
* Acoustic sample rate: 22,050 Hz.
* STFT configuration: `n_fft=1024`, `win_length=1024`, `hop_length=256` (11.6 ms hop, 46.4 ms window).
* Mel scale: 80 filterbank channels spanning 0 Hz to 8,000 Hz with Slaney normalization.
* Variance floor: $10^{-6}$ to prevent numerical instability and division by zero.

### 3. Computed Deterministically during Preprocessing or Inference
* Text tokenization and acronym detection (`#fbi`).
* Lexicon phoneme mapping.
* Frame mapping: $T = 1 + \lfloor \text{n\_samples} / 256 \rfloor$.
* Window matrix $W$ in V2 (finite difference coefficients).
* Pseudo-inverse Mel basis (`pinv(MEL_BASIS)`).
* Phase estimation via Griffin-Lim (60 iterations, momentum 0.99).

> [!TIP]
> **Viva Takeaway:** The acoustic model does NOT use backpropagation or gradient descent. Because forced alignment provides supervised frame-to-state labels, parameters are estimated via direct closed-form Maximum Likelihood Estimation (MLE)—calculating sample means and sample variances from sufficient statistics ($\sum x$ and $\sum x^2$).

---

## 7. What Does Our Model Predict/Generate?

At inference time, the model takes an unseen text sentence and performs the following step-by-step transformations:

1. **Text Normalization Output:** Normalized token list (words and `<sil>` markers).
2. **G2P Output:** Sequence of discrete phonetic strings, e.g., `['SIL', 'M', 'AH', 'SH', 'IY', 'N', ..., 'SIL']`.
3. **State Sequence Output:** Expansion of each phoneme into 3 sequential temporal states:
   $$(P_1, S_0), (P_1, S_1), (P_1, S_2), (P_2, S_0), \dots$$
4. **Predicted Durations:** Integer frame count for each state $d_{p, s} = \max(1, \lfloor \text{round}(\mu_{dur}(p, s)) \rfloor)$.
5. **Predicted Acoustic Representation:**
   * **In V1:** Tiled Gaussian mean vectors yielding a piecewise-constant matrix $[T, 80]$.
   * **In V2:** Utterance-wide trajectory optimization via MLPG yielding a continuous, smooth matrix $[T, 80]$.
6. **Synthesized Waveform:** Continuous 1D pressure array $[N]$ at 22,050 Hz produced by Griffin-Lim inversion.

---

## 8. Complete End-to-End Pipeline

```
                       [Raw Input Text]
                              │
                              ▼
            ┌───────────────────────────────────┐
            │  1. Deterministic Text Processing │
            │  - Lowercase, ASCII mapping (NFKD)│
            │  - Punctuation -> SIL tokens      │
            │  - Acronym detection (#fbi)       │
            └───────────────────────────────────┘
                              │
                              ▼
            ┌───────────────────────────────────┐
            │  2. Grapheme-to-Phoneme (G2P)     │
            │  - CMU Pronouncing Dictionary     │
            │  - Possessive rule expansion      │
            │  - Neural g2p_en fallback for OOV │
            └───────────────────────────────────┘
                              │
               Sequence of 40 ARPAbet Phones
                              │
                              ▼
            ┌───────────────────────────────────┐
            │  3. Duration & State Lookup       │
            │  - 3 states per phone (0, 1, 2)   │
            │  - Frame duration = round(mean)   │
            │  - Min duration clamp >= 1 frame  │
            └───────────────────────────────────┘
                              │
            State Sequence + Frame Durations [T frames]
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       [V1 PIPELINE]                 [V2 PIPELINE]
┌─────────────────────────────┐ ┌─────────────────────────────┐
│ Direct Mean Emission        │ │ MLPG Trajectory Solver      │
│ - Tile 80-D mean for dur[s] │ │ - Tile 240-D means & vars   │
│ - Piecewise-constant matrix │ │ - Solve: (W^T U^-1 W) C = r │
│ - Abrupt boundary jumps     │ │ - Banded Cholesky O(T) time │
└─────────────────────────────┘ └─────────────────────────────┘
               │                             │
       [T, 80] Log-Mel               [T, 80] Log-Mel
       (Piecewise-Constant)          (Smooth & Continuous)
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
            ┌───────────────────────────────────┐
            │  4. Acoustic Inversion            │
            │  - Linear scale: exp(log_mel)     │
            │  - Pseudo-inverse Mel Filterbank  │
            │  - Approx Magnitude Spectrum      │
            └───────────────────────────────────┘
                              │
                              ▼
            ┌───────────────────────────────────┐
            │  5. Phase Reconstruction          │
            │  - Fast Griffin-Lim Algorithm     │
            │  - 60 iterations, momentum = 0.99 │
            └───────────────────────────────────┘
                              │
                              ▼
            [Synthesized 22,050 Hz Audio WAV]
```

---

## 9. Dataset

### LJ Speech Dataset (v1.1)
The system is built and evaluated exclusively on the **LJ Speech Dataset**, a widely recognized public-domain benchmark in speech synthesis literature.

* **Speaker:** Single female English speaker (Linda Johnson).
* **Content:** Reading passages from 7 non-fiction books (e.g., historical trials, prison history).
* **Audio Characteristics:**
  * Utterances: Exactly 13,100 short audio clips.
  * Total Audio Duration: Approximately 23 hours and 55 minutes (~24 hours).
  * Sampling Rate: 22,050 Hz.
  * Bit Depth: 16-bit signed PCM.
  * Channels: 1 (Single-channel Mono).
  * Container: WAV format.
* **Metadata Structure:**
  * Provided in `metadata.csv` delimited by vertical pipe characters (`|`).
  * Column 1: Utterance ID (e.g., `LJ001-0001`).
  * Column 2: Raw unnormalized transcription (e.g., `"in 1465"`).
  * Column 3: Normalized transcription (e.g., `"in fourteen sixty-five"`).

### Why LJ Speech is Ideal for this Project
1. **Single-Speaker Consistency:** Statistical acoustic models (especially monophones) assume that acoustic variance is caused by phonetic identity, not speaker differences. Multi-speaker datasets conflate speaker identity with phonetic identity, causing blurry Gaussian emissions.
2. **High Studio Quality:** Recorded in a consistent quiet environment, minimizing background noise and reverberation.
3. **Pre-Normalized Transcriptions:** Eliminates arbitrary text ambiguities (numbers, dates, currency), ensuring deterministic evaluation.

---

## 10. Train / Validation / Test Split

To preserve rigorous machine learning hygiene and prevent data leakage, the 13,100 utterances are partitioned deterministically using a fixed random seed (`seed=42`).

```
Total LJ Speech Dataset: 13,100 Utterances (100%)
├── Training Set:   11,790 Utterances (90%) -> Used ONLY to fit means, variances, durations
├── Validation Set:    655 Utterances ( 5%) -> Used for model diagnostics and validation
└── Held-Out Test:     655 Utterances ( 5%) -> STRICTLY FROZEN until final evaluation
```

### Critical Split Guarantees
1. **Utterance-Level Splitting:** The split is performed strictly at the utterance level, NOT at the frame level. Audio frames from the same utterance never cross split boundaries. If frames from the same sentence appeared in both train and test, the test evaluation would suffer from temporal autocorrelation leakage.
2. **Mutual Exclusivity:**
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
3. **Strict Test Set Quarantine:**
   During the entire design, tuning, and debugging of V2, the 655 test utterances (`data_splits/test.txt`) remained completely untouched. Hyperparameters, boundary padding methods, and solver choices were validated solely on the validation set.

---

## 11. Key Concepts and Glossary

Every technical term and abbreviation used across the project is defined below:

| Full Form | Abbreviation | Simple Intuition | Exact Role in This Project |
| :--- | :--- | :--- | :--- |
| **Text-to-Speech** | **TTS** | Converting written text into spoken sound. | The overarching problem this project solves. |
| **Statistical Parametric Speech Synthesis** | **SPSS** | Generating speech by predicting parameters of a statistical model rather than concatenating recorded audio snippets. | The fundamental machine learning paradigm used. |
| **Hidden Markov Model** | **HMM** | A probabilistic model where unobserved (hidden) states generate observable data points sequentially. | Our acoustic model representing the temporal states of each phoneme. |
| **Grapheme-to-Phoneme** | **G2P** | Translating written letters into pronunciation sound codes. | Frontend module converting words into ARPAbet phonetic symbols. |
| **Montreal Forced Aligner** | **MFA** | An external alignment tool that synchronizes text phonemes with audio frames using acoustic models. | Used offline during preprocessing to generate frame-to-phone timestamps. |
| **Short-Time Fourier Transform** | **STFT** | Slicing audio into overlapping time windows and computing frequency contents for each window. | Fundamental DSP operation converting 1D audio waveforms into 2D time-frequency spectra. |
| **Mel Spectrogram** | **Mel-Spec** | A spectrogram whose frequency axis is warped to match human ear pitch sensitivity. | The target acoustic representation our machine learning model learns to generate. |
| **Maximum Likelihood Parameter Generation** | **MLPG** | An algorithm that finds the most probable smooth continuous trajectory given static and dynamic Gaussian distributions. | The core innovation of Version 2 that eliminates stair-step discontinuities. |
| **Dynamic Features** | **$\Delta$ & $\Delta\Delta$** | Discrete time derivatives representing acoustic velocity and acceleration across frames. | Appended to static features to give the HMM temporal momentum constraints. |
| **Word Error Rate** | **WER** | Standard metric measuring the percentage of words incorrectly recognized by an ASR system. | Objective proxy for speech intelligibility using Whisper-tiny. |
| **Mel-Cepstral Distortion** | **MCD** | Distance metric measuring spectral distortion between synthesized and reference speech in decibels. | Objective acoustic metric measuring spectral envelope reconstruction accuracy. |
| **Global Variance** | **GV** | The overall variance of an acoustic parameter across an entire utterance. | Diagnostic metric indicating trajectory smoothing or over-smoothing. |

---

## 12. Evaluation Metrics

All metrics used to evaluate V1 and V2 are formally defined below:

### 1. Whisper-tiny Word Error Rate (WER) (ASR-Based Intelligibility Proxy)
* **What it measures:** The proportion of word errors made by an automated speech recognizer (OpenAI Whisper-tiny) when transcribing synthesized audio, compared against the reference text.
* **Mathematical Formula:**
  $$\text{WER} = \frac{S + D + I}{N}$$
  Where:
  * $S$ = Number of word substitutions.
  * $D$ = Number of word deletions.
  * $I$ = Number of word insertions.
  * $N$ = Total number of words in the ground-truth reference text.
* **Optimization Direction:** Lower is better ($0.0$ is perfect transcription).
* **Role in Project:** Serves as an objective, automated proxy for human intelligibility.
* **Limitations:** Whisper-tiny is a neural model with internal language modeling priors; it can hallucinate or autocorrect slightly distorted phonemes based on context, meaning WER does not perfectly equate to human perceptual listener effort.

### 2. DCT-Derived Cepstral Distortion (MCD-Style Metric) (dB)
* **What it measures:** The distance between the spectral envelope of the generated audio and the ground-truth natural recording.
* **Mathematical Formulation:**
  1. Transform 80-D log-Mel vectors into 16 Mel-Cepstral Coefficients (MCCs) via an orthonormal Type-II Discrete Cosine Transform (DCT-II):
     $$c_t(k) = \text{DCT-II}(\mathbf{m}_t)$$
  2. Drop the 0th coefficient $c_0$ (which represents overall frame energy/volume), retaining $k \in \{1, \dots, 15\}$ to isolate pure spectral vocal tract shape.
  3. Align the reference and synthesized trajectories using Dynamic Time Warping (DTW) with Euclidean distance cost to compensate for duration variations.
  4. Convert the mean path cost to decibels (dB) using the standard acoustic scaling factor:
     $$\text{MCD} = \frac{10\sqrt{2}}{\ln 10} \cdot \frac{1}{L} \sum_{l=1}^L \|\mathbf{c}_{\text{ref}}(i_l) - \mathbf{c}_{\text{gen}}(j_l)\|_2 \approx 6.14185 \cdot \text{Mean Path Cost}$$
* **Optimization Direction:** Lower is better ($0\text{ dB}$ is an exact spectral match).
* **Role in Project:** Quantifies how closely our generated log-Mel filterbank captures natural formant peaks and valleys.

### 3. Boundary Discontinuity (BD)
* **What it measures:** The average absolute spectral step across state and phoneme transitions:
  $$\text{BD} = \frac{1}{K-1} \sum_{k=1}^{K-1} \frac{1}{80} \sum_{d=1}^{80} |m_{t_k}(d) - m_{t_k - 1}(d)|$$
  where $t_k$ represents the exact frame index where state $k$ transitions to state $k+1$.
* **Optimization Direction:** Lower is better.
* **Role in Project:** Direct numerical test of whether the V2 MLPG solver successfully smoothed out the stair-step jumps observed in V1.

### 4. Trajectory First-Order ($\bar{d}_1$) and Second-Order ($\bar{d}_2$) Change
* **What they measure:** The average utterance-wide frame-to-frame velocity and acceleration magnitudes:
  $$\bar{d}_1 = \frac{1}{(T-1)D} \sum_{t=1}^{T-1} \sum_{d=1}^D |m_{t+1}(d) - m_t(d)|$$
  $$\bar{d}_2 = \frac{1}{(T-2)D} \sum_{t=1}^{T-2} \sum_{d=1}^D |m_{t+2}(d) - 2m_{t+1}(d) + m_t(d)|$$
* **Optimization Direction:** Lower values indicate smoother, less volatile parameter curves.

### 5. Global Variance (GV)
* **What it measures:** The empirical variance of the acoustic trajectory across time:
  $$\text{GV} = \frac{1}{D} \sum_{d=1}^D \left[ \frac{1}{T} \sum_{t=1}^T (m_t(d) - \bar{m}(d))^2 \right]$$
* **Role in Project:** Used as a diagnostic for over-smoothing. Standard MLPG reduces trajectory variance because it maximizes likelihood under Gaussian assumptions, pulling extreme formant peaks toward the mean.

### 6. Duration Prediction Error (%)
* **What it measures:** The relative percentage error between total synthesized audio duration and natural human recording duration:
  $$\text{Dur Error} = \frac{|T_{\text{synth}} - T_{\text{orig}}|}{T_{\text{orig}}} \times 100\%$$
* **Optimization Direction:** Lower is better. In both V1 and V2, mean duration error is approximately **8.5%**.

---

## 13. Overall Project Evolution: V1 vs. V2

```
=============================================================================
VERSION 1 (Baseline Monophone HMM)
-----------------------------------------------------------------------------
Representation:      80-D static log-Mel spectrogram
Acoustic Generation: Direct Gaussian mean emission (tiled across duration)
Visual Trajectory:   Piecewise-constant "stair-steps"
Vocal Quality:       Robotic, buzz-like timbre, harsh frame jumps
Test WER / MCD:      78.78% WER  |  40.95 dB MCD
Boundary Jump:       0.6136 mean step magnitude
=============================================================================
                                     │
                 IDENTIFIED MATHEMATICAL LIMITATION:
                 Frame-by-frame mean emission assumes frames are independent!
                 Physical articulators have inertia and velocity constraints.
                                     │
                                     ▼
=============================================================================
VERSION 2 (Dynamic Features + MLPG Optimization)
-----------------------------------------------------------------------------
Representation:      240-D features (80 Static + 80 Delta + 80 Delta-Delta)
Acoustic Generation: Maximum Likelihood Parameter Generation (MLPG)
Visual Trajectory:   Smooth, continuous, dynamically constrained curve
Vocal Quality:       Substantially reduced boundary buzz, clearer phonetics
Test WER / MCD:      47.10% WER  |  39.31 dB MCD
Boundary Jump:       0.1974 mean step magnitude (-67.8% reduction)
Statistical Gain:    -31.68 percentage points WER (95% CI: [-37.10, -27.54])
=============================================================================
```

> [!WARNING]
> **Common Viva Trap:** A professor might ask: *"Did V2 implement Global Variance (GV) optimization or deep neural networks?"*
> **Answer:** **No.** Version 2 implements classical analytical MLPG using a banded Cholesky solver with static, delta, and delta-delta distributions. Global Variance (GV) was measured only as a *diagnostic metric*, not as an active penalty term in the cost function. Future extensions like GV optimization, GMMs, and triphones were intentionally left out to keep the system theoretically rigorous, lightweight, and cleanly syllabus-aligned.
