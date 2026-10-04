# 02. Version 1 (V1) — Complete Pipeline & Viva Study Notes

---

## 1. V1 Motivation

### 1. What is it?
Version 1 (V1) is a 3-state left-to-right monophone **Hidden Markov Model (HMM)** speech synthesis system that directly emits learned 80-dimensional Gaussian mean vectors for acoustic frame generation.

### 2. Why do we need it in this project?
In machine learning research, jumping immediately to a complex architecture without an empirically validated, theoretically grounded baseline makes it impossible to isolate which algorithmic component caused improvements. V1 establishes the clean, fully explainable classical baseline.

### 3. How does it relate to the Machine Learning syllabus?
The Hidden Markov Model is a central syllabus topic under:
* Sequential Probabilistic Graphical Models.
* Maximum Likelihood Parameter Estimation.
* Temporal modeling of non-stationary signals.
* Markovian state transitions and conditional independence assumptions.

### 4. Why is HMM appropriate for speech?
Speech is inherently temporal and non-stationary. A word cannot be modeled as a single static vector because its acoustic properties change dramatically from the start to the middle to the end. The HMM naturally accommodates this structure by decomposing each sound into a sequence of left-to-right temporal hidden states.

---

## 2. Raw Dataset

The project uses the **LJ Speech Dataset (v1.1)**:
* **Audio Format:** 13,100 uncompressed single-channel (mono) WAV files.
* **Sampling Rate:** 22,050 Hz (22,050 audio samples per second).
* **Quantization:** 16-bit signed Pulse-Code Modulation (PCM) (dynamic range of $\approx 96\text{ dB}$).
* **Metadata:** Stored in `metadata.csv` using pipe-separated values:
  `Utterance_ID | Raw_Text | Normalized_Text`

### Why These Details Matter
* **22,050 Hz Sampling Rate:** By the Nyquist-Shannon Sampling Theorem, the maximum representable acoustic frequency is the Nyquist frequency:
  $$f_{\text{Nyquist}} = \frac{f_s}{2} = \frac{22,050}{2} = 11,025\text{ Hz}$$
  Human speech formants and fricative energy typically reside below 8,000 Hz, making 22,050 Hz more than sufficient while avoiding the computational overhead of 44,100 Hz or 48,000 Hz.
* **Normalized Transcriptions:** Using Column 3 (e.g., `"fourteen sixty-five"` rather than `"1465"`) removes non-deterministic text normalization ambiguity.

---

## 3. Text Preprocessing

Implemented in `preprocessing/text_processing.py`, text preprocessing transforms raw text into a strictly validated list of word tokens and silence markers.

### Exact Implemented Rules:
1. **Typographic ASCII Mapping:** Converts Unicode quotes and apostrophes to standard ASCII (`“` and `”` $\rightarrow$ `"`, `‘` and `’` $\rightarrow$ `'`).
2. **NFKD Diacritic Normalization:** Strips accents using Unicode Normalization Form KD (NFKD) decomposition (e.g., `é` $\rightarrow$ `e`, `ü` $\rightarrow$ `u`).
3. **Acronym Detection (`#fbi`):** Before lowercasing, any sequence of $\ge 2$ consecutive capital letters (e.g., `FBI`, `CIA`, `USSR`) is tagged with a `#` prefix (e.g., `#fbi`). This instructs the G2P engine to spell the word out letter-by-letter (`F-B-I`) rather than attempting a phonetic pronunciation.
4. **Lowercasing:** Converts all remaining characters to lowercase `[a-z']`.
5. **Phrase Punctuation Mapping:** Punctuation marks (`,`, `;`, `:`, `.`, `?`, `!`, `(`, `)`) are converted to the silence token `<sil>`.
6. **Double Quotes and Brackets:** Stripped completely (`"`, `[`, `]`) because punctuation quotes have no independent acoustic realization.
7. **Hyphen Handling:** Hyphens are converted to word boundaries (e.g., `"bric-a-brac"` $\rightarrow$ `"bric a brac"`).
8. **Consecutive Silence Collapsing:** Multiple adjacent `<sil>` tokens are collapsed into a single `<sil>`.
9. **Mandatory Utterance Boundary Silence:** Every processed utterance is strictly enforced to begin and end with `<sil>` to account for pre- and post-speech silence in natural recordings.
10. **Strict Validation:** Any unexpected symbol or digit triggers an explicit `ValueError`. No character is dropped silently.

---

## 4. Grapheme-to-Phoneme (G2P)

Implemented in `preprocessing/g2p.py`.

### 1. What is it?
**Grapheme-to-Phoneme (G2P)** is the process of converting written characters (graphemes) into pronunciation symbols (phonemes).

### 2. Why do we need it?
English orthography is notoriously non-phonetic. For example:
* `"cough"` (/k AO f/)
* `"tough"` (/t AH f/)
* `"through"` (/th R UW/)
* `"bough"` (/b AW/)
The sequence of letters `"ough"` has completely different acoustic realizations in each word. Feeding raw characters directly into an acoustic model forces the model to memorize complex spelling exceptions rather than acoustic acoustics.

### 3. Exact Implementation Resolution Hierarchy:
For every word token:
1. **Acronym Check:** If prefixed with `#`, look up letter names in a fixed dictionary (e.g., `'a'` $\rightarrow$ `EY1`, `'b'` $\rightarrow$ `B IY1`).
2. **CMU Pronouncing Dictionary Lookup:** If the word exists in CMUdict, retrieve its primary pronunciation.
3. **Possessive Suffix Rule:** If the word ends in `'s` and the base word exists in CMUdict, apply standard English plural/possessive morphophonemic rules:
   * After sibilants (`S`, `Z`, `SH`, `ZH`, `CH`, `JH`) $\rightarrow$ append `IH0 Z`.
   * After voiceless consonants (`P`, `T`, `K`, `F`, `TH`) $\rightarrow$ append `S`.
   * After all other voiced sounds $\rightarrow$ append `Z`.
4. **Out-of-Vocabulary (OOV) Fallback:** For novel words or rare names, use the `g2p_en` neural fallback (a pre-trained recurrent neural network) to deterministically predict ARPAbet phonemes.

### 4. Phonetic Inventory
* **39 ARPAbet Phonemes:**
  `AA, AE, AH, AO, AW, AY, B, CH, D, DH, EH, ER, EY, F, G, HH, IH, IY, JH, K, L, M, N, NG, OW, OY, P, R, S, SH, T, TH, UH, UW, V, W, Y, Z, ZH`
* **1 Silence Marker:** `SIL` (assigned index `0`).
* **Total Symbols:** Exactly **40 discrete classes**.
* **Stress Digit Stripping:** Lexical stress markers (`0`, `1`, `2`) are stripped for monophone HMM modeling (e.g., `AH0`, `AH1`, `AH2` all collapse to `AH`). This prevents splitting data across three sparse distributions.

---

## 5. Forced Alignment via Montreal Forced Aligner (MFA)

### 1. What is Forced Alignment?
Forced alignment is the automated process of taking an audio file and its known phonetic transcript, and determining the exact start and end timestamps (in seconds) of every phoneme in the recording.

### 2. Why do we need it?
To train an HMM in a supervised fashion, we must know which specific frames of audio belong to which phoneme. Manually segmenting 24 hours of audio is impossible. Forced alignment uses a pre-existing alignment model to generate these timestamps automatically.

### 3. What does MFA produce?
MFA outputs a Praat **TextGrid** file for each audio clip. A TextGrid contains time-interval tiers indicating the start time ($t_0$), end time ($t_1$), and phone label for every segment.

### 4. Mapping Timestamps to Acoustic Frames
In `preprocessing/alignment.py`, time intervals are converted to discrete frame boundaries using the sampling rate ($f_s = 22,050\text{ Hz}$) and hop length ($H = 256$ samples):
$$\text{start\_frame} = \left\lfloor \frac{t_0 \times f_s}{H} \right\rfloor = \left\lfloor \frac{t_0 \times 22,050}{256} \right\rfloor$$
The final boundary of the last segment is strictly set to $T = 1 + \lfloor \text{n\_samples} / H \rfloor$ to ensure a complete, non-overlapping partition of all $T$ frames.

### 5. Critical Distinction: Alignment HMM vs. Synthesis HMM
> [!IMPORTANT]
> **Common Viva Question:** *"Is the MFA model the same HMM that you trained for synthesis?"*
> **Answer:** **No.** 
> * **MFA Model:** A pre-trained, speaker-independent Kaldi-based triphone HMM-GMM acoustic model (`english_us_arpa`) used solely offline as an alignment tool to extract time boundaries.
> * **Our V1 HMM:** A single-speaker, 3-state left-to-right monophone HMM trained from scratch on the extracted frame boundaries to synthesize 80-D log-Mel spectrograms.

---

## 6. Short-Time Fourier Transform (STFT)

Speech is non-stationary—its frequency content changes continuously over time. The **Short-Time Fourier Transform (STFT)** addresses this by applying the Fourier transform to short, overlapping windowed segments where the speech signal is approximately quasi-stationary.

### Exact Parameter Configuration (`preprocessing/config.py`):
* **Sampling Rate ($f_s$):** 22,050 Hz.
* **FFT Window Size ($N_{\text{fft}}$):** 1024 samples.
  $$\text{Window Duration} = \frac{1024}{22,050} \approx 46.44\text{ ms}$$
* **Window Function:** Hann window ($w[n] = 0.5 - 0.5 \cos(2\pi n / N)$) to minimize spectral leakage at window edges.
* **Hop Length ($H$):** 256 samples.
  $$\text{Frame Shift (Hop Duration)} = \frac{256}{22,050} \approx 11.61\text{ ms}$$
* **Frame Overlap:** $1024 - 256 = 768$ samples (75% overlap).
* **Center Padding:** `center=True` (reflect padding of $N_{\text{fft}} // 2 = 512$ samples on each side) so that frame $t$ is centered precisely at sample $t \times H$.

### Frequency Resolution:
An $N_{\text{fft}} = 1024$ point transform produces $1 + N_{\text{fft}} / 2 = 513$ positive linear frequency bins, with a frequency resolution of:
$$\Delta f = \frac{f_s}{N_{\text{fft}}} = \frac{22,050}{1024} \approx 21.53\text{ Hz per bin}$$

---

## 7. Mel Spectrogram Representation

### 1. What is the Mel Scale?
Human pitch perception is non-linear. The human ear has high frequency resolution at low frequencies (below 1,000 Hz) and poor frequency resolution at high frequencies. The **Mel Scale** is a perceptual scale of pitches judged by listeners to be equal in distance from one another.

### 2. Mel Filterbank Construction
We compress the 513 linear STFT magnitude bins into **80 Mel bins** using triangular bandpass filters:
* Lower Frequency Limit: $f_{\text{min}} = 0.0\text{ Hz}$.
* Upper Frequency Limit: $f_{\text{max}} = 8,000.0\text{ Hz}$.
* Normalization: Slaney normalization (area-normalized triangular filters so each filter has unit area).

### 3. Log-Magnitude Transformation
The linear Mel energies are converted to a logarithmic scale:
$$\mathbf{m}_t = \ln(\max(\text{Mel}_t, 10^{-5}))$$
* **Why Logarithm?** Human perception of sound loudness is logarithmic (Weber-Fechner law). Furthermore, taking the log transforms multiplicative acoustic effects (source-filter convolution) into additive linear components.
* **Why Clamp to $10^{-5}$?** Prevents taking $\ln(0)$, which would produce negative infinity ($-\infty$) and destroy mathematical operations.

---

## 8. Why 80-D Log-Mel?

1. **Dimensionality Reduction:** Compresses 513 linear STFT bins into 80 meaningful acoustic features per frame (an 84.4% reduction in parameter space).
2. **Perceptual Alignment:** Emphasizes lower-frequency formant structures crucial for vowel identification while smoothly grouping high-frequency noise.
3. **Gaussian Suitability:** Log-spectral energies follow approximately Gaussian distributions, making them ideal targets for HMM single-Gaussian emissions.
4. **Why NOT PCA in V1?** Principal Component Analysis (PCA) performs linear dimensionality reduction across orthogonal axes of maximum variance, but it destroys the local spatial continuity of spectral formants and complicates vocoder inversion. 80-D log-Mel filterbanks are directly invertible via linear pseudo-inverses.

---

## 9. Frame-to-State Mapping (The 3 Temporal Sub-States)

Each phoneme interval identified by MFA is partitioned into **3 sequential temporal sub-states**:
* **State 0:** Beginning temporal state (transition into the phoneme).
* **State 1:** Middle temporal state (steady-state core).
* **State 2:** End temporal state (transition out of the phoneme).

> [!WARNING]
> **Terminology Warning:** Do NOT call these states "onset, nucleus, coda." In linguistic phonology, onset, nucleus, and coda refer to sub-syllabic parts of an entire syllable (e.g., in "cat", /k/ is onset, /ae/ is nucleus, /t/ is coda). In our HMM, every single phoneme (including stop consonants like /t/) has 3 temporal sub-states. Refer to them strictly as **beginning, middle, and end temporal sub-states**.

### Exact Frame Partition Formula:
If a phoneme interval spans $n$ frames (from frame index $0$ to $n-1$), frame $i$ is assigned to state $s$ via integer division:
$$s(i) = \left\lfloor \frac{3 \times i}{n} \right\rfloor \quad \text{for } i \in \{0, 1, \dots, n-1\}$$

### Handling Short Intervals ($n < 3$ frames):
When a phoneme duration is very short:
* If $n = 1$: $s(0) = \lfloor 3 \times 0 / 1 \rfloor = 0$. Only State 0 receives a frame; States 1 and 2 receive 0 frames.
* If $n = 2$: $s(0) = 0$, $s(1) = 1$. States 0 and 1 receive 1 frame each; State 2 receives 0 frames.
The implementation counts and logs these empty state occurrences transparently rather than fabricating artificial frames.

---

## 10. V1 Hidden Markov Model (HMM) Architecture

### Formal Definition from First Principles
A Hidden Markov Model is defined by the 5-tuple $\lambda = (S, V, A, B, \pi)$:
1. **$S = \{s_0, s_1, s_2\}$:** A set of discrete hidden states representing temporal progression.
2. **$V = \mathbb{R}^{80}$:** A continuous observation space (80-D log-Mel vectors).
3. **$A$:** State transition probability matrix. In our strictly left-to-right topology with no skip transitions, the transition graph is:
   $$s_0 \rightarrow s_1 \rightarrow s_2$$
4. **$B$:** Continuous emission probability distribution $b_j(\mathbf{o}_t) = P(\mathbf{o}_t \mid q_t = s_j)$, modeled as a single multivariate Gaussian with diagonal covariance.
5. **$\pi$:** Initial state distribution, strictly fixed to $\pi = [1.0, 0.0, 0.0]$ (synthesis always begins at State 0).

### Overall Model Structure
* 40 phonetic classes $\times$ 3 states per phone = **120 distinct HMM states**.

---

## 11. Gaussian Emission Probability Distributions

Each of the 120 states models its acoustic output as an 80-dimensional multivariate Gaussian:
$$\mathbf{o}_t \sim \mathcal{N}(\boldsymbol{\mu}_{p, s}, \boldsymbol{\Sigma}_{p, s})$$

### Why Diagonal Covariance?
Full covariance matrices require estimating $80 \times 80 = 6,400$ covariance parameters per state ($120 \times 6,400 = 768,000$ parameters), which risks singular matrices, overfitting, and severe computational cost. Diagonal covariance assumes that the 80 Mel bins are conditionally independent given the hidden state:
$$\boldsymbol{\Sigma}_{p, s} = \text{diag}(\sigma_{p, s, 1}^2, \sigma_{p, s, 2}^2, \dots, \sigma_{p, s, 80}^2)$$
This requires estimating only 80 variance values per state ($120 \times 80 = 9,600$ parameters).

### Variance Flooring
If a state has few observations or static values in certain bins, the empirical variance $\sigma_d^2$ could approach zero, causing numerical explosion in Gaussian density calculations ($1 / \sigma_d$). We enforce a strict variance floor:
$$\sigma_{p, s, d}^2 \ge 10^{-6}$$

---

## 12. Supervised Parameter Estimation (Why NOT Baum-Welch?)

> [!IMPORTANT]
> **Crucial Viva Question:** *"How did you train the HMM? Did you use the Baum-Welch (EM) algorithm?"*
> **Answer:** **No.** Baum-Welch is an Expectation-Maximization algorithm used when frame-to-state alignments are unknown (unsupervised/latent). Because Montreal Forced Aligner provided exact frame boundaries, our frame-to-state alignments were **fully known and supervised**. Therefore, parameters were estimated directly via closed-form **Maximum Likelihood Estimation (MLE)**.

### Mathematical Formulation of Sufficient Statistics:
For each state $(p, s)$, the training script (`models/train_hmm.py`) iterates across all 11,790 training utterances, accumulating:
1. Total observation count: $N_{p, s} = \sum_{i} 1$
2. Linear sum vector (64-bit float): $\mathbf{S}_{p, s} = \sum_{t \in (p, s)} \mathbf{o}_t$
3. Squared sum vector (64-bit float): $\mathbf{S2}_{p, s} = \sum_{t \in (p, s)} \mathbf{o}_t^2$

The closed-form MLE solutions are computed in a single pass:
$$\boldsymbol{\mu}_{p, s} = \frac{\mathbf{S}_{p, s}}{N_{p, s}}$$
$$\boldsymbol{\sigma}_{p, s}^2 = \max\left( \frac{\mathbf{S2}_{p, s}}{N_{p, s}} - \boldsymbol{\mu}_{p, s}^2, \, 10^{-6} \right)$$

---

## 13. Duration Modeling

HMM state transitions are memoryless (geometric duration distributions), which causes standard HMMs to generate unrealistically short, exponential phoneme durations. To overcome this, V1 uses an explicit **duration model**:
* During training, for every phoneme occurrence, the number of frames allocated to each sub-state is recorded.
* The model computes and stores:
  * `duration_mean` ($\bar{d}_{p, s}$)
  * `duration_std` ($\sigma_{dur}(p, s)$)
  * `duration_min` and `duration_max`
  * `phone_total_duration_mean` ($\sum_{s=0}^2 \bar{d}_{p, s}$)
* At synthesis time, state duration is predicted deterministically:
  $$d_{p, s} = \max(1, \lfloor \text{round}(\bar{d}_{p, s}) \rfloor)$$

---

## 14. V1 Inference Step-by-Step Walk-Through

For an unseen sentence: `"Machine learning synthesizes speech."`

```
1. Text Input:
   "Machine learning synthesizes speech."
   ↓
2. Text Normalization:
   Tokens: ['<sil>', 'machine', 'learning', 'synthesizes', 'speech', '<sil>']
   ↓
3. G2P Conversion:
   Phonemes: ['SIL', 'M', 'AH', 'SH', 'IY', 'N', 'L', 'ER', 'N', 'IH', 'NG', ...]
   ↓
4. State & Duration Lookup:
   For each phone P:
     State 0: allocate d_0 = round(mean_dur[P, 0]) frames
     State 1: allocate d_1 = round(mean_dur[P, 1]) frames
     State 2: allocate d_2 = round(mean_dur[P, 2]) frames
   Total Utterance Length = T = sum(d) frames
   ↓
5. Direct Acoustic Emission:
   For each state (P, s), replicate its 80-D mean vector mu[P, s] for d_{P,s} frames.
   Result: A [T, 80] Log-Mel Spectrogram Matrix.
   ↓
6. Acoustic Inversion:
   Linear Mel = exp(Log-Mel)
   Linear Magnitude = INV_MEL_BASIS @ Linear Mel   [513, T]
   ↓
7. Phase Estimation (Fast Griffin-Lim):
   Iterate 60 times with momentum 0.99 to reconstruct phase.
   ↓
8. Audio Output:
   22,050 Hz Mono WAV audio file.
```

---

## 15. Why V1 Produces Piecewise-Constant Trajectories

### The Mathematical Cause of "Stair-Steps"
In V1, the acoustic feature generation for frame $t$ is formulated as:
$$\hat{\mathbf{c}}_t = \arg\max_{\mathbf{c}_t} \mathcal{N}(\mathbf{c}_t \mid \boldsymbol{\mu}_{q_t}, \boldsymbol{\Sigma}_{q_t}) = \boldsymbol{\mu}_{q_t}$$
Because the mean vector $\boldsymbol{\mu}_{q_t}$ is identical for every frame within the same state, the resulting trajectory is a flat, horizontal line for the entire duration of that state.

```
Log-Mel Energy (dB)
     ▲
     │              ┌────────────┐ State 1 (Middle)
     │              │ (Flat Mean)│
     │              │            │
     │  ┌───────────┘            └───────────┐ State 2 (End)
     │  │ State 0                            │
     │  │ (Flat Mean)                        │
     └──┴────────────────────────────────────┴────────► Time (Frames)
        ▲           ▲            ▲
        State Boundary 1         State Boundary 2
        (Instantaneous Jump)     (Instantaneous Jump)
```

### Consequences on Audio Quality:
1. **Zero Articulatory Motion:** Real speech formants are constantly moving curves. Flat trajectories sound synthetic, buzzy, and robotic.
2. **Abrupt Boundary Jumps:** When transitioning from State 0 to State 1, the spectrum jumps instantaneously in a single frame hop (11.6 ms), creating harsh phase-inversion artifacts in the Griffin-Lim vocoder.

---

## 16. Mel-to-Waveform Reconstruction (Vocoding)

Because the model predicts 80-D log-Mel spectrograms rather than time-domain audio samples, an inversion process is required:

### 1. Pseudo-Inverse Mel Basis
The 80-channel Mel filterbank matrix $M$ has shape $[80, 513]$. Because $M$ is non-square ($80 < 513$), it has no direct matrix inverse. We compute the Moore-Penrose pseudo-inverse $M^+$ (`INV_MEL_BASIS = np.linalg.pinv(MEL_BASIS)`):
$$M^+ = M^T (M M^T)^{-1} \quad \text{shape: } [513, 80]$$
The approximate linear STFT magnitude matrix $\hat{S}$ is:
$$\hat{S} = \max(M^+ \cdot \exp(\mathbf{M}_{\text{log}}), \, 10^{-8}) \quad \text{shape: } [513, T]$$

### 2. Fast Griffin-Lim Algorithm
The STFT magnitude $|\hat{S}|$ contains no phase information. Inverting $|\hat{S}|$ via standard inverse STFT (iSTFT) requires phase angles $\angle S$. 

The **Griffin-Lim Algorithm** iteratively estimates phase by alternating between time-domain projection and frequency-domain constraint enforcement:
1. Start with random initial phase.
2. Synthesize time-domain signal: $x^{(i)} = \text{iSTFT}(|\hat{S}| e^{j \phi^{(i)}})$.
3. Re-analyze signal via STFT: $X^{(i)} = \text{STFT}(x^{(i)})$.
4. Discard resulting magnitude and replace with known target magnitude $|\hat{S}|$, retaining updated phase:
   $$\tilde{X}^{(i+1)} = |\hat{S}| \frac{X^{(i)}}{|X^{(i)}|}$$
5. Apply momentum acceleration (momentum = 0.99) across 60 iterations (`random_state=42`).

---

## 17. V1 Empirical Evaluation Setup

To evaluate V1 rigorously, two experimental evaluations were performed:
1. **Preliminary Benchmark (Gate 7):** A representative 20-utterance stratified subset (short, medium, long) used during initial development to verify pipeline integrity.
2. **Final Test Benchmark:** The complete, untouched **655-utterance held-out test split** (`data_splits/test.txt`).

---

## 18. V1 Quantitative Results (655-Utterance Final Test Set)

| Metric | Measured Value | Detailed Interpretation |
| :--- | :--- | :--- |
| **Whisper-tiny WER (%)** | **78.78%** | Extremely high error rate. Whisper misidentifies the majority of words due to severe boundary buzz and robotic phoneme distortion. |
| **DCT-Derived MCD (dB)** | **40.95 dB** | Substantial spectral envelope distortion relative to natural speech recordings. |
| **Boundary Discontinuity** | **0.6136** | High average step magnitude at state transitions, directly confirming the stair-step problem. |
| **First-Order Change ($\bar{d}_1$)** | **0.2359** | Reflects flat plateaus punctuated by sharp boundary jumps. |
| **Second-Order Change ($\bar{d}_2$)** | **0.4661** | Extremely high acceleration/curvature spikes caused by square-wave state transitions. |
| **Global Variance (GV)** | **1.4949** | Baseline spectral variance across the test set. |
| **Duration Prediction Error** | **8.5%** | Highly accurate duration modeling; the duration model predicts realistic utterance lengths. |

---

## 19. V1 Strengths and Limitations

### Strengths:
* **Fully Syllabus-Aligned:** 100% compliant with foundational machine learning principles (HMM, MLE, sufficient statistics).
* **Supervised & Closed-Form:** Requires zero iterative gradient descent; trains across 11,790 utterances in minutes.
* **Deterministic Inference:** Synthesis is fully reproducible.
* **Accurate Durations:** 8.5% duration error matches human speech tempo.

### Limitations (Motivating Version 2):
* **Piecewise-Constant Generation:** Generates flat horizontal spectral blocks with unnatural boundary jumps.
* **Frame Independence Assumption:** Treats each acoustic frame independently, ignoring physical articulatory velocity and momentum.
* **Robotic Acoustic Quality:** The Griffin-Lim vocoder produces metallic phase buzz when presented with square-wave spectral jumps.

---

## 20. Comprehensive V1 Viva Questions & Answers

### Q1: Why did you divide each phoneme into 3 states rather than 1 or 5?
**Answer:** In speech synthesis, a single phoneme is dynamic. A single state would assume a phoneme is completely static. Dividing each phone into 3 states captures the three fundamental temporal phases: transition in (beginning), steady-state target (middle), and transition out (end). Using 5 states would cause data sparsity for short phonemes (which often span fewer than 5 frames).

### Q2: Why is the transition matrix $A$ not parameterized with learned probabilities?
**Answer:** Because our V1 system uses an explicit duration model. Standard HMM transition self-loop probabilities $a_{ii}$ enforce a memoryless geometric duration distribution $P(d) = (1 - a_{ii}) a_{ii}^{d-1}$, which is physically inaccurate for human speech. Using an explicit duration lookup is the established standard in parametric synthesis.

### Q3: Why didn't you use the Forward-Backward / Baum-Welch algorithm?
**Answer:** Baum-Welch is designed for unsupervised training when hidden state sequences are unknown. In our pipeline, forced alignment with MFA provided explicit, supervised frame-to-state boundaries. When state sequences are observed, Maximum Likelihood Estimation reduces to direct sample means and sample variances, which is mathematically exact and avoids local optima.

### Q4: What does the variance floor do?
**Answer:** The variance floor ($10^{-6}$) prevents numerical instability. If a frequency bin has zero variance across training frames, the Gaussian probability density formula would divide by zero.

### Q5: What is the primary bottleneck in V1 speech quality?
**Answer:** Direct mean emission. Repeating the 80-D mean vector across all frames of a state produces a piecewise-constant trajectory with abrupt jumps at state boundaries, resulting in unnatural, robotic audio. This bottleneck directly motivates Version 2.
