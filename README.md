# End-to-End Statistical Parametric Text-to-Speech (TTS) System

## 1. Introduction and Problem Statement
This repository contains an implementation of an **End-to-End Statistical Parametric Speech Synthesis (SPSS)** system. The objective of this project is to model the highly complex relationship between written text and acoustic speech waveforms (Text $\rightarrow$ Speech) using foundational mathematical and statistical principles. 

Rather than treating the problem as a black box using modern Deep Learning (e.g., Transformers or Diffusion models), we built a classical HMM-based SPSS pipeline from scratch. This approach provides deep insights into phonetic alignments, acoustic feature spaces, duration modeling, and the mathematical constraints required to generate intelligible, natural-sounding speech trajectories. The project incrementally iterates on the architecture across two distinct versions (V1 and V2), methodically addressing the theoretical limitations of piecewise-constant trajectory generation using advanced linear algebra techniques.

## 2. Dataset Used
The model is trained and evaluated on the **LJ Speech Dataset (v1.1)**. 
- **Description:** A public domain speech dataset consisting of 13,100 short audio clips of a single female speaker reading passages from 7 non-fiction books.
- **Duration:** Approximately 24 hours of audio.
- **Format:** Single-channel mono, 16-bit PCM, 22,050 Hz sampling rate.
- **Link:** [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

## 3. Version-1 Approach Details

### V1 Specification & Pipeline
To establish a clear baseline and guarantee reproducibility, the V1 pipeline was strictly bound to the following architectural choices:
- **Audio Sample Rate:** 22,050 Hz (single-channel mono, 16-bit PCM).
- **Acoustic Representation:** 80-dimensional log-Mel spectrogram (`n_fft=1024`, `win_length=1024`, `hop_length=256`, Hann window, Slaney normalization).
- **Phonetic Inventory:** 40 discrete symbols (39 ARPAbet phonemes + `SIL` silence marker).
- **Acoustic Model:** 3-state left-to-right monophone HMM per symbol (State 0: beginning, State 1: middle, State 2: end temporal states) with no skip transitions.
- **Emissions:** Single Gaussian per state with diagonal covariance matrix ($10^{-6}$ variance floor).
- **Alignment:** Supervised phone-to-frame boundary extraction via Montreal Forced Aligner (MFA).

### V1 Trajectory Generation Constraint
In V1, the synthesis process relies on **direct emission of Gaussian means**. Once the duration of a phoneme state is predicted, the system generates the 80-D mean vector of that state repeatedly for the predicted number of frames. 
- **Result:** The generated log-Mel spectrogram consists of piecewise-constant segments (i.e., stair-steps). This causes severe discontinuities at state and phoneme boundaries, resulting in a robotic and degraded timbre when inverted via Griffin-Lim.

### V1 Architecture Diagram
```text
Text Input
    ↓
Text Normalization (Punctuation, casing, acronym expansion)
    ↓
G2P / Lexicon (CMU Pronouncing Dictionary + OOV handling)
    ↓
Phoneme Sequence (39 ARPAbet phones + SIL = 40 classes)
    ↓
State Duration Lookup (Empirical frame counts per state)
    ↓
Acoustic Generation (Emission of learned 80-D Gaussian Means)
    ↓
Generated 80-D Log-Mel Spectrogram [T, 80] (Piecewise-Constant)
    ↓
Fast Griffin-Lim Algorithm (Phase estimation, 60 iterations)
    ↓
Synthesized Speech Audio (22,050 Hz WAV)
```

## 4. Version-2 Approach Details

### V2 Enhancements: Dynamic Features & MLPG
The primary limitation of V1 was its robotic, disjointed output caused by spectral discontinuity at state boundaries. Version 2 solves this by enforcing temporal smoothness constraints across the entire utterance.

1. **240-D Dynamic Features:**
   During training, we append dynamic features to the static 80-D log-Mel spectrogram. Specifically, we compute:
   - **Delta ($\Delta$)**: First-order derivative (velocity) of the spectrum.
   - **Delta-Delta ($\Delta\Delta$)**: Second-order derivative (acceleration) of the spectrum.
   The resulting representation is a 240-D vector per frame (80 static + 80 delta + 80 delta-delta). The V2 HMM learns the means and variances for all 240 dimensions.

2. **Maximum Likelihood Parameter Generation (MLPG):**
   During synthesis, instead of greedily tiling the static mean, we solve for the continuous static trajectory $\hat{\mathbf{C}}$ that maximizes the joint likelihood of the static and dynamic distributions simultaneously. 
   - This optimization enforces that the generated sequence of frames $\hat{\mathbf{C}}$ mathematically respects the learned transitions (velocities and accelerations) of human speech.
   - We formulate this mathematically as a system of linear equations: $(W^T U^{-1} W) \mathbf{C} = W^T U^{-1} \boldsymbol{\mu}$. 
   - Because $W$ (the window matrix) is highly sparse, we construct the system as a **banded matrix** and solve it in $\mathcal{O}(T)$ linear time using Cholesky decomposition (`scipy.linalg.solveh_banded`).

### V2 Architecture Changes
- Replaced the direct 80-D mean emission with the 240-D MLPG solver.
- Substantially reduced the piecewise-constant spectral discontinuities.
- Dramatically improved the Griffin-Lim vocoder's ability to reconstruct phase due to the continuous spectral trajectories.

## 5. Comparison Study of V1 and V2

To objectively quantify the improvement, both the frozen V1 and frozen V2 models were evaluated on the **completely unseen 655-utterance held-out test set** (`data_splits/test.txt`).

### Quantitative Benchmarks (Test Set)

| Metric | V1 Monophone HMM | V2 Monophone HMM (MLPG) | Absolute Improvement | Bootstrap 95% CI |
| :--- | :--- | :--- | :--- | :--- |
| **Whisper-tiny WER (%) (ASR-based intelligibility proxy)** | 78.78% | **47.10%** | **-31.68 percentage points** | [-37.10, -27.54] percentage points |
| **DCT-derived cepstral distortion (MCD-style metric) (dB)** | 40.95 dB | **39.31 dB** | **-1.64 dB** | [-1.70, -1.59] dB |
| **Boundary Discontinuity** | 0.6136 | **0.1974** | **-68% reduction in "stair-step" jumps** | — |
| **Second-order Change (d2)** | 0.4661 | **0.0963** | **Significantly smoother trajectories** | — |
| **Solver Success Rate** | N/A | **100.0%** | **Numerically stable across all utterances** | — |

**Conclusion:** 
By correctly structuring the problem as an MLPG trajectory optimization using strict utterance-wide dynamic features and banded matrix solvers, the V2 architecture substantially reduced the state-boundary discontinuities observed in V1. The smoother acoustic trajectories produced by MLPG resulted in improved downstream reconstruction quality, reflected by lower WER and MCD on the completely held-out test set.

## 6. How to Reproduce and Run

### 1. Download the Dataset
```bash
mkdir -p Dataset
cd Dataset
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
cd ..
```

### 2. Environment Setup
Create and activate the dedicated project conda environment:
```bash
conda create -n tts-hmm python=3.11 -y
conda activate tts-hmm
pip install -r requirements.txt
```

### 3. Data Split Generation & Canonical Phoneme Extraction
```bash
# Creates train/val/test splits deterministically
python data_splits/create_split.py

# Extracts canonical phonemes using CMUdict (G2P)
python preprocessing/g2p.py --n 0 --out outputs/g2p/phonemes.jsonl
```

### 4. Model Training
```bash
# Train the V1 Baseline Model (80-D)
python models/train_hmm.py

# Train the V2 MLPG Model (240-D)
python models/train_hmm_v2.py
```

### 5. Final Master Evaluation (Test Set)
```bash
# Evaluates both V1 and V2 on the unseen 655-utterance test split
python evaluation/run_evaluation_test_final.py
```

## 7. Team Member Details
- **Sumukh Shandilya** - PES1UG24CS481, CSE H
- **Manoj Shrikant Gaonkar** -
PES1UG25CS827, CSE H
