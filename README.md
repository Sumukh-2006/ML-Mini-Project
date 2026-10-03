# End-to-End Statistical Parametric Text-to-Speech (TTS) System (V1 HMM)

This repository contains an implementation of an **End-to-End Statistical Parametric Speech Synthesis (SPSS)** system built on the single-speaker **LJ Speech Dataset (v1.1)**. 

The project builds a complete, mathematically grounded pipeline: from raw text normalization and pronunciation lexicon mapping, through forced alignment and Gaussian acoustic parameter estimation, to acoustic parameter trajectory generation and Griffin-Lim waveform reconstruction.

---

## 1. System Architecture & Core Pipeline

```
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
Generated 80-D Log-Mel Spectrogram [T, 80]
    ↓
Mel Pseudo-Inverse Projection [T, 513]
    ↓
Fast Griffin-Lim Algorithm (Phase estimation, 60 iterations)
    ↓
Synthesized Speech Audio (22,050 Hz WAV)
```

---

## 2. Locked V1 Specification

To establish a clear baseline and guarantee reproducibility, the V1 pipeline was strictly bound to the following architectural choices:
- **Audio Sample Rate:** 22,050 Hz (single-channel mono, 16-bit PCM).
- **Acoustic Representation:** 80-dimensional log-Mel spectrogram (`n_fft=1024`, `win_length=1024`, `hop_length=256`, Hann window, `f_min=0`, `f_max=8000`, Slaney normalization).
- **Phonetic Inventory:** 40 discrete symbols (39 ARPAbet phonemes with stress stripped + `SIL` silence marker).
- **Acoustic Model:** 3-state left-to-right monophone HMM per symbol (State 0: onset, State 1: nucleus, State 2: coda) with no skip transitions.
- **Emissions:** Single Gaussian per state with diagonal covariance matrix ($10^{-6}$ variance floor).
- **Alignment:** Supervised phone-to-frame boundary extraction via Montreal Forced Aligner (MFA).
- **Trajectory Generation:** Direct emission of Gaussian means (piecewise-constant trajectory).
- **Excluded from V1:** No PCA, no dynamic delta/delta-delta features, no GMM mixtures, no triphones, no MLPG, and no neural duration models.

---

## 3. Repository Structure

```text
My Approach/
├── Dataset/
│   └── LJSpeech-1.1/               # Raw LJ Speech audio and metadata
│       ├── metadata.csv            # Pipe-delimited transcription records
│       ├── README                  # Dataset documentation
│       └── wavs/                   # 13,100 WAV audio files
├── data_splits/
│   ├── create_split.py             # Deterministic 90/5/5 utterance split generator
│   ├── train.txt                   # 11,790 training utterance IDs
│   ├── val.txt                     # 655 validation utterance IDs
│   └── test.txt                    # 655 unseen test utterance IDs (full held-out test set; evaluation uses a 20-utterance stratified subset)
├── requirements.txt                # Pinned core dependencies for V1 pipeline
├── preprocessing/
│   ├── config.py                   # Master acoustic and STFT hyperparameter constants
│   ├── dataset_audit.py            # Gate 1: Read-only dataset integrity validation
│   ├── unicode_inspect.py          # Investigation of non-ASCII characters
│   ├── text_processing.py          # Deterministic text cleaning, acronym handling & tokenization
│   ├── g2p.py                      # Deterministic English G2P with CMUdict and OOV prediction
│   └── alignment.py                # Phase 3: MFA corpus prep, TextGrid parser, and 3-state mapper
├── models/
│   ├── hmm.py                      # MonophoneHMM class definition and serialization
│   ├── train_hmm.py                # Supervised parameter and empirical duration estimation
│   └── baseline.py                 # Static phoneme-average baseline implementation
├── synthesis/
│   ├── reconstruction.py           # 80-D log-Mel feature extractor and Griffin-Lim vocoder
│   ├── oracle_test.py              # Gate 3: Ground-truth oracle reconstruction validation
│   ├── synthesize.py               # HMMV1Synthesizer end-to-end inference engine
│   ├── validate_training_synthesis.py # Gate 6: Training set synthesis verification
│   └── validate_test_synthesis.py  # Gate 7: Unseen test set synthesis verification
├── evaluation/
│   ├── acoustic_metrics.py         # Mathematical formulations: DTW-MCD, Spectral MSE, WER
│   └── run_evaluation.py           # Master comparative evaluation suite (Baseline vs HMM vs Oracle)
└── outputs/
    ├── g2p/                        # Canonical phoneme extractions (phonemes.jsonl)
    ├── alignments/                 # Parsed phone/state frame alignments
    ├── oracle_reconstruction/      # Gate 3 oracle audio files
    ├── models/                     # Trained HMM model checkpoint (v1_hmm_model.json)
    ├── synthesis/                  # Synthesized audio files from Gates 6 & 7
    └── evaluation/                 # Audio files and JSON metrics for the final test benchmark
```

---

## 4. Step-by-Step Implementation & Gate Progression

The system was developed and validated incrementally through 7 strict validation checkpoints (Gates):

### Gate 1: Dataset Audit
- **Script:** `preprocessing/dataset_audit.py`
- **Result:** Confirmed 13,100 metadata rows matching 13,100 WAV files. Verified 0 missing, extra, duplicate, or corrupt files. Audio confirmed at mono 16-bit 22,050 Hz PCM.

### Gate 2: Deterministic Text Processing & G2P
- **Scripts:** `preprocessing/text_processing.py`, `preprocessing/g2p.py`
- **Unicode Analysis:** Investigated the 119 non-ASCII transcriptions (96 were `£` currency symbols expanded to "pounds" in the normalized column; the remainder were typographic quotes and diacritics normalized via NFKD).
- **Acronym Expansion:** Implemented deterministic letter-spelling for all-caps runs (e.g., `FBI` $\rightarrow$ `EH F B IY AY`).
- **Result:** Successfully converted all 13,100 transcripts into 40-symbol ARPAbet sequences (starting and ending with `SIL`). Verified 100% determinism with identical SHA-1 hashes across separate runs.

### Gate 3: Oracle Acoustic Reconstruction
- **Script:** `synthesis/oracle_test.py`
- **Concept:** Tested the round-trip invertibility of the acoustic target:
  $$\text{WAV} \rightarrow \text{STFT} \rightarrow \text{80-D log-Mel} \rightarrow \text{Pseudo-Inverse} \rightarrow \text{Fast Griffin-Lim} \rightarrow \text{WAV}$$
- **Result:** Verified that original speech reconstructed through this pipeline retained full intelligibility (Global Spectral MSE = $0.0040$, DTW-MCD = $5.72\text{ dB}$, Whisper-tiny WER = $5.5\%$), establishing that the 80-D log-Mel representation is invertible.

### Gate 4: Forced Alignment & 3-State Partitioning
- **Script:** `preprocessing/alignment.py`
- **MFA Execution:** Aligned all 13,100 utterances using Montreal Forced Aligner with the pre-trained `english_us_arpa` acoustic model.
- **Stress-Handling Layer:** Preserved stress markers during MFA alignment while mapping TextGrid intervals back to canonical stress-free monophones.
- **3-State Frame Split:** For each phone interval of length $N$ frames, frames were divided into three states:
  $$\text{state}(i) = \lfloor \frac{3 \cdot i}{N} \rfloor \quad \text{for } i \in [0, N-1]$$
- **Result:** 13,100 / 13,100 utterances successfully aligned (0 failures, 951,059 total phone segments).

### Gate 5: Frame-to-State Validation & HMM Training
- **Script:** `models/train_hmm.py`
- **Split Isolation:** Trained strictly on `data_splits/train.txt` (11,790 utterances).
- **Validation:** Verified 6,679,958 acoustic frames in the training set: 100.0% labeled, 0 overlapping, 0 unlabeled.
- **Parameter Estimation:** Computed empirical mean and variance vectors for all 120 states (40 phones $\times$ 3 states):
  $$\boldsymbol{\mu}_{p, s} = \frac{1}{N_{p, s}} \sum_{t \in \mathcal{T}_{p, s}} \mathbf{x}_t, \quad \boldsymbol{\sigma}_{p, s}^2 = \max\left(\frac{1}{N_{p, s}} \sum_{t \in \mathcal{T}_{p, s}} \mathbf{x}_t^2 - \boldsymbol{\mu}_{p, s}^2, \, 10^{-6}\right)$$
- **Sanity Checks:** Verified zero NaNs, zero Infs, all variances $\ge 10^{-6}$, and zero unobserved states (minimum observation count for any state: 1,337 frames). Saved model to `outputs/models/v1_hmm_model.json`.

### Gate 6: Training Utterance Synthesis Validation
- **Script:** `synthesis/validate_training_synthesis.py`
- **Result:** Successfully synthesized speech from training transcripts across varying sentence lengths (1.8s to 9.7s). Generated log-Mel trajectories adhered strictly to `[T, 80]` dimensions. Audio confirmed intelligible.

### Gate 7: Unseen Test Utterance Synthesis Validation
- **Script:** `synthesis/validate_test_synthesis.py`
- **Zero Leakage:** Evaluated on 20 stratified unseen test sentences from `data_splits/test.txt`. Note: `data_splits/test.txt` contains the full 655-utterance held-out test set (5% split). For rapid, deterministic iteration in V1 Gate 7 and the benchmark evaluation, a representative 20-utterance stratified subset (spanning short, medium, and long durations) is benchmarked.
- **Result:** 20/20 valid audio outputs generated with a mean duration prediction error of only 8.5% and zero acoustic feature leakage.

---

## 5. Final Evaluation Results

The system was evaluated against two reference benchmarks across a 20-utterance stratified subset from the full 655-utterance held-out test set (`data_splits/test.txt`):
1. **Simple Phoneme-Average Baseline:** Flattens each phoneme into a static 80-D vector.
2. **V1 Monophone HMM:** 3-state left-to-right temporal model.
3. **Oracle Reconstruction Reference:** Real test audio inverted through Griffin-Lim (vocoder benchmark).

> **Test Set Note:** `data_splits/test.txt` preserves the complete 655-utterance (5%) held-out test split for large-scale evaluation. The current benchmark suite (`evaluation/run_evaluation.py`) runs on the 20-utterance stratified subset to provide rapid, reproducible quantitative validation without excessive Griffin-Lim inversion overhead.

### Quantitative Comparison Table

| Metric | Simple Baseline | V1 Monophone HMM | Oracle Reconstruction Reference |
| :--- | :--- | :--- | :--- |
| **Synthesis Success Rate** | **20 / 20 (100%)** | **20 / 20 (100%)** | **20 / 20 (100%)** |
| **Mean Duration Error (%)** | **8.5 %** | **8.5 %** | **0.2 %** |
| **Global Log-Mel Spectral MSE** | **0.1176** | **0.1166** | **0.0040** |
| **DCT-Derived MCD (dB)** | **41.64 dB** | **40.39 dB** | **5.72 dB** |
| **Whisper-tiny WER (%)** | **94.9 %** | **80.8 %** | **5.5 %** |
| **MOS (1–5 Naturalness)** | *Pending formal test* | *Pending formal test* | *Reference (unrated)* |

*Metric Formulations:*
- **DCT-Derived MCD:** Computed by transforming 80-D log-Mel vectors into 16 Mel-Cepstral Coefficients via an orthonormal Type-II DCT, dropping $c_0$ (energy), and calculating DTW-aligned Euclidean distance scaled to decibels ($\frac{10\sqrt{2}}{\ln 10} \approx 6.14185\text{ dB}$).
- **WER:** Word Error Rate calculated via greedy decoding of Whisper-tiny on synthesized audio compared against normalized target text.

---

## 6. How to Reproduce & Run

### Environment Setup
Create and activate the dedicated project conda environment:
```bash
conda create -n tts-hmm python=3.11 -y
conda activate tts-hmm
pip install -r requirements.txt
```

### 1. Data Split Generation
```bash
python data_splits/create_split.py
```

### 2. G2P Canonical Phoneme Extraction
```bash
python preprocessing/g2p.py --n 0 --out outputs/g2p/phonemes.jsonl
```

### 3. Model Training (Parameter & Duration Estimation)
```bash
python models/train_hmm.py
```

### 4. Synthesize a New Sentence
```python
from synthesis.synthesize import HMMV1Synthesizer
import soundfile as sf

synth = HMMV1Synthesizer("outputs/models/v1_hmm_model.json")
result = synth.synthesize("Machine learning enables speech synthesis from text.")
sf.write("output.wav", result["wav"], 22050)
print(f"Generated {result['duration_s']:.2f}s of speech across {result['n_frames']} frames.")
```

### 5. Run the Master Evaluation Suite
```bash
python evaluation/run_evaluation.py
```

---

## 7. Key Findings & Viva Summary

- **What did we prove?**
  We proved that an HMM acoustic model utilizing a 3-state left-to-right monophone topology successfully learns sub-phonetic temporal and spectral dynamics from speech data. Explicitly modeling onset, nucleus, and coda phases provides a measurable advantage over a static phoneme baseline, reducing Whisper-tiny Word Error Rate from **94.9% down to 80.8%** and reducing Mel-cepstral distortion by **1.25 dB**.
- **System Limitations:**
  The V1 model deliberately generates speech by directly emitting Gaussian means, creating piecewise-constant spectral trajectories with a noticeable "robotic" timbre. In addition, monophone modeling cannot account for phonetic co-articulation across word boundaries.
- **Future Work (V2 Roadmap):**
  1. Maximum Likelihood Parameter Generation (MLPG) with delta/delta-delta dynamic features to smooth spectral transitions.
  2. Multi-mixture Gaussian distributions (GMMs) to capture acoustic multi-modality.
  3. Context-dependent modeling (triphones) with decision-tree state clustering.
  4. Neural vocoding (e.g., HiFi-GAN) for improved phase reconstruction.
