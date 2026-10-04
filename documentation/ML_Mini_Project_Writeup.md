# Statistical Parametric Speech Synthesis: Monophone HMM Approach
**Authors:** Sumukh Shandilya (PES1UG24CS481, CSE H), Manoj Shrikant Gaonkar (PES1UG25CS827, CSE H)

## 1. Problem Statement
The objective of this project is to build an end-to-end Statistical Parametric Speech Synthesis (SPSS) pipeline capable of converting English text into an audible speech waveform. Rather than adopting modern end-to-end neural networks, this project intentionally implements a classical statistical machine learning approach based on Hidden Markov Models (HMMs). The primary challenge lies in mathematically modeling the temporal evolution of speech acoustic features and using these statistical models to synthesize intelligible audio from unseen text. The pipeline must handle text normalization, grapheme-to-phoneme conversion, acoustic feature extraction, statistical sequence modeling, and signal reconstruction.

## 2. Dataset Details
The models were trained and evaluated using the **LJ Speech Dataset**, a public domain speech corpus containing 13,100 short audio clips of a single speaker reading passages from non-fiction books.
*   **Total Duration:** ~24 hours of English speech.
*   **Audio Format:** 16-bit PCM, 22,050 Hz sampling rate, single channel (mono).
*   **Transcriptions:** Normalized text transcriptions are provided for every utterance.
*   **Splits:** The dataset was partitioned to ensure rigorous evaluation. Crucially, a held-out test set of **655 utterances** (`data_splits/test.txt`) was kept completely isolated during all phases of development, parameter tuning, and model selection. Validation and testing were performed exclusively on distinct subsets.

## 3. Approach Taken to Solve the Problem
The speech synthesis problem was framed as a probabilistic sequence generation task. A linguistic frontend processes text into a sequence of phonemes, which are then modeled by HMMs to generate acoustic representations (log-Mel spectrograms). A vocoder finally inverts these spectrograms into waveforms. Two distinct versions of the acoustic model were implemented and compared.

### Version 1: Static HMM (Baseline)
The initial approach modeled each phoneme using a 3-state left-to-right HMM, capturing the **beginning, middle, and end temporal sub-states**.
*   **Features:** 80-dimensional static log-Mel filterbank energies.
*   **Emission Probabilities:** Modeled via single-component Gaussian distributions per state.
*   **Synthesis:** During inference, the phoneme sequence is expanded into HMM states using the learned empirical state-duration statistics, and the corresponding learned Gaussian mean vectors are used to construct the acoustic trajectory.

### Version 2: Dynamic Features and MLPG (Improved)
V1 suffered from severe unnaturalness due to piecewise-constant spectral outputs (state-boundary discontinuities). V2 addressed this by incorporating dynamic features and global constraints.
*   **Features:** Extended to 240 dimensions by appending first-order (delta) and second-order (delta-delta) temporal derivatives to the 80 static features.
*   **Maximum Likelihood Parameter Generation (MLPG):** Instead of simply concatenating state means, V2 employs MLPG. This algorithm solves a global optimization problem to find the static feature trajectory that maximizes the likelihood of the generated sequence, subject to the explicit mathematical constraints between the static features and their derivatives.
*   **Solver:** The MLPG optimization is solved efficiently in $\mathcal{O}(T)$ time using a banded Cholesky solver, ensuring numerical stability.

## 4. Brief Overview of Implementation
*   **Linguistic Frontend:** Utilizes the Montreal Forced Aligner (MFA) to generate time-aligned phoneme sequences for the training data. A rule-based text normalizer and `g2p-en` provide phoneme sequences during inference.
*   **Acoustic Modeling:** Custom Python implementations using `numpy` and `scipy` are used for HMM parameter estimation, including state-wise Gaussian statistics and empirical duration statistics, as well as the MLPG solver.
*   **Vocoder:** The Griffin-Lim algorithm is utilized for phase reconstruction and waveform generation in both versions, holding the vocoder constant to isolate the impact of the acoustic model improvements.
*   **Evaluation Metrics:** Synthesis quality was objectively evaluated using Mel-Cepstral Distortion (MCD) to measure spectral distance, and Word Error Rate (WER) via an ASR-based intelligibility proxy (using a pre-trained Whisper model) to assess phonetic clarity.

## 5. Conclusions
The implementation successfully demonstrated the core principles of statistical parametric speech synthesis. The transition from V1 to V2 yielded substantial empirical improvements:
*   **Quantitative Results:** Evaluated on the strict 655-utterance test set, V1 achieved a WER of 78.78% and an MCD of 40.95 dB. V2 achieved a WER of 47.10% and an MCD of 39.31 dB.
*   **Analysis:** The incorporation of dynamic features and the MLPG algorithm in V2 **substantially reduced the state-boundary discontinuities** observed in V1. The smoother acoustic trajectories produced by MLPG resulted in improved downstream waveform reconstruction quality, reflected by lower WER and MCD.
*   **Impact:** This architectural improvement resulted in a massive 31.68 percentage-point drop in WER and a 1.64 dB reduction in MCD, demonstrating that explicitly modeling the temporal dynamics of speech can substantially improve statistical speech synthesis.
