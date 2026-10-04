# 03. Version 2 (V2) — Complete Pipeline, MLPG Formulation & Comparative Study

---

## 1. Motivation for Version 2

### 1. Starting from the Measured V1 Results
In Version 1, our 3-state monophone Hidden Markov Model successfully learned sub-phonetic acoustic distributions and achieved an 8.5% duration prediction error. However, when evaluated on the full 655-utterance held-out test set, V1 exhibited severe empirical limitations:
* **Word Error Rate (WER):** **78.78%** (Whisper-tiny failed to transcribe the majority of words).
* **Mel-Cepstral Distortion (MCD):** **40.95 dB** (large spectral mismatch against natural recordings).
* **Boundary Discontinuity:** **0.6136** (severe average step jump across state transitions).
* **Trajectory Curvature ($\bar{d}_2$):** **0.4661** (spiky acceleration profile caused by abrupt square-wave corners).

### 2. Identifying the Root Cause: The Piecewise-Constant Flaw
The primary failure of V1 was not data quantity or phonetic inventory. The failure was a fundamental mathematical limitation of **direct Gaussian mean emission**:
* In V1, the model emits $\hat{\mathbf{c}}_t = \boldsymbol{\mu}_{q_t}$ independently at every frame $t$.
* Because all frames assigned to a state share the exact same mean vector, the acoustic parameter trajectory is a flat, horizontal line that instantaneously jumps at state boundaries.
* Human vocal articulators (tongue, jaw, lips) cannot jump instantaneously—they possess physical mass and inertia. When Griffin-Lim attempts to reconstruct phase from a stair-step magnitude spectrogram, the unnatural boundary jumps produce severe metallic phase distortion and unintelligible speech.

### 3. Why Not Simply Increase Model Size or Data?
Adding more training data or increasing the number of phoneme states would not resolve this problem: tiling means from a larger dataset still produces flat stair-steps! To fix the root cause, the model must explicitly incorporate **temporal continuity and velocity constraints**.

---

## 2. The V2 Hypothesis

> [!NOTE]
> **Formal Hypothesis:**
> If the acoustic model learns not only static spectral features ($\mathbf{c}_t$) but also their temporal first derivatives (velocities $\Delta \mathbf{c}_t$) and second derivatives (accelerations $\Delta\Delta \mathbf{c}_t$), and synthesis is formulated as an utterance-wide joint trajectory optimization problem via **Maximum Likelihood Parameter Generation (MLPG)**, then:
> 1. The resulting static acoustic trajectory $\hat{\mathbf{C}}$ will be smooth and continuous.
> 2. Spectral boundary discontinuities will be substantially reduced.
> 3. Downstream vocoder phase reconstruction will improve, yielding lower Word Error Rate (WER) and lower Mel-Cepstral Distortion (MCD) on completely held-out test data.

---

## 3. Dynamic Features ($\Delta$ and $\Delta\Delta$)

### 1. What are Dynamic Features?
* **Static Features ($\mathbf{c}_t \in \mathbb{R}^{80}$):** The instantaneous log-Mel spectral envelope at frame $t$.
* **Delta Features ($\Delta \mathbf{c}_t \in \mathbb{R}^{80}$):** The first-order temporal derivative (acoustic velocity), representing the rate and direction of spectral change.
* **Delta-Delta Features ($\Delta\Delta \mathbf{c}_t \in \mathbb{R}^{80}$):** The second-order temporal derivative (acoustic acceleration), representing the curvature and rate of transition.

### 2. Exact Finite Difference Convolutions (`synthesis/mlpg.py`):
In speech synthesis literature, continuous derivatives are approximated via symmetric linear regression windows:
* **Delta Velocity Window:**
  $$\Delta \mathbf{c}_t = -0.5 \, \mathbf{c}_{t-1} + 0.5 \, \mathbf{c}_{t+1}$$
  Filter kernel: $w_1 = [-0.5, \, 0.0, \, 0.5]$.
* **Delta-Delta Acceleration Window:**
  $$\Delta\Delta \mathbf{c}_t = \mathbf{c}_{t-1} - 2.0 \, \mathbf{c}_t + \mathbf{c}_{t+1}$$
  Filter kernel: $w_2 = [1.0, \, -2.0, \, 1.0]$.

### 3. Utterance-Wide Computation & Edge-Padding Boundary Handling
Crucially, dynamic features are **computed utterance-wide** on the complete continuous recording *before* slicing frames into states:
$$\text{padded} = \text{np.pad}(\mathbf{C}, \, ((1, 1), (0, 0)), \, \text{mode}='edge')$$
* At the beginning of an utterance ($t = 0$), the preceding frame is replicated: $\mathbf{c}_{-1} = \mathbf{c}_0$.
  $$\Delta \mathbf{c}_0 = -0.5 \mathbf{c}_0 + 0.5 \mathbf{c}_1$$
  $$\Delta\Delta \mathbf{c}_0 = \mathbf{c}_0 - 2\mathbf{c}_0 + \mathbf{c}_1 = -\mathbf{c}_0 + \mathbf{c}_1$$
* At the end of an utterance ($t = T-1$), the succeeding frame is replicated: $\mathbf{c}_T = \mathbf{c}_{T-1}$.
  $$\Delta \mathbf{c}_{T-1} = -0.5 \mathbf{c}_{T-2} + 0.5 \mathbf{c}_{T-1}$$
  $$\Delta\Delta \mathbf{c}_{T-1} = \mathbf{c}_{T-2} - 2\mathbf{c}_{T-1} + \mathbf{c}_{T-1} = \mathbf{c}_{T-2} - \mathbf{c}_{T-1}$$
Computing dynamic features utterance-wide preserves natural inter-state transitions, enabling the HMM to learn the velocity of transition into and out of each state.

---

## 4. V2 Hidden Markov Model Architecture

### What Remains Unchanged from V1:
* **Dataset:** LJ Speech v1.1 (11,790 train, 655 val, 655 test).
* **Text Normalization:** `preprocessing/text_processing.py`.
* **G2P Engine:** 40 phonetic classes (39 ARPAbet + SIL).
* **Forced Alignment:** MFA supervised phone boundaries.
* **HMM Topology:** 3 left-to-right temporal states per phone (beginning, middle, end) with no skip transitions.
* **Duration Model:** Integer frame counts determined by empirical state duration means.
* **Vocoder:** Pseudo-inverse Mel filterbank + 60-iteration Fast Griffin-Lim.

### What Changes in V2:
* **Observation Dimensionality:** Expanded from 80-D to **240-D** (80 static + 80 delta + 80 delta-delta).
* **Learned Acoustic Parameters:**
  * Mean array $\boldsymbol{\mu}$: Shape `[40, 3, 240]` (28,800 parameters).
  * Variance array $\boldsymbol{\Sigma}$: Shape `[40, 3, 240]` diagonal elements floored at $10^{-6}$.
* **Synthesis Trajectory Generation:** Replaced direct mean tiling with the **MLPG linear solver**.

---

## 5. V2 Model Training

Implemented in `models/train_hmm_v2.py`.

```
For each of the 11,790 training utterances:
  1. Load 22,050 Hz audio WAV.
  2. Compute 80-D log-Mel spectrogram: C [T, 80].
  3. Compute 240-D dynamic features UTTERANCE-WIDE:
     O_t = [c_t, Delta c_t, DeltaDelta c_t]  shape: [T, 240]
  4. Load supervised MFA frame boundaries.
  5. For each segment (phone p, frames s to e):
     - Assign frame i to sub-state st = (3 * i) // n
     - Accumulate 240-D vectors into sum_x[p, st] and sum_x2[p, st]
     - Increment observation count obs_counts[p, st]
     - Record state frame durations
```

### Closed-Form MLE Parameter Estimation:
$$\boldsymbol{\mu}_{p, s} = \frac{\sum_{t \in (p, s)} \mathbf{o}_t}{N_{p, s}} \quad \in \mathbb{R}^{240}$$
$$\boldsymbol{\sigma}_{p, s}^2 = \max\left( \frac{\sum_{t \in (p, s)} \mathbf{o}_t^2}{N_{p, s}} - \boldsymbol{\mu}_{p, s}^2, \, 10^{-6} \right) \quad \in \mathbb{R}^{240}$$

---

## 6. Maximum Likelihood Parameter Generation (MLPG)

### 1. Intuition First: Why Mean Emission Fails
In a standard Gaussian distribution, the single most probable vector is its mean $\boldsymbol{\mu}$. If we independently pick the most probable vector at every frame, we obtain the mean at every frame, which produces flat lines.

However, in speech, a frame's static value $\mathbf{c}_t$, velocity $\Delta \mathbf{c}_t$, and acceleration $\Delta\Delta \mathbf{c}_t$ are mathematically linked:
$$\Delta \mathbf{c}_t = \frac{\mathbf{c}_{t+1} - \mathbf{c}_{t-1}}{2}$$
If $\mathbf{c}_t$ is flat, then its velocity $\Delta \mathbf{c}_t$ must be **zero**. But the HMM's learned dynamic distribution says that velocity is **not zero** during phonetic transitions! 

**MLPG resolves this contradiction:** It does not maximize the probability of each frame in isolation. Instead, it finds the single continuous static trajectory $\mathbf{C} = [\mathbf{c}_1, \dots, \mathbf{c}_T]^T$ that maximizes the joint probability across all frames while strictly satisfying the mathematical relationship between static positions, velocities, and accelerations.

---

### 2. Mathematical Formulation

Let $\mathbf{C} = [c_1, c_2, \dots, c_T]^T$ be the static sequence for one frequency channel over $T$ frames (length $T$).

Let $\mathbf{O} = [\mathbf{C}^T, \, \Delta\mathbf{C}^T, \, \Delta\Delta\mathbf{C}^T]^T$ be the complete observation vector (length $3T$).

The transformation from static trajectory $\mathbf{C}$ to full observation vector $\mathbf{O}$ is linear and represented by the **Window Matrix $W$**:
$$\mathbf{O} = W \mathbf{C} \quad \text{where } W \in \mathbb{R}^{3T \times T}$$

The probability density of the observation sequence given the state sequence $Q$ is:
$$P(\mathbf{O} \mid Q) = \mathcal{N}(\mathbf{O} \mid \boldsymbol{\mu}, U) = \frac{1}{\sqrt{(2\pi)^{3T} |U|}} \exp\left( -\frac{1}{2} (\mathbf{O} - \boldsymbol{\mu})^T U^{-1} (\mathbf{O} - \boldsymbol{\mu}) \right)$$
Where:
* $\boldsymbol{\mu} \in \mathbb{R}^{3T}$: Concatenated target means $[\boldsymbol{\mu}_c^T, \, \boldsymbol{\mu}_{\Delta}^T, \, \boldsymbol{\mu}_{\Delta\Delta}^T]^T$.
* $U \in \mathbb{R}^{3T \times 3T}$: Diagonal covariance matrix of state variances.
* $U^{-1}$: Diagonal precision matrix with entries $u_{ii}^{-1} = 1 / \sigma_i^2$.

Substituting $\mathbf{O} = W \mathbf{C}$ into the log-likelihood:
$$\ln P(W \mathbf{C} \mid Q) = -\frac{1}{2} (W \mathbf{C} - \boldsymbol{\mu})^T U^{-1} (W \mathbf{C} - \boldsymbol{\mu}) - \text{constant}$$

To find the optimal trajectory $\hat{\mathbf{C}}$, we take the matrix derivative with respect to $\mathbf{C}$ and set it to zero:
$$\frac{\partial \ln P}{\partial \mathbf{C}} = -W^T U^{-1} (W \mathbf{C} - \boldsymbol{\mu}) = \mathbf{0}$$
$$(W^T U^{-1} W) \mathbf{C} = W^T U^{-1} \boldsymbol{\mu}$$

This is a symmetric system of linear equations:
$$R \, \mathbf{C} = \mathbf{r}$$
Where:
* $R = W^T U^{-1} W \in \mathbb{R}^{T \times T}$ is the precision-weighted system matrix.
* $\mathbf{r} = W^T U^{-1} \boldsymbol{\mu} \in \mathbb{R}^T$ is the precision-weighted target vector.

---

### 3. The Banded Matrix Structure & Linear Time Solution

A naive matrix inversion of $R$ would cost $\mathcal{O}(T^3)$, which is prohibitively slow for long utterances (e.g., $T = 1,000$ frames $\rightarrow 10^9$ operations).

However, because the delta windows span only $\pm 1$ frame, the transformation matrix $W$ is composed of three blocks:
$$W = \begin{bmatrix} W_0 \\ W_1 \\ W_2 \end{bmatrix} \in \mathbb{R}^{3T \times T}$$
* $W_0 = I_T$ (Identity matrix, bandwidth 0).
* $W_1$ (Tridiagonal, bandwidth 1).
* $W_2$ (Tridiagonal, bandwidth 1).

When we compute $R = W^T U^{-1} W = W_0^T U_0^{-1} W_0 + W_1^T U_1^{-1} W_1 + W_2^T U_2^{-1} W_2$, the product of two tridiagonal matrices produces a **pentadiagonal banded matrix**:
* Main diagonal: $R_{t, t}$
* First sub-diagonal: $R_{t, t-1}$
* Second sub-diagonal: $R_{t, t-2}$
* All entries with $|i - j| > 2$ are identically **zero**!

### Lower Bandwidth = 2
Because $R$ is symmetric positive-definite and pentadiagonal (lower bandwidth $k_l = 2$), we extract its lower triangular compact form $[3, T]$ and solve it using **Cholesky Banded Decomposition** (`scipy.linalg.solveh_banded`):
$$\text{Computational Complexity} = \mathcal{O}(k_l^2 \cdot T) = \mathcal{O}(4T) = \mathcal{O}(T)$$
Synthesis runs in **linear time**, solving an entire utterance in less than 2 milliseconds per channel!

### Independent Channel Solvability
Because the HMM covariance matrix is diagonal (assuming independence across the 80 Mel frequency channels), the joint optimization decomposes into **80 independent 1D linear systems**:
$$R^{(d)} \mathbf{C}^{(d)} = \mathbf{r}^{(d)} \quad \text{for } d \in \{0, 1, \dots, 79\}$$
Each channel is solved independently, enabling fast, vectorized synthesis.

---

## 7. MLPG Boundary Handling

To avoid numerical mismatch between training and inference, the window matrix $W$ strictly mirrors the edge-padding logic used during feature extraction:

* **Delta Matrix ($W_1$):**
  * Interior row $t$: $W_1[t, t-1] = -0.5$, $W_1[t, t+1] = 0.5$.
  * Boundary row $0$: $W_1[0, 0] = -0.5$, $W_1[0, 1] = 0.5$.
  * Boundary row $T-1$: $W_1[T-1, T-2] = -0.5$, $W_1[T-1, T-1] = 0.5$.
* **Delta-Delta Matrix ($W_2$):**
  * Interior row $t$: $W_2[t, t-1] = 1.0$, $W_2[t, t] = -2.0$, $W_2[t, t+1] = 1.0$.
  * Boundary row $0$: $W_2[0, 0] = -1.0$, $W_2[0, 1] = 1.0$.
  * Boundary row $T-1$: $W_2[T-1, T-2] = 1.0$, $W_2[T-1, T-1] = -1.0$.

---

## 8. V2 Inference Step-by-Step Walk-Through

```
Text Input: "Speech synthesis with dynamic constraints."
    ↓
Text Normalization & G2P -> Phoneme Sequence: [P_1, P_2, ..., P_K]
    ↓
Duration Lookup -> State Frame Counts d_{p,s}
    ↓
Assemble Utterance Statistics Matrices [T, 240]:
  - Tile 240-D mean vector mu[p, s] across d_{p,s} frames -> Means Matrix [T, 240]
  - Tile 240-D variance vector var[p, s] across d_{p,s} frames -> Vars Matrix [T, 240]
    ↓
MLPG Solver (Executed independently for each of the 80 Mel dimensions):
  For channel d in 0..79:
    1. Extract 3T mean vector mu_d and precision vector U_inv_d = 1 / var_d
    2. Construct pentadiagonal R = W^T U_inv W and r = W^T U_inv mu
    3. Extract 3-row banded matrix ab
    4. Solve R * c_d = r via scipy.linalg.solveh_banded
    5. Place c_d into column d of C_hat
    ↓
Output: Smooth Continuous Log-Mel Spectrogram Matrix C_hat [T, 80]
    ↓
Acoustic Inversion:
  Linear Mel = exp(C_hat)
  Linear Mag = INV_MEL_BASIS @ Linear Mel   [513, T]
    ↓
Phase Estimation:
  Fast Griffin-Lim (60 iterations, momentum = 0.99)
    ↓
Synthesized Speech WAV (22,050 Hz)
```

---

## 9. Comprehensive Comparison: V1 vs. V2 Architecture

| Component | Version 1 (V1 Baseline) | Version 2 (V2 MLPG) | Why the Change Was Made |
| :--- | :--- | :--- | :--- |
| **Acoustic Features** | 80-D static log-Mel | 240-D (80 Static + 80 Delta + 80 Delta-Delta) | Provide velocity and acceleration constraints for smooth trajectory generation. |
| **Observation Dimension** | 80 features per frame | 240 features per frame | Captures rate of spectral change across time. |
| **Learned Model Statistics** | Means `[40, 3, 80]`, Vars `[40, 3, 80]` | Means `[40, 3, 240]`, Vars `[40, 3, 240]` | Enables joint likelihood optimization across static and dynamic domains. |
| **Trajectory Generation** | Direct state-mean tiling | MLPG Banded Cholesky Solver | Eliminates piecewise-constant stair-steps by optimizing utterance-wide trajectory. |
| **Mathematical Solver** | None (greedy tiling) | Analytical linear system: $(W^T U^{-1} W) C = W^T U^{-1} \boldsymbol{\mu}$ | Finds the globally optimal smooth trajectory under dynamic constraints in $\mathcal{O}(T)$ time. |
| **Duration Model** | Empirical state duration lookup | Empirical state duration lookup | Preserved unchanged to isolate the effect of trajectory generation. |
| **Vocoder** | Pseudo-inverse Mel + Griffin-Lim | Pseudo-inverse Mel + Griffin-Lim | Preserved unchanged to guarantee an unconfounded comparison. |
| **Phonetic Inventory** | 40 symbols (39 ARPAbet + SIL) | 40 symbols (39 ARPAbet + SIL) | Preserved unchanged. |
| **Alignment Method** | Supervised MFA boundaries | Supervised MFA boundaries | Preserved unchanged. |

---

## 10. Mathematical Sanity Checks & Unit Verification

Before conducting large-scale evaluations, three rigorous mathematical unit tests were implemented (`synthesis/test_mlpg.py`):

### 1. Gate V2-1: Dynamic Feature Extraction Correctness
* Generated a known synthetic static trajectory:
  $$\mathbf{C} = [1, 2, 4, 7, 11]^T$$
* Hand-calculated exact analytical deltas with edge padding ($[-0.5, 0, 0.5]$):
  $$\Delta \mathbf{C}_{\text{expected}} = [0.5, \, 1.5, \, 2.5, \, 3.5, \, 2.0]^T$$
* Verified that `compute_dynamic_features` reproduces hand-calculated values to machine precision.

### 2. Gate V2-3: MLPG Trajectory Inversion Sanity Test
* Constructed a random synthetic trajectory $\mathbf{C} \in \mathbb{R}^{10 \times 2}$.
* Generated observations $\mathbf{O} = W \mathbf{C}$.
* Set means $\boldsymbol{\mu} = \mathbf{O}$ and uniform variances $\boldsymbol{\Sigma} = 1.0$.
* Fed statistics into the MLPG solver.
* Verified that recovered trajectory $\hat{\mathbf{C}}$ matches original trajectory $\mathbf{C}$ within numerical tolerance:
  $$\max |\mathbf{C} - \hat{\mathbf{C}}| < 10^{-5}$$
* Verified that the maximum residual norm $\|R \hat{\mathbf{C}} - \mathbf{r}\| < 10^{-14}$.

### 3. Numerical Stability Checks across All Utterances
* Monitored precision matrix condition: $U^{-1} = 1 / \boldsymbol{\sigma}^2$ bounded by variance floor $10^{-6}$ (maximum precision $10^6$).
* Checked solver success rate across all 655 test utterances: **100.0%** (zero singular matrices, zero NaNs, zero infinities).

---

## 11. Validation Methodology & Test Set Quarantine

To guarantee scientific validity:
1. **Frozen V1 Baseline:** The V1 model was completely frozen. Its parameters and code were never modified during V2 development.
2. **Validation on Full 655 Utterances:** All development choices (window kernels, edge-padding logic, solver implementations) were validated exclusively on the 655 validation utterances (`data_splits/val.txt`).
3. **Strict Test Set Quarantine:** The 655 test utterances (`data_splits/test.txt`) remained completely untouched and uninspected until the final master evaluation script was executed.

---

## 12. Final Test Results (655 Unseen Utterances)

Both frozen V1 and frozen V2 were evaluated across all **655 held-out test utterances**:

| Metric | V1 Monophone HMM | V2 Monophone HMM (MLPG) | Absolute Improvement | Bootstrap 95% Confidence Interval |
| :--- | :--- | :--- | :--- | :--- |
| **Whisper-tiny WER (%)** | 78.78% | **47.10%** | **-31.68 percentage points** | **[-37.10, -27.54] percentage points** |
| **DCT-Derived MCD (dB)** | 40.95 dB | **39.31 dB** | **-1.64 dB** | **[-1.70, -1.59] dB** |
| **Boundary Discontinuity** | 0.6136 | **0.1974** | **-67.8% reduction** | — |
| **First-Order Change ($\bar{d}_1$)** | 0.2359 | **0.1814** | **-23.1% reduction** | — |
| **Second-Order Change ($\bar{d}_2$)** | 0.4661 | **0.0963** | **-79.3% reduction** | — |
| **Global Variance (GV)** | 1.4949 | **1.1792** | -21.1% (over-smoothing diagnostic) | — |
| **Duration Error (%)** | 8.5% | **8.5%** | 0.0% (identical duration model) | — |
| **Solver Success Rate** | N/A | **100.0%** | Numerically stable on all utterances | — |

---

## 13. Detailed Interpretation of Results

### 1. Word Error Rate (WER) (-31.68 Percentage Points)
* **What it shows:** An enormous gain in intelligibility. Whisper-tiny recognized significantly more words in V2 than V1.
* **Bootstrap 95% Confidence Interval:** `[-37.10, -27.54]` percentage points does not cross zero, confirming that the improvement is statistically robust and not an artifact of random test sample variation.
* **Why WER Dropped:** In V1, square-wave boundary transitions generated metallic phase buzz in Griffin-Lim, corrupting consonant formants and causing Whisper to hallucinate or drop words. In V2, smooth transitions allowed the vocoder to reconstruct cleaner acoustic cues.

### 2. Mel-Cepstral Distortion (MCD) (-1.64 dB)
* **What it shows:** A 1.64 dB reduction in cepstral distortion across 655 test utterances.
* **Bootstrap 95% Confidence Interval:** `[-1.70, -1.59]` dB.
* **Context:** In speech synthesis literature, an MCD improvement of $1.0\text{ dB}$ is considered perceptually noticeable; a $1.64\text{ dB}$ reduction confirms that the generated spectral envelope aligns substantially closer to natural human recordings.

### 3. Trajectory Diagnostics ($\bar{d}_1$, $\bar{d}_2$, and Boundary Discontinuity)
* **Boundary Discontinuity dropped from 0.6136 to 0.1974 (-67.8%):** Proves that MLPG substantially reduced the stair-step jumps at state boundaries.
* **Second-order change ($\bar{d}_2$) collapsed from 0.4661 to 0.0963 (-79.3%):** Second derivative measures acceleration and spikiness. A 79.3% drop proves that sharp corners were successfully smoothed into continuous curves.

### 4. Global Variance (GV) (1.4949 $\rightarrow$ 1.1792)
* **Diagnostic Significance:** The drop in Global Variance illustrates the classic, well-documented trade-off of standard MLPG: **over-smoothing**. Because MLPG finds the maximum likelihood trajectory under Gaussian distributions, extreme spectral peaks (formant resonances) are pulled slightly toward the mean. This explains why speech sounds much smoother but slightly muffled.

> [!CAUTION]
> **Strict Scientific Caution:**
> 1. Do NOT claim that V2 "completely eliminated" discontinuities. The boundary discontinuity is **0.1974**, which is non-zero. State the technically conservative truth: *"V2 substantially reduced state-boundary discontinuities."*
> 2. Do NOT claim that WER measures human listening pleasure. Whisper-tiny is an automated ASR proxy for intelligibility, not a human subjective listening score.

---

## 14. Why V2 Improved: Causal / Mechanistic Explanation

```
[V1 PIPELINE CAUSAL CHAIN]
Direct State Mean Tiling
    ↓
Piecewise-Constant Step Trajectory (Zero Articulatory Motion)
    ↓
Instantaneous Boundary Jumps (Discontinuity = 0.6136, d2 = 0.4661)
    ↓
Griffin-Lim Phase Estimation Failure (Harsh High-Frequency Phase Buzz)
    ↓
Degraded Acoustic Spectrum (MCD = 40.95 dB) & Unintelligible ASR (WER = 78.78%)

                          VERSUS

[V2 PIPELINE CAUSAL CHAIN]
Utterance-Wide Dynamic Features (Velocity & Acceleration Distributions)
    ↓
Joint Trajectory Optimization via MLPG: (W^T U^-1 W) C = W^T U^-1 mu
    ↓
Continuous, Differentiable Trajectory (Discontinuity = 0.1974, d2 = 0.0963)
    ↓
Improved Downstream Phase Inversion in Griffin-Lim
    ↓
Cleaner Acoustic Realization (MCD = 39.31 dB) & Substantially Lower WER (47.10%)
```

---

## 15. What V2 Still Does NOT Solve (Remaining Limitations)

Honesty about system limitations is the hallmark of an exceptional engineering defense. V2 is an important pedagogical milestone, but it is not a commercial TTS system:

1. **WER is Still 47.10%:** While 47.10% is dramatically better than 78.78%, roughly half of the words are still misrecognized by Whisper. Monophone models lack the context to distinguish co-articulated sounds.
2. **Griffin-Lim Vocoder Limitations:** Griffin-Lim is an algorithmic phase reconstructor, not a neural vocoder. It cannot synthesize natural breathiness or glottal pulse shaping, leaving a faint mechanical background timbre.
3. **Context-Insensitivity (Monophone Assumption):** The phoneme /k/ in `"cat"` (front vowel) has a completely different vocal tract shape than /k/ in `"cool"` (back rounded vowel). Our monophone model forces both to share the exact same Gaussian distribution.
4. **Unimodal Gaussian Distributions:** A single Gaussian per state cannot capture multi-modal acoustic variations.
5. **Over-Smoothing (GV Reduction):** Because standard MLPG averages out variance, synthesized speech can sound slightly flat or muffled compared to human speech.
6. **No Formal Human MOS Test:** The evaluation relies on objective metrics (WER, MCD, boundary discontinuity). A formal Mean Opinion Score (MOS) listening test with human raters was not conducted.

---

## 16. Why We Stopped at Version 2

1. **Syllabus and Pedagogical Coherence:** The primary educational objective was to master classical statistical machine learning (HMMs, sequential graphical models, parameter estimation, dynamic programming, and linear systems). Version 2 pushed this mathematical paradigm to its theoretical limit.
2. **Clean Experimental Attribution:** Because V2 changed *only* the trajectory generation mechanism while keeping dataset, G2P, duration, and vocoder strictly frozen, we can attribute 100% of the 31.68 percentage-point improvement directly to the dynamic feature constraints and MLPG solver.
3. **Reproducibility & Lightweight Execution:** The entire V2 training and inference pipeline runs in minutes on standard CPU hardware without multi-gigabyte GPU checkpoints.

---

## 17. Comprehensive V2 Viva Questions & Answers

### Q1: Is MLPG part of the standard course syllabus?
**Answer:** The core syllabus covers Hidden Markov Models and sequential probabilistic models. MLPG is a domain-specific mathematical extension developed in statistical speech synthesis (Tokuda et al.) to address the piecewise-constant failure mode of standard HMM emissions. We implemented MLPG to demonstrate how linear algebra and trajectory optimization solve the physical continuity problem inherent to speech.

### Q2: What exact mathematical objective does MLPG optimize?
**Answer:** MLPG maximizes the joint conditional log-likelihood of the static, velocity, and acceleration observation sequence:
$$\hat{\mathbf{C}} = \arg\max_{\mathbf{C}} P(W \mathbf{C} \mid Q, \lambda)$$
where $W$ is the deterministic finite-difference window matrix mapping the static trajectory $\mathbf{C}$ into concatenated static, delta, and delta-delta features.

### Q3: Why is the matrix $R = W^T U^{-1} W$ pentadiagonal?
**Answer:** Because the finite difference kernels for delta and delta-delta span $\pm 1$ frame. $W$ is composed of matrices with bandwidth 1. Multiplying $W^T W$ convolves these bandwidths, producing a non-zero bandwidth of $1 + 1 = 2$. Thus, only the main diagonal and two adjacent diagonals on either side are non-zero (a pentadiagonal matrix).

### Q4: Why can you solve MLPG independently for each Mel dimension?
**Answer:** Because our HMM uses a diagonal covariance matrix $\boldsymbol{\Sigma}$. Diagonal covariance assumes that the 80 frequency channels are conditionally independent given the state sequence. As a result, the joint likelihood decomposes into a sum of 80 independent likelihoods, allowing each channel to be solved as an independent 1D linear system.

### Q5: Why did Global Variance (GV) decrease from 1.4949 to 1.1792?
**Answer:** MLPG optimizes a quadratic likelihood objective under Gaussian distributions. Finding the trajectory that best satisfies the mean velocity and mean static targets inevitably smooths out extreme peaks and troughs. This phenomenon is known as "over-smoothing" in parametric synthesis.

### Q6: How did you ensure zero test data leakage during V2 development?
**Answer:** Utterances were partitioned deterministically at the utterance level before any training. The V1 model was frozen. During V2 development, hyperparameter selection, boundary padding, and solver validation were conducted exclusively on the 655 validation utterances (`data_splits/val.txt`). The 655 test utterances (`data_splits/test.txt`) were executed only once during the final benchmark run.

### Q7: If you had another month, what would be the single most impactful next step?
**Answer:** Replacing the monophone HMM with **context-dependent triphone HMMs clustered via decision trees**. Monophone models cannot capture co-articulation (the acoustic effect of neighboring sounds). Decision-tree clustered triphones would dramatically lower both MCD and WER by modeling phonetic context.
