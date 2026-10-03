"""Acoustic and ASR Evaluation Metrics for V1 TTS Evaluation.

Formulations:
1. Global Spectral MSE:
   Mean Squared Error between average log-Mel spectral vectors of ground truth and generated audio.
   MSE = mean((mean(L_orig, axis=0) - mean(L_gen, axis=0))^2)

2. Mel-Cepstral Distortion (MCD):
   Standard MCD is defined on Mel-Cepstral Coefficients (MCCs) rather than raw log-Mel bins.
   To obtain mathematically correct MCCs from 80-D log-Mel spectrogram:
     c_t(k) = Type-II DCT of log-Mel vector at frame t (ortho-normalized)
   We retain the first K=16 cepstral coefficients (c_1 to c_15, dropping c_0 energy term as standard in speech synthesis literature):
     MCD_t = (10 * sqrt(2) / ln(10)) * sqrt( sum_{k=1}^{K-1} (c_{orig, t}(k) - c_{gen, t}(k))^2 )  [dB]
   For sequences with differing frame lengths, Dynamic Time Warping (DTW) aligns c_orig and c_gen along the minimum Euclidean distance path prior to calculating average MCD:
     MCD = (10 * sqrt(2) / ln(10)) * mean( Euclidean_dist(c_orig[path_i], c_gen[path_j]) )

3. Word Error Rate (WER):
   Standard Levenshtein distance on normalized text tokens:
     WER = (Insertions + Deletions + Substitutions) / Total_Reference_Words
"""
import numpy as np
from scipy.fftpack import dct
from scipy.spatial.distance import cdist


def log_mel_to_mcep(log_mel: np.ndarray, num_cep: int = 16) -> np.ndarray:
    """Convert [T, 80] log-Mel spectrogram to [T, num_cep] Mel-Cepstral Coefficients via DCT-II.
    Drops c_0 (overall energy) to measure pure spectral shape distortion as standard in speech synthesis.
    """
    # Type-II DCT across the 80 mel frequency bins
    # shape: [T, 80]
    cepstra = dct(log_mel, type=2, axis=-1, norm="ortho")
    # Exclude c_0, retain c_1 through c_{num_cep-1} -> [T, num_cep - 1]
    return cepstra[:, 1:num_cep]


def compute_dtw_mcd(log_mel_ref: np.ndarray, log_mel_gen: np.ndarray, num_cep: int = 16) -> float:
    """Compute DTW-aligned Mel-Cepstral Distortion (dB) between reference and generated log-Mel.
    MCD factor = (10 * sqrt(2)) / ln(10) ≈ 6.14185 dB
    """
    c_ref = log_mel_to_mcep(log_mel_ref, num_cep=num_cep)
    c_gen = log_mel_to_mcep(log_mel_gen, num_cep=num_cep)

    # Cost matrix: Euclidean distance between cepstral frames
    cost_matrix = cdist(c_ref, c_gen, metric="euclidean")

    # Simple dynamic time warping to find optimal alignment path
    T_ref, T_gen = cost_matrix.shape
    dtw = np.full((T_ref + 1, T_gen + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, T_ref + 1):
        for j in range(1, T_gen + 1):
            cost = cost_matrix[i - 1, j - 1]
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # Traceback minimum-cost path
    i, j = T_ref, T_gen
    path_costs = []
    while i > 0 and j > 0:
        path_costs.append(cost_matrix[i - 1, j - 1])
        steps = [dtw[i - 1, j - 1], dtw[i - 1, j], dtw[i, j - 1]]
        min_idx = np.argmin(steps)
        if min_idx == 0:
            i -= 1
            j -= 1
        elif min_idx == 1:
            i -= 1
        else:
            j -= 1

    # Standard conversion factor to decibels (dB)
    scaling_factor = (10.0 * np.sqrt(2.0)) / np.log(10.0)
    mcd_db = scaling_factor * float(np.mean(path_costs))
    return mcd_db


def compute_spectral_mse(log_mel_ref: np.ndarray, log_mel_gen: np.ndarray) -> float:
    """Compute global MSE between average spectral profiles [80-D]."""
    mean_ref = np.mean(log_mel_ref, axis=0)
    mean_gen = np.mean(log_mel_gen, axis=0)
    return float(np.mean((mean_ref - mean_gen) ** 2))


def compute_wer(reference_text: str, hypothesis_text: str) -> float:
    """Compute standard Word Error Rate (WER) using Levenshtein distance on words."""
    import re
    # Normalize: lowercase, strip punctuation
    def clean(text):
        return re.findall(r"\b\w+\b", text.lower())

    ref_words = clean(reference_text)
    hyp_words = clean(hypothesis_text)

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # Dynamic programming for Levenshtein distance
    r_len = len(ref_words)
    h_len = len(hyp_words)
    d = np.zeros((r_len + 1, h_len + 1), dtype=np.int32)

    for i in range(r_len + 1):
        d[i, 0] = i
    for j in range(h_len + 1):
        d[0, j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    return float(d[r_len, h_len]) / float(r_len)
