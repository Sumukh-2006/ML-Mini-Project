"""Implementation of the approved Simple Phoneme Average Baseline.

Baseline Specification:
- Maps each phoneme to its single global average 80-D log-Mel vector across all states and occurrences in the training set.
- Duration for each phoneme is its empirical mean total duration (sum of the 3 state durations).
- Trajectory generation: emits the static 80-D phoneme-average vector repeated for the total phone duration.
- Identical vocoder: uses the exact same pseudo-inverse Mel filterbank and Griffin-Lim (60 iters, momentum=0.99) reconstruction.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.g2p import Pronouncer, PHONE_TO_ID
from preprocessing.config import N_STATES, SAMPLE_RATE
from models.hmm import MonophoneHMM
from synthesis.reconstruction import log_mel_to_wav


class PhonemeAverageBaseline:
    def __init__(self, model_path: str):
        # We load the trained HMM to extract empirical training statistics cleanly
        self.hmm = MonophoneHMM.load(model_path)
        self.pronouncer = Pronouncer()

        # Compute global average 80-D vector per phone across all states weighted by state counts
        num_phones = self.hmm.num_phones
        n_mels = self.hmm.n_mels
        self.phone_means = np.zeros((num_phones, n_mels), dtype=np.float32)

        for p_idx in range(num_phones):
            total_counts = np.sum(self.hmm.state_counts[p_idx])
            if total_counts > 0:
                weighted_sum = np.zeros(n_mels, dtype=np.float32)
                for st in range(N_STATES):
                    weighted_sum += self.hmm.means[p_idx, st] * self.hmm.state_counts[p_idx, st]
                self.phone_means[p_idx] = weighted_sum / total_counts
            else:
                self.phone_means[p_idx] = np.mean(self.hmm.means[p_idx], axis=0)

    def synthesize(self, text: str, n_iter: int = 60, momentum: float = 0.99):
        """Full baseline synthesis: text -> phonemes -> static phone average vectors -> Griffin-Lim."""
        g2p_res = self.pronouncer.text_to_phonemes(text)
        phones = g2p_res["phones"]

        frames_list = []
        for p_name in phones:
            p_idx = PHONE_TO_ID[p_name]
            # Total phone duration in frames
            total_dur = max(1, int(np.round(self.hmm.phone_total_duration_mean[p_idx])))
            # Static 80-D vector for this entire phoneme
            p_vec = self.phone_means[p_idx]
            frames_list.append(np.tile(p_vec, (total_dur, 1)))

        log_mel = np.vstack(frames_list)  # [T, 80]

        # Reconstruct audio using identical vocoder configuration
        wav = log_mel_to_wav(log_mel, n_iter=n_iter, momentum=momentum)
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav / max(1.0, peak)

        return {
            "wav": wav,
            "log_mel": log_mel,
            "phones": phones,
            "n_frames": log_mel.shape[0],
            "duration_s": len(wav) / SAMPLE_RATE,
        }
