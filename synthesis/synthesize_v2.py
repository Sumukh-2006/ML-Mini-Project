"""Acoustic parameter generation and end-to-end synthesis module for V2 HMM.

V2 Generation Strategy:
- Given a phoneme sequence, query duration model for integer frame counts.
- Construct the full utterance 240-D mean and variance sequences.
- Pass to MLPG solver to obtain the optimal 80-D log-Mel trajectory.
- Reconstruct the continuous log-Mel trajectory via Griffin-Lim.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.g2p import Pronouncer, PHONE_TO_ID, SIL_PHONE
from preprocessing.config import N_STATES, HOP_LENGTH, SAMPLE_RATE
from models.hmm_v2 import MonophoneHMMV2
from synthesis.reconstruction import log_mel_to_wav
from synthesis.mlpg import mlpg


class HMMV2Synthesizer:
    def __init__(self, model_path: str):
        self.hmm = MonophoneHMMV2.load(model_path)
        self.pronouncer = Pronouncer()

    def text_to_phonemes(self, text: str):
        """Convert input text to canonical phonemes using V1 deterministic G2P."""
        return self.pronouncer.text_to_phonemes(text)

    def generate_log_mel(self, phone_list: list):
        """Generate [T, 80] log-Mel trajectory from phoneme sequence using MLPG.
        
        Returns:
            log_mel: [T, 80] numpy array (optimized static features)
            state_sequence: list of (phone_name, state_idx, frame_count)
            metrics: MLPG solver diagnostics
        """
        means_list = []
        vars_list = []
        state_sequence = []

        for p_name in phone_list:
            p_idx = PHONE_TO_ID[p_name]
            for st in range(N_STATES):
                dur_float = self.hmm.duration_mean[p_idx, st]
                dur_int = max(1, int(np.round(dur_float)))

                mean_vec = self.hmm.means[p_idx, st]  # [240]
                var_vec = self.hmm.variances[p_idx, st] # [240]

                means_list.append(np.tile(mean_vec, (dur_int, 1)))
                vars_list.append(np.tile(var_vec, (dur_int, 1)))
                state_sequence.append((p_name, st, dur_int))

        utterance_means = np.vstack(means_list)  # [T, 240]
        utterance_vars = np.vstack(vars_list)    # [T, 240]

        # MLPG optimization
        log_mel, metrics = mlpg(utterance_means, utterance_vars)
        return log_mel, state_sequence, metrics

    def synthesize(self, text: str, n_iter: int = 60, momentum: float = 0.99):
        """Full text -> audio pipeline."""
        g2p_res = self.text_to_phonemes(text)
        phones = g2p_res["phones"]
        log_mel, state_seq, metrics = self.generate_log_mel(phones)

        # Invert to WAV via Griffin-Lim
        wav = log_mel_to_wav(log_mel, n_iter=n_iter, momentum=momentum)
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav / max(1.0, peak)

        return {
            "wav": wav,
            "log_mel": log_mel,
            "phones": phones,
            "tokens": g2p_res["tokens"],
            "state_sequence": state_seq,
            "mlpg_metrics": metrics,
            "n_frames": log_mel.shape[0],
            "duration_s": len(wav) / SAMPLE_RATE,
        }
