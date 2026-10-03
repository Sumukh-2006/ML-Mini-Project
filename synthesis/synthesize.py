"""Acoustic parameter generation and end-to-end synthesis module for V1 HMM.

V1 Generation Strategy:
- Given a phoneme sequence, query the duration model to determine integer frame counts per state.
  Each state frame count: round(mean_duration), clamped to >= 1 frame.
- For each state, emit its learned 80-D Gaussian mean vector repeated for its allocated frames.
- Reconstruct the resulting [T, 80] piecewise-constant log-Mel trajectory via Griffin-Lim.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.g2p import Pronouncer, PHONE_TO_ID, SIL_PHONE
from preprocessing.config import N_STATES, HOP_LENGTH, SAMPLE_RATE
from models.hmm import MonophoneHMM
from synthesis.reconstruction import log_mel_to_wav


class HMMV1Synthesizer:
    def __init__(self, model_path: str):
        self.hmm = MonophoneHMM.load(model_path)
        self.pronouncer = Pronouncer()

    def text_to_phonemes(self, text: str):
        """Convert input text to canonical phonemes using V1 deterministic G2P."""
        return self.pronouncer.text_to_phonemes(text)

    def generate_log_mel(self, phone_list: list):
        """Generate [T, 80] log-Mel trajectory from phoneme sequence using Gaussian means.
        
        Returns:
            log_mel: [T, 80] numpy array
            state_sequence: list of (phone_name, state_idx, frame_count)
        """
        frames_list = []
        state_sequence = []

        for p_name in phone_list:
            p_idx = PHONE_TO_ID[p_name]
            for st in range(N_STATES):
                # Integer frame duration: round to nearest integer, enforcing at least 1 frame
                dur_float = self.hmm.duration_mean[p_idx, st]
                dur_int = max(1, int(np.round(dur_float)))

                # 80-D mean vector for this state
                mean_vec = self.hmm.means[p_idx, st]  # [80]

                # Tile mean vector for dur_int frames
                state_frames = np.tile(mean_vec, (dur_int, 1))  # [dur_int, 80]
                frames_list.append(state_frames)
                state_sequence.append((p_name, st, dur_int))

        log_mel = np.vstack(frames_list)  # [T, 80]
        return log_mel, state_sequence

    def synthesize(self, text: str, n_iter: int = 60, momentum: float = 0.99):
        """Full text -> audio pipeline."""
        g2p_res = self.text_to_phonemes(text)
        phones = g2p_res["phones"]
        log_mel, state_seq = self.generate_log_mel(phones)

        # Invert to WAV via Griffin-Lim
        wav = log_mel_to_wav(log_mel, n_iter=n_iter, momentum=momentum)
        # Normalize audio peak
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav / max(1.0, peak)

        return {
            "wav": wav,
            "log_mel": log_mel,
            "phones": phones,
            "tokens": g2p_res["tokens"],
            "state_sequence": state_seq,
            "n_frames": log_mel.shape[0],
            "duration_s": len(wav) / SAMPLE_RATE,
        }
