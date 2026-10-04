"""Module for V2 HMM definitions and parameter estimation.

V2 Model Specification:
- 240-D features (80 Static + 80 Delta + 80 Delta-Delta)
- 40 phonetic symbols
- 3 left-to-right states per phone
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.g2p import PHONE_LIST, PHONE_TO_ID
from preprocessing.config import N_MELS, N_STATES

NUM_PHONES = len(PHONE_LIST)
N_MELS_V2 = N_MELS * 3  # 240

class MonophoneHMMV2:
    def __init__(self, num_phones=NUM_PHONES, num_states=N_STATES, n_mels=N_MELS_V2, var_floor=1e-6):
        self.num_phones = num_phones
        self.num_states = num_states
        self.n_mels = n_mels
        self.var_floor = var_floor

        # Model parameters
        self.means = np.zeros((num_phones, num_states, n_mels), dtype=np.float32)
        self.variances = np.full((num_phones, num_states, n_mels), var_floor, dtype=np.float32)
        self.state_counts = np.zeros((num_phones, num_states), dtype=np.int64)

        # Duration statistics
        self.duration_mean = np.zeros((num_phones, num_states), dtype=np.float32)
        self.duration_std = np.zeros((num_phones, num_states), dtype=np.float32)
        self.duration_min = np.zeros((num_phones, num_states), dtype=np.float32)
        self.duration_max = np.zeros((num_phones, num_states), dtype=np.float32)
        self.phone_total_duration_mean = np.zeros(num_phones, dtype=np.float32)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "num_phones": self.num_phones,
            "num_states": self.num_states,
            "n_mels": self.n_mels,
            "var_floor": self.var_floor,
            "phone_list": PHONE_LIST,
            "means": self.means.tolist(),
            "variances": self.variances.tolist(),
            "state_counts": self.state_counts.tolist(),
            "duration_mean": self.duration_mean.tolist(),
            "duration_std": self.duration_std.tolist(),
            "duration_min": self.duration_min.tolist(),
            "duration_max": self.duration_max.tolist(),
            "phone_total_duration_mean": self.phone_total_duration_mean.tolist(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"HMM V2 model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls(
            num_phones=data["num_phones"],
            num_states=data["num_states"],
            n_mels=data["n_mels"],
            var_floor=data["var_floor"],
        )
        model.means = np.array(data["means"], dtype=np.float32)
        model.variances = np.array(data["variances"], dtype=np.float32)
        model.state_counts = np.array(data["state_counts"], dtype=np.int64)
        model.duration_mean = np.array(data["duration_mean"], dtype=np.float32)
        model.duration_std = np.array(data["duration_std"], dtype=np.float32)
        model.duration_min = np.array(data["duration_min"], dtype=np.float32)
        model.duration_max = np.array(data["duration_max"], dtype=np.float32)
        model.phone_total_duration_mean = np.array(data["phone_total_duration_mean"], dtype=np.float32)
        return model
