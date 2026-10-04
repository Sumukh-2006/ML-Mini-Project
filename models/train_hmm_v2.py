"""Gate V2-2: Train V2 Monophone HMM parameters with dynamic features.

Guarantees:
1. Strict train split isolation: Only utterances from `data_splits/train.txt` are consumed.
2. Complete frame-to-state validation.
3. Feature extraction: 240-D dynamic features (Static + Delta + Delta-Delta)
   computed UTTERANCE-WIDE with strict edge-padding before state slicing.
"""
import os
import sys
import json
import numpy as np
import soundfile as sf
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.g2p import PHONE_LIST, PHONE_TO_ID
from preprocessing.config import N_MELS, N_STATES, SAMPLE_RATE, HOP_LENGTH
from synthesis.reconstruction import wav_to_log_mel
from synthesis.mlpg import compute_dynamic_features
from models.hmm_v2 import MonophoneHMMV2, N_MELS_V2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_SPLIT = os.path.join(ROOT, "data_splits", "train.txt")
VAL_SPLIT = os.path.join(ROOT, "data_splits", "val.txt")
TEST_SPLIT = os.path.join(ROOT, "data_splits", "test.txt")
ALIGN_FILE = os.path.join(ROOT, "outputs", "alignments", "full", "alignments.jsonl")
WAV_DIR = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "wavs")
OUT_MODEL_PATH = os.path.join(ROOT, "outputs", "models", "v2_hmm_model.json")


def train_v2_hmm():
    print("=== Phase: Training V2 Monophone HMM (240-D Dynamic Features) ===")
    
    with open(TRAIN_SPLIT, "r", encoding="utf-8") as f:
        train_ids = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(train_ids)} training utterance IDs.")

    with open(VAL_SPLIT, "r", encoding="utf-8") as f:
        val_ids = set(line.strip() for line in f if line.strip())
    with open(TEST_SPLIT, "r", encoding="utf-8") as f:
        test_ids = set(line.strip() for line in f if line.strip())

    assert len(train_ids & val_ids) == 0, "Data leakage between train and val!"
    assert len(train_ids & test_ids) == 0, "Data leakage between train and test!"

    print("Reading alignments for training set...")
    train_alignments = {}
    with open(ALIGN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            uid = rec["id"]
            if uid in train_ids:
                train_alignments[uid] = rec

    # Accumulators for 240-D features
    sum_x = np.zeros((len(PHONE_LIST), N_STATES, N_MELS_V2), dtype=np.float64)
    sum_x2 = np.zeros((len(PHONE_LIST), N_STATES, N_MELS_V2), dtype=np.float64)
    obs_counts = np.zeros((len(PHONE_LIST), N_STATES), dtype=np.int64)
    state_durations = defaultdict(lambda: [[] for _ in range(N_STATES)])

    total_frames = 0
    total_labeled_frames = 0
    overlapping_frames = 0
    unlabeled_frames = 0

    print("Beginning V2 feature extraction...")
    for idx, uid in enumerate(train_ids):
        rec = train_alignments[uid]
        wav_path = os.path.join(WAV_DIR, f"{uid}.wav")
        audio, sr = sf.read(wav_path)
        assert sr == SAMPLE_RATE

        # 80-D log-Mel features
        log_mel = wav_to_log_mel(audio)
        T, D = log_mel.shape
        assert D == N_MELS

        # V2: Compute 240-D dynamic features UTTERANCE-WIDE
        features_240 = compute_dynamic_features(log_mel)
        assert features_240.shape == (T, N_MELS_V2)

        frame_visited = np.zeros(T, dtype=np.int32)
        total_frames += T
        prev_end = 0

        for seg in rec["segments"]:
            phone = seg["phone"]
            phone_idx = PHONE_TO_ID[phone]
            s = seg["start"]
            e = seg["end"]
            n = seg["n_frames"]
            state_frames = seg["state_frames"]

            assert s == prev_end
            assert 0 <= s <= e <= T
            prev_end = e

            frame_visited[s:e] += 1

            for st in range(N_STATES):
                state_durations[phone_idx][st].append(state_frames[st])

            if n > 0:
                seg_frames = features_240[s:e]  # [n, 240]
                for i in range(n):
                    st = (N_STATES * i) // n
                    frame_vec = seg_frames[i]
                    sum_x[phone_idx, st] += frame_vec
                    sum_x2[phone_idx, st] += frame_vec ** 2
                    obs_counts[phone_idx, st] += 1

        assert prev_end == T

        total_labeled_frames += np.sum(frame_visited == 1)
        overlapping_frames += np.sum(frame_visited > 1)
        unlabeled_frames += np.sum(frame_visited == 0)

        if (idx + 1) % 2000 == 0 or (idx + 1) == len(train_ids):
            print(f"  Processed {idx + 1}/{len(train_ids)} training utterances...")

    assert total_frames == total_labeled_frames
    assert overlapping_frames == 0
    assert unlabeled_frames == 0

    hmm = MonophoneHMMV2(num_phones=len(PHONE_LIST), num_states=N_STATES, n_mels=N_MELS_V2, var_floor=1e-6)
    hmm.state_counts = obs_counts

    zero_obs_states = []
    print("\nEstimating means, variances, and duration statistics...")
    for p_idx, p_name in enumerate(PHONE_LIST):
        for st in range(N_STATES):
            c = obs_counts[p_idx, st]
            durs = np.array(state_durations[p_idx][st], dtype=np.float32)

            if len(durs) > 0:
                hmm.duration_mean[p_idx, st] = float(np.mean(durs))
                hmm.duration_std[p_idx, st] = float(np.std(durs))
                hmm.duration_min[p_idx, st] = float(np.min(durs))
                hmm.duration_max[p_idx, st] = float(np.max(durs))
            
            if c == 0:
                zero_obs_states.append((p_name, st))
                hmm.means[p_idx, st] = 0.0
                hmm.variances[p_idx, st] = 1.0
            else:
                mu = sum_x[p_idx, st] / c
                var = (sum_x2[p_idx, st] / c) - (mu ** 2)
                var = np.maximum(var, 1e-6)
                hmm.means[p_idx, st] = mu.astype(np.float32)
                hmm.variances[p_idx, st] = var.astype(np.float32)

        hmm.phone_total_duration_mean[p_idx] = float(np.sum(hmm.duration_mean[p_idx]))

    print("\n--- Running Parameter Sanity Checks ---")
    assert not np.isnan(hmm.means).any(), "NaN found in means!"
    assert not np.isinf(hmm.means).any(), "Inf found in means!"
    assert not np.isnan(hmm.variances).any(), "NaN found in variances!"
    assert not np.isinf(hmm.variances).any(), "Inf found in variances!"
    assert (hmm.variances >= 1e-6).all(), "Variance floor violation!"
    assert hmm.means.shape == (40, 3, 240)
    assert hmm.variances.shape == (40, 3, 240)
    
    hmm.save(OUT_MODEL_PATH)
    print("V2 Training Completed!")


if __name__ == "__main__":
    train_v2_hmm()
