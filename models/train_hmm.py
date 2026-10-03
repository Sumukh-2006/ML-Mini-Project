"""Train locked V1 Monophone HMM parameters via supervised alignment-to-frame mapping.

Guarantees:
1. Strict train split isolation: Only utterances from `data_splits/train.txt` are consumed.
2. Complete frame-to-state validation:
   - Exactly one (phone, state) label per acoustic frame.
   - 0 unlabeled frames, 0 overlapping labels.
   - All frame indices within [0, T).
3. Deterministic handling of short intervals (N < 3 frames):
   - state(i) = (3 * i) // N. Only states that actually receive frames accumulate statistics.
   - Duration statistics record the actual allocated frames per state.
4. Parameter estimation:
   - empirical mean of assigned 80-D log-Mel vectors
   - empirical variance of assigned 80-D log-Mel vectors
   - variance floored at 1e-6
5. Duration modeling:
   - Empirical mean, std, min, max frame counts per (phone, state).
"""
import os
import sys
import json
import numpy as np
import soundfile as sf
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.g2p import PHONE_LIST, PHONE_TO_ID, SIL_PHONE
from preprocessing.config import N_MELS, N_STATES, SAMPLE_RATE, HOP_LENGTH
from synthesis.reconstruction import wav_to_log_mel
from models.hmm import MonophoneHMM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_SPLIT = os.path.join(ROOT, "data_splits", "train.txt")
VAL_SPLIT = os.path.join(ROOT, "data_splits", "val.txt")
TEST_SPLIT = os.path.join(ROOT, "data_splits", "test.txt")
ALIGN_FILE = os.path.join(ROOT, "outputs", "alignments", "full", "alignments.jsonl")
WAV_DIR = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "wavs")
OUT_MODEL_PATH = os.path.join(ROOT, "outputs", "models", "v1_hmm_model.json")


def train_v1_hmm():
    print("=== Phase: Training Locked V1 Monophone HMM ===")
    
    # 1. Load train split IDs
    with open(TRAIN_SPLIT, "r", encoding="utf-8") as f:
        train_ids = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(train_ids)} training utterance IDs.")

    with open(VAL_SPLIT, "r", encoding="utf-8") as f:
        val_ids = set(line.strip() for line in f if line.strip())
    with open(TEST_SPLIT, "r", encoding="utf-8") as f:
        test_ids = set(line.strip() for line in f if line.strip())

    # Double check zero leakage
    assert len(train_ids & val_ids) == 0, "Data leakage between train and val!"
    assert len(train_ids & test_ids) == 0, "Data leakage between train and test!"

    # 2. Load alignments
    print("Reading alignments for training set...")
    train_alignments = {}
    with open(ALIGN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            uid = rec["id"]
            if uid in train_ids:
                train_alignments[uid] = rec
    print(f"Loaded alignments for {len(train_alignments)} training utterances.")

    # Accumulators for sufficient statistics: sum(x) and sum(x^2)
    # Using 64-bit float accumulators for numerical precision
    sum_x = np.zeros((len(PHONE_LIST), N_STATES, N_MELS), dtype=np.float64)
    sum_x2 = np.zeros((len(PHONE_LIST), N_STATES, N_MELS), dtype=np.float64)
    obs_counts = np.zeros((len(PHONE_LIST), N_STATES), dtype=np.int64)

    # Accumulator for duration statistics (list of frame counts per phone/state)
    state_durations = defaultdict(lambda: [[] for _ in range(N_STATES)])

    total_frames = 0
    total_labeled_frames = 0
    overlapping_frames = 0
    unlabeled_frames = 0

    print("Beginning feature extraction and frame-to-state validation...")
    for idx, uid in enumerate(train_ids):
        rec = train_alignments[uid]
        wav_path = os.path.join(WAV_DIR, f"{uid}.wav")
        audio, sr = sf.read(wav_path)
        assert sr == SAMPLE_RATE

        # 80-D log-Mel features
        log_mel = wav_to_log_mel(audio)
        T, D = log_mel.shape
        assert D == N_MELS

        # Verification arrays: check label assignment for every frame in [0, T)
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

            # Bounds & chronological order check
            assert s == prev_end, f"Chronological gap/overlap in {uid}: seg start {s} != prev end {prev_end}"
            assert 0 <= s <= e <= T, f"Segment [{s}, {e}) out of bounds [0, {T}) in {uid}"
            prev_end = e

            # Track frame visits
            frame_visited[s:e] += 1

            # Accumulate duration
            for st in range(N_STATES):
                state_durations[phone_idx][st].append(state_frames[st])

            # Accumulate acoustic frames
            if n > 0:
                seg_frames = log_mel[s:e]  # [n, 80]
                for i in range(n):
                    st = (N_STATES * i) // n
                    frame_vec = seg_frames[i]
                    sum_x[phone_idx, st] += frame_vec
                    sum_x2[phone_idx, st] += frame_vec ** 2
                    obs_counts[phone_idx, st] += 1

        assert prev_end == T, f"Final segment end {prev_end} != total frames {T} in {uid}"

        # Count validation stats
        labeled = np.sum(frame_visited == 1)
        overlaps = np.sum(frame_visited > 1)
        unlabeled = np.sum(frame_visited == 0)

        total_labeled_frames += labeled
        overlapping_frames += overlaps
        unlabeled_frames += unlabeled

        assert overlaps == 0, f"Overlapping frame assignments found in {uid}!"
        assert unlabeled == 0, f"Unlabeled frames found in {uid}!"

        if (idx + 1) % 2000 == 0 or (idx + 1) == len(train_ids):
            print(f"  Processed {idx + 1}/{len(train_ids)} training utterances...")

    print("\n--- Frame-to-State Mapping Validation Results ---")
    print(f"Total Acoustic Frames: {total_frames}")
    print(f"Total Labeled Frames:  {total_labeled_frames} (100.0%)")
    print(f"Overlapping Frames:    {overlapping_frames} (0.0%)")
    print(f"Unlabeled Frames:      {unlabeled_frames} (0.0%)")
    assert total_frames == total_labeled_frames
    assert overlapping_frames == 0
    assert unlabeled_frames == 0
    print("Frame-to-state validation: PASSED perfectly.")

    # 3. Model Parameter Estimation
    hmm = MonophoneHMM(num_phones=len(PHONE_LIST), num_states=N_STATES, n_mels=N_MELS, var_floor=1e-6)
    hmm.state_counts = obs_counts

    zero_obs_states = []
    print("\nEstimating means, variances, and duration statistics...")
    for p_idx, p_name in enumerate(PHONE_LIST):
        phone_tot_dur = []
        for st in range(N_STATES):
            c = obs_counts[p_idx, st]
            durs = np.array(state_durations[p_idx][st], dtype=np.float32)

            # Durations
            if len(durs) > 0:
                hmm.duration_mean[p_idx, st] = float(np.mean(durs))
                hmm.duration_std[p_idx, st] = float(np.std(durs))
                hmm.duration_min[p_idx, st] = float(np.min(durs))
                hmm.duration_max[p_idx, st] = float(np.max(durs))
            
            # Emissions
            if c == 0:
                zero_obs_states.append((p_name, st))
                hmm.means[p_idx, st] = 0.0
                hmm.variances[p_idx, st] = 1.0
            else:
                mu = sum_x[p_idx, st] / c
                var = (sum_x2[p_idx, st] / c) - (mu ** 2)
                # Apply variance flooring
                var = np.maximum(var, 1e-6)
                hmm.means[p_idx, st] = mu.astype(np.float32)
                hmm.variances[p_idx, st] = var.astype(np.float32)

        # Total phone duration mean
        hmm.phone_total_duration_mean[p_idx] = float(np.sum(hmm.duration_mean[p_idx]))

    # Report observation availability
    print("\n--- State Observation Availability ---")
    print(f"Zero observation states: {zero_obs_states if zero_obs_states else 'NONE (All 120 states have observations)'}")
    min_obs = np.min(obs_counts)
    max_obs = np.max(obs_counts)
    print(f"Observation counts per state: min={min_obs}, max={max_obs}")

    # 4. Parameter Sanity Checks
    print("\n--- Running Parameter Sanity Checks ---")
    assert not np.isnan(hmm.means).any(), "NaN found in means!"
    assert not np.isinf(hmm.means).any(), "Inf found in means!"
    assert not np.isnan(hmm.variances).any(), "NaN found in variances!"
    assert not np.isinf(hmm.variances).any(), "Inf found in variances!"
    assert (hmm.variances >= 1e-6).all(), "Variance floor violation!"
    assert hmm.means.shape == (40, 3, 80), f"Incorrect means shape {hmm.means.shape}"
    assert hmm.variances.shape == (40, 3, 80), f"Incorrect variances shape {hmm.variances.shape}"
    print("Sanity checks PASSED: All 40 phones x 3 states have valid, finite, positive parameters.")

    # 5. Save Model
    hmm.save(OUT_MODEL_PATH)

    # 6. Display Representative Inspection Samples
    print("\n--- Inspection of Learned Parameters (Selected Phonemes) ---")
    sample_phones = ["SIL", "AA", "S", "P", "IY"]
    for sp in sample_phones:
        p_idx = PHONE_TO_ID[sp]
        print(f"\nPhoneme: {sp} (Total observations across states: {int(np.sum(obs_counts[p_idx]))})")
        print(f"  Total Phone Mean Duration: {hmm.phone_total_duration_mean[p_idx]:.2f} frames ({hmm.phone_total_duration_mean[p_idx]*HOP_LENGTH/SAMPLE_RATE*1000:.1f} ms)")
        for st in range(N_STATES):
            st_name = ["Onset (State 0)", "Nucleus (State 1)", "Coda (State 2)"][st]
            m_slice = hmm.means[p_idx, st]
            v_slice = hmm.variances[p_idx, st]
            dur_m = hmm.duration_mean[p_idx, st]
            print(f"  [{st_name}] Obs: {obs_counts[p_idx, st]:<6d} | Duration Mean: {dur_m:.2f} frames")
            print(f"    Mean (first 5 dims):     {np.round(m_slice[:5], 3)}")
            print(f"    Variance (first 5 dims): {np.round(v_slice[:5], 4)}")

    print("\nHMM parameter estimation and duration modeling successfully completed!")


if __name__ == "__main__":
    train_v1_hmm()
