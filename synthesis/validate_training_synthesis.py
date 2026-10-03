"""Gate 6: Validation script for training utterance synthesis."""
import os
import sys
import json
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.config import SAMPLE_RATE, HOP_LENGTH
from synthesis.synthesize import HMMV1Synthesizer
from synthesis.reconstruction import wav_to_log_mel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "outputs", "models", "v1_hmm_model.json")
WAV_DIR = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "wavs")
META_PATH = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "metadata.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "synthesis", "training_validation")
os.makedirs(OUT_DIR, exist_ok=True)

# Select 4 representative training utterances of varying length and phonetic diversity
TARGET_IDS = [
    "LJ001-0002",  # Short (~1.9s)
    "LJ001-0008",  # Short (~2.3s)
    "LJ001-0004",  # Medium (~5.1s)
    "LJ001-0001",  # Long (~9.7s)
]


def load_metadata_map():
    meta = {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                meta[parts[0]] = parts[2]
    return meta


def validate_training_synthesis():
    print("=== Running Gate 6: Training-Utterance Synthesis Validation ===")
    meta = load_metadata_map()
    synth = HMMV1Synthesizer(MODEL_PATH)

    results = []

    for uid in TARGET_IDS:
        text = meta[uid]
        orig_wav_path = os.path.join(WAV_DIR, f"{uid}.wav")
        orig_audio, sr = sf.read(orig_wav_path)
        orig_dur = len(orig_audio) / sr
        orig_mel = wav_to_log_mel(orig_audio)
        orig_frames = orig_mel.shape[0]

        # Full synthesis pipeline
        syn_out = synth.synthesize(text, n_iter=60, momentum=0.99)
        gen_wav = syn_out["wav"]
        gen_mel = syn_out["log_mel"]
        gen_dur = syn_out["duration_s"]
        gen_frames = syn_out["n_frames"]
        phones = syn_out["phones"]
        n_states_total = len(syn_out["state_sequence"])

        # Checks
        has_nans = np.isnan(gen_mel).any() or np.isnan(gen_wav).any()
        has_infs = np.isinf(gen_mel).any() or np.isinf(gen_wav).any()
        correct_dim = (gen_mel.shape[1] == 80)

        # Numerical comparison: range and spectral energy
        gen_min = float(np.min(gen_mel))
        gen_max = float(np.max(gen_mel))
        gen_mean = float(np.mean(gen_mel))
        gen_std = float(np.std(gen_mel))

        # Save synthesized WAV
        out_wav_path = os.path.join(OUT_DIR, f"synth_{uid}.wav")
        sf.write(out_wav_path, gen_wav, SAMPLE_RATE)

        # Spectral distance (MSE between mean spectral vectors)
        orig_spec_mean = np.mean(orig_mel, axis=0)
        gen_spec_mean = np.mean(gen_mel, axis=0)
        spec_mse = float(np.mean((orig_spec_mean - gen_spec_mean) ** 2))

        results.append({
            "id": uid,
            "text": text,
            "orig_duration": orig_dur,
            "gen_duration": gen_dur,
            "orig_frames": orig_frames,
            "gen_frames": gen_frames,
            "n_phones": len(phones),
            "n_states": n_states_total,
            "correct_dim": correct_dim,
            "has_nans": has_nans,
            "has_infs": has_infs,
            "gen_min": gen_min,
            "gen_max": gen_max,
            "gen_mean": gen_mean,
            "gen_std": gen_std,
            "spec_mse": spec_mse,
            "wav_path": out_wav_path,
        })

    # Print report
    for r in results:
        print(f"\nUtterance ID: {r['id']}")
        print(f"  Text: \"{r['text']}\"")
        print(f"  Original Duration:   {r['orig_duration']:.2f}s ({r['orig_frames']} frames)")
        print(f"  Generated Duration:  {r['gen_duration']:.2f}s ({r['gen_frames']} frames)")
        print(f"  Phones/States:       {r['n_phones']} phones -> {r['n_states']} HMM states")
        print(f"  Trajectory Check:    Shape [{r['gen_frames']}, 80] | NaNs: {r['has_nans']} | Infs: {r['has_infs']}")
        print(f"  Log-Mel Statistics:  min={r['gen_min']:.3f}, max={r['gen_max']:.3f}, mean={r['gen_mean']:.3f}, std={r['gen_std']:.3f}")
        print(f"  Global Spectral MSE: {r['spec_mse']:.4f}")
        print(f"  Saved WAV:           {r['wav_path']}")

    print("\nGate 6 execution complete.")


if __name__ == "__main__":
    validate_training_synthesis()
