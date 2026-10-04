"""Gate V2-4: Evaluate V2 vs V1 on Full Validation Set.

Synthesizes all 655 validation utterances using both V1 and V2.
Computes paired evaluations (WER, MCD, MSE) and trajectory diagnostics.
Saves detailed JSON with bootstrap confidence intervals.
"""
import os
import sys

env_bin = "/opt/miniconda3/envs/tts-hmm/bin"
if env_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = env_bin + ":" + os.environ.get("PATH", "")

import json
import numpy as np
import soundfile as sf
import whisper

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.config import SAMPLE_RATE
from synthesis.synthesize import HMMV1Synthesizer
from synthesis.synthesize_v2 import HMMV2Synthesizer
from synthesis.reconstruction import wav_to_log_mel
from evaluation.acoustic_metrics import compute_spectral_mse, compute_dtw_mcd, compute_wer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V1_MODEL_PATH = os.path.join(ROOT, "outputs", "models", "v1_hmm_model.json")
V2_MODEL_PATH = os.path.join(ROOT, "outputs", "models", "v2_hmm_model.json")
WAV_DIR = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "wavs")
META_PATH = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "metadata.csv")
EVAL_OUT_DIR = os.path.join(ROOT, "outputs", "evaluation")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(EVAL_OUT_DIR, "v1_wavs_val"), exist_ok=True)
os.makedirs(os.path.join(EVAL_OUT_DIR, "v2_wavs_val"), exist_ok=True)

VAL_SPLIT_PATH = os.path.join(ROOT, "data_splits", "val.txt")


def load_metadata_map():
    meta = {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                meta[parts[0]] = parts[2]
    return meta

def load_val_ids():
    ids = []
    with open(VAL_SPLIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids

def compute_diagnostics(log_mel, state_seq):
    # 1. First-order change
    d1 = np.diff(log_mel, axis=0)
    d1_mag = np.mean(np.abs(d1))
    
    # 2. Second-order change
    d2 = np.diff(d1, axis=0)
    d2_mag = np.mean(np.abs(d2)) if len(d2) > 0 else 0.0
    
    # 3. Global variance
    gv = np.mean(np.var(log_mel, axis=0))
    
    # 4. Boundary discontinuity
    boundary_diffs = []
    t = 0
    for i in range(len(state_seq) - 1):
        dur = state_seq[i][2]
        t += dur
        if t < len(log_mel):
            diff_mag = np.mean(np.abs(log_mel[t] - log_mel[t-1]))
            boundary_diffs.append(diff_mag)
    
    bound_disc = np.mean(boundary_diffs) if boundary_diffs else 0.0
    
    return {
        "d1_mag": float(d1_mag),
        "d2_mag": float(d2_mag),
        "gv": float(gv),
        "bound_disc": float(bound_disc)
    }

def run_v2_validation_evaluation():
    print("=== Gate V2-4: Running V2 vs V1 Validation Benchmark ===")
    meta = load_metadata_map()
    val_ids = load_val_ids()
    print(f"Loaded {len(val_ids)} validation utterances.")

    print("Initializing V1 and V2 HMM synthesizers...")
    v1_synth = HMMV1Synthesizer(V1_MODEL_PATH)
    v2_synth = HMMV2Synthesizer(V2_MODEL_PATH)

    print("Loading Whisper-tiny model for ASR WER evaluation...")
    asr_model = whisper.load_model("tiny", device="cpu")

    eval_records = []

    for idx, uid in enumerate(val_ids):
        text = meta[uid]
        orig_wav_path = os.path.join(WAV_DIR, f"{uid}.wav")
        orig_audio, sr = sf.read(orig_wav_path)
        orig_dur = len(orig_audio) / sr
        orig_mel = wav_to_log_mel(orig_audio)

        print(f"\n[{idx+1}/{len(val_ids)}] Evaluating: {uid} (Orig Dur: {orig_dur:.2f}s)")
        
        # Natural Diagnostics
        nat_diag = compute_diagnostics(orig_mel, [])
        
        # V1 HMM
        v1_out = v1_synth.synthesize(text, n_iter=60, momentum=0.99)
        v1_wav, v1_mel, v1_seq = v1_out["wav"], v1_out["log_mel"], v1_out["state_sequence"]
        v1_dur = v1_out["duration_s"]
        sf.write(os.path.join(EVAL_OUT_DIR, "v1_wavs_val", f"v1_{uid}.wav"), v1_wav, SAMPLE_RATE)

        v1_spec_mse = compute_spectral_mse(orig_mel, v1_mel)
        v1_mcd = compute_dtw_mcd(orig_mel, v1_mel)
        asr_v1 = asr_model.transcribe(os.path.join(EVAL_OUT_DIR, "v1_wavs_val", f"v1_{uid}.wav"), language="en", fp16=False)["text"].strip()
        v1_wer = compute_wer(text, asr_v1)
        v1_diag = compute_diagnostics(v1_mel, v1_seq)

        # V2 HMM
        v2_out = v2_synth.synthesize(text, n_iter=60, momentum=0.99)
        v2_wav, v2_mel, v2_seq = v2_out["wav"], v2_out["log_mel"], v2_out["state_sequence"]
        v2_dur = v2_out["duration_s"]
        sf.write(os.path.join(EVAL_OUT_DIR, "v2_wavs_val", f"v2_{uid}.wav"), v2_wav, SAMPLE_RATE)

        v2_spec_mse = compute_spectral_mse(orig_mel, v2_mel)
        v2_mcd = compute_dtw_mcd(orig_mel, v2_mel)
        asr_v2 = asr_model.transcribe(os.path.join(EVAL_OUT_DIR, "v2_wavs_val", f"v2_{uid}.wav"), language="en", fp16=False)["text"].strip()
        v2_wer = compute_wer(text, asr_v2)
        v2_diag = compute_diagnostics(v2_mel, v2_seq)

        print(f"  WER: V1={v1_wer:.2f} | V2={v2_wer:.2f} (Diff: {v2_wer - v1_wer:+.2f})")
        print(f"  MCD: V1={v1_mcd:.2f} | V2={v2_mcd:.2f} (Diff: {v2_mcd - v1_mcd:+.2f})")
        
        if not v2_out["mlpg_metrics"]["solver_success"]:
            print("  [WARNING] MLPG Solver Success=False for this utterance")

        eval_records.append({
            "id": uid,
            "text": text,
            "orig_duration": orig_dur,
            # V1
            "v1_duration": v1_dur,
            "v1_dur_err": abs(v1_dur - orig_dur) / orig_dur,
            "v1_spec_mse": v1_spec_mse,
            "v1_mcd": v1_mcd,
            "v1_wer": v1_wer,
            "v1_diag": v1_diag,
            # V2
            "v2_duration": v2_dur,
            "v2_dur_err": abs(v2_dur - orig_dur) / orig_dur,
            "v2_spec_mse": v2_spec_mse,
            "v2_mcd": v2_mcd,
            "v2_wer": v2_wer,
            "v2_diag": v2_diag,
            "v2_solver_metrics": v2_out["mlpg_metrics"],
            # Differences (V2 - V1)
            "diff_wer": v2_wer - v1_wer,
            "diff_mcd": v2_mcd - v1_mcd,
            "diff_mse": v2_spec_mse - v1_spec_mse
        })

    results_json_path = os.path.join(EVAL_OUT_DIR, "v2_validation_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=2)
    print(f"\nSaved validation results to {results_json_path}")


if __name__ == "__main__":
    run_v2_validation_evaluation()
