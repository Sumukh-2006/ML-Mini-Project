"""Gate V2-0: Evaluate V1 Baseline on Full Validation Set.

Evaluates the completely frozen V1 model (and Oracle/Baseline for reference)
on all 655 utterances in data_splits/val.txt.
Saves results to outputs/evaluation/v1_validation_results.json.
"""
import os
import sys

# Ensure environment binary path is present for ffmpeg
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
from models.baseline import PhonemeAverageBaseline
from synthesis.reconstruction import wav_to_log_mel, log_mel_to_wav
from evaluation.acoustic_metrics import compute_spectral_mse, compute_dtw_mcd, compute_wer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "outputs", "models", "v1_hmm_model.json")
WAV_DIR = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "wavs")
META_PATH = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "metadata.csv")
EVAL_OUT_DIR = os.path.join(ROOT, "outputs", "evaluation")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(EVAL_OUT_DIR, "baseline_wavs_val"), exist_ok=True)
os.makedirs(os.path.join(EVAL_OUT_DIR, "hmm_wavs_val"), exist_ok=True)
os.makedirs(os.path.join(EVAL_OUT_DIR, "oracle_wavs_val"), exist_ok=True)

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


def run_v1_validation_evaluation():
    print("=== Gate V2-0: Running V1 Evaluation on Validation Set ===")
    meta = load_metadata_map()
    val_ids = load_val_ids()
    print(f"Loaded {len(val_ids)} validation utterances.")

    print("Initializing V1 HMM synthesizer...")
    hmm_synth = HMMV1Synthesizer(MODEL_PATH)

    print("Initializing Simple Phoneme Baseline...")
    base_synth = PhonemeAverageBaseline(MODEL_PATH)

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
        
        # Oracle Reference
        oracle_wav = log_mel_to_wav(orig_mel, n_iter=60, momentum=0.99)
        oracle_path = os.path.join(EVAL_OUT_DIR, "oracle_wavs_val", f"oracle_{uid}.wav")
        sf.write(oracle_path, oracle_wav / max(1.0, np.max(np.abs(oracle_wav))), SAMPLE_RATE)

        oracle_dur = len(oracle_wav) / SAMPLE_RATE
        oracle_dur_err = abs(oracle_dur - orig_dur) / orig_dur
        oracle_mel = wav_to_log_mel(oracle_wav)
        oracle_spec_mse = compute_spectral_mse(orig_mel, oracle_mel)
        oracle_mcd = compute_dtw_mcd(orig_mel, oracle_mel)
        asr_oracle = asr_model.transcribe(oracle_path, language="en", fp16=False)["text"].strip()
        oracle_wer = compute_wer(text, asr_oracle)

        # Baseline
        base_out = base_synth.synthesize(text, n_iter=60, momentum=0.99)
        base_wav = base_out["wav"]
        base_mel = base_out["log_mel"]
        base_dur = base_out["duration_s"]
        base_dur_err = abs(base_dur - orig_dur) / orig_dur
        base_path = os.path.join(EVAL_OUT_DIR, "baseline_wavs_val", f"base_{uid}.wav")
        sf.write(base_path, base_wav, SAMPLE_RATE)

        base_spec_mse = compute_spectral_mse(orig_mel, base_mel)
        base_mcd = compute_dtw_mcd(orig_mel, base_mel)
        asr_base = asr_model.transcribe(base_path, language="en", fp16=False)["text"].strip()
        base_wer = compute_wer(text, asr_base)

        # V1 HMM
        hmm_out = hmm_synth.synthesize(text, n_iter=60, momentum=0.99)
        hmm_wav = hmm_out["wav"]
        hmm_mel = hmm_out["log_mel"]
        hmm_dur = hmm_out["duration_s"]
        hmm_dur_err = abs(hmm_dur - orig_dur) / orig_dur
        hmm_path = os.path.join(EVAL_OUT_DIR, "hmm_wavs_val", f"hmm_{uid}.wav")
        sf.write(hmm_path, hmm_wav, SAMPLE_RATE)

        hmm_spec_mse = compute_spectral_mse(orig_mel, hmm_mel)
        hmm_mcd = compute_dtw_mcd(orig_mel, hmm_mel)
        asr_hmm = asr_model.transcribe(hmm_path, language="en", fp16=False)["text"].strip()
        hmm_wer = compute_wer(text, asr_hmm)

        print(f"  WER: Base={base_wer:.2f} | V1 HMM={hmm_wer:.2f} | Oracle={oracle_wer:.2f}")
        print(f"  MCD (dB): Base={base_mcd:.2f} | V1 HMM={hmm_mcd:.2f} | Oracle={oracle_mcd:.2f}")

        eval_records.append({
            "id": uid,
            "text": text,
            "orig_duration": orig_dur,
            "base_duration": base_dur,
            "base_dur_err": base_dur_err,
            "base_spec_mse": base_spec_mse,
            "base_mcd": base_mcd,
            "base_asr": asr_base,
            "base_wer": base_wer,
            "hmm_duration": hmm_dur,
            "hmm_dur_err": hmm_dur_err,
            "hmm_spec_mse": hmm_spec_mse,
            "hmm_mcd": hmm_mcd,
            "hmm_asr": asr_hmm,
            "hmm_wer": hmm_wer,
            "oracle_duration": oracle_dur,
            "oracle_dur_err": oracle_dur_err,
            "oracle_spec_mse": oracle_spec_mse,
            "oracle_mcd": oracle_mcd,
            "oracle_asr": asr_oracle,
            "oracle_wer": oracle_wer,
        })

    results_json_path = os.path.join(EVAL_OUT_DIR, "v1_validation_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=2)
    print(f"\nSaved validation results to {results_json_path}")

    # Compute Summary Aggregates
    summary = {
        "Metric": [
            "Duration Error (%)",
            "Global Log-Mel Spectral MSE",
            "DTW-MCD (dB)",
            "Whisper-tiny WER (%)"
        ],
        "Baseline": [
            f"{np.mean([r['base_dur_err'] for r in eval_records])*100:.1f} %",
            f"{np.mean([r['base_spec_mse'] for r in eval_records]):.4f}",
            f"{np.mean([r['base_mcd'] for r in eval_records]):.2f} dB",
            f"{np.mean([r['base_wer'] for r in eval_records])*100:.1f} %"
        ],
        "V1 HMM": [
            f"{np.mean([r['hmm_dur_err'] for r in eval_records])*100:.1f} %",
            f"{np.mean([r['hmm_spec_mse'] for r in eval_records]):.4f}",
            f"{np.mean([r['hmm_mcd'] for r in eval_records]):.2f} dB",
            f"{np.mean([r['hmm_wer'] for r in eval_records])*100:.1f} %"
        ],
        "Oracle Reference": [
            f"{np.mean([r['oracle_dur_err'] for r in eval_records])*100:.1f} %",
            f"{np.mean([r['oracle_spec_mse'] for r in eval_records]):.4f}",
            f"{np.mean([r['oracle_mcd'] for r in eval_records]):.2f} dB",
            f"{np.mean([r['oracle_wer'] for r in eval_records])*100:.1f} %"
        ]
    }

    print("\n" + "="*80)
    print(f"{'V1 VALIDATION EVALUATION COMPARISON TABLE (655 UTTERANCES)':^80}")
    print("="*80)
    print(f"{'Metric':<30} | {'Baseline':<14} | {'V1 HMM':<14} | {'Oracle Reference':<16}")
    print("-"*80)
    for i in range(len(summary["Metric"])):
        print(f"{summary['Metric'][i]:<30} | {summary['Baseline'][i]:<14} | {summary['V1 HMM'][i]:<14} | {summary['Oracle Reference'][i]:<16}")
    print("="*80)


if __name__ == "__main__":
    run_v1_validation_evaluation()
