"""Comprehensive Final Evaluation Pipeline:
1. Systems compared:
   - System A: Simple Phoneme-Average Baseline
   - System B: V1 HMM (3-state left-to-right monophone HMM)
   - System C: Oracle Reconstruction Reference (ground-truth acoustic representation ceiling)
2. Evaluation Dataset:
   - Evaluated on the 20 stratified unseen test utterances from Gate 7 (short, medium, long).
3. Metrics Computed:
   - Duration Error (mean absolute and relative %)
   - Global Log-Mel Spectral MSE
   - DTW-MCD (dB, on 16 Mel-Cepstral Coefficients excluding energy c_0)
   - ASR Word Error Rate (WER) using Whisper-tiny
   - MOS protocol documentation
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
os.makedirs(os.path.join(EVAL_OUT_DIR, "baseline_wavs"), exist_ok=True)
os.makedirs(os.path.join(EVAL_OUT_DIR, "hmm_wavs"), exist_ok=True)
os.makedirs(os.path.join(EVAL_OUT_DIR, "oracle_wavs"), exist_ok=True)

# 20 stratified test IDs from Gate 7
TEST_IDS = [
    'LJ007-0009', 'LJ043-0159', 'LJ003-0339', 'LJ033-0003', 'LJ016-0051',
    'LJ031-0067', 'LJ018-0254', 'LJ040-0037', 'LJ010-0221', 'LJ019-0267',
    'LJ044-0071', 'LJ028-0256', 'LJ015-0313', 'LJ019-0192', 'LJ029-0030',
    'LJ018-0052', 'LJ017-0077', 'LJ018-0251', 'LJ044-0114', 'LJ018-0306'
]


def load_metadata_map():
    meta = {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                meta[parts[0]] = parts[2]
    return meta


def run_full_evaluation():
    print("=== Running Comprehensive Final Evaluation Pipeline ===")
    meta = load_metadata_map()

    # Load Models
    print("Initializing V1 HMM synthesizer...")
    hmm_synth = HMMV1Synthesizer(MODEL_PATH)

    print("Initializing Simple Phoneme Baseline...")
    base_synth = PhonemeAverageBaseline(MODEL_PATH)

    print("Loading Whisper-tiny model for ASR WER evaluation...")
    asr_model = whisper.load_model("tiny", device="cpu")

    eval_records = []

    for idx, uid in enumerate(TEST_IDS):
        text = meta[uid]
        orig_wav_path = os.path.join(WAV_DIR, f"{uid}.wav")
        orig_audio, sr = sf.read(orig_wav_path)
        orig_dur = len(orig_audio) / sr
        orig_mel = wav_to_log_mel(orig_audio)

        print(f"\n[{idx+1}/{len(TEST_IDS)}] Evaluating: {uid} (Orig Dur: {orig_dur:.2f}s)")
        print(f"  Transcript: \"{text}\"")

        # -------------------------------------------------------------
        # System C: Oracle Reconstruction Reference (Acoustic Representation Ceiling)
        # -------------------------------------------------------------
        oracle_wav = log_mel_to_wav(orig_mel, n_iter=60, momentum=0.99)
        oracle_path = os.path.join(EVAL_OUT_DIR, "oracle_wavs", f"oracle_{uid}.wav")
        sf.write(oracle_path, oracle_wav / max(1.0, np.max(np.abs(oracle_wav))), SAMPLE_RATE)

        # Oracle metrics
        oracle_dur = len(oracle_wav) / SAMPLE_RATE
        oracle_dur_err = abs(oracle_dur - orig_dur) / orig_dur
        oracle_mel = wav_to_log_mel(oracle_wav)
        oracle_spec_mse = compute_spectral_mse(orig_mel, oracle_mel)
        oracle_mcd = compute_dtw_mcd(orig_mel, oracle_mel)
        # Whisper ASR on Oracle
        asr_oracle = asr_model.transcribe(oracle_path, language="en", fp16=False)["text"].strip()
        oracle_wer = compute_wer(text, asr_oracle)

        # -------------------------------------------------------------
        # System A: Simple Phoneme-Average Baseline
        # -------------------------------------------------------------
        base_out = base_synth.synthesize(text, n_iter=60, momentum=0.99)
        base_wav = base_out["wav"]
        base_mel = base_out["log_mel"]
        base_dur = base_out["duration_s"]
        base_dur_err = abs(base_dur - orig_dur) / orig_dur
        base_path = os.path.join(EVAL_OUT_DIR, "baseline_wavs", f"base_{uid}.wav")
        sf.write(base_path, base_wav, SAMPLE_RATE)

        base_spec_mse = compute_spectral_mse(orig_mel, base_mel)
        base_mcd = compute_dtw_mcd(orig_mel, base_mel)
        asr_base = asr_model.transcribe(base_path, language="en", fp16=False)["text"].strip()
        base_wer = compute_wer(text, asr_base)

        # -------------------------------------------------------------
        # System B: V1 Monophone HMM
        # -------------------------------------------------------------
        hmm_out = hmm_synth.synthesize(text, n_iter=60, momentum=0.99)
        hmm_wav = hmm_out["wav"]
        hmm_mel = hmm_out["log_mel"]
        hmm_dur = hmm_out["duration_s"]
        hmm_dur_err = abs(hmm_dur - orig_dur) / orig_dur
        hmm_path = os.path.join(EVAL_OUT_DIR, "hmm_wavs", f"hmm_{uid}.wav")
        sf.write(hmm_path, hmm_wav, SAMPLE_RATE)

        hmm_spec_mse = compute_spectral_mse(orig_mel, hmm_mel)
        hmm_mcd = compute_dtw_mcd(orig_mel, hmm_mel)
        asr_hmm = asr_model.transcribe(hmm_path, language="en", fp16=False)["text"].strip()
        hmm_wer = compute_wer(text, asr_hmm)

        print(f"  ASR Recognized Texts:")
        print(f"    Oracle:   \"{asr_oracle}\" (WER: {oracle_wer:.2f})")
        print(f"    Baseline: \"{asr_base}\" (WER: {base_wer:.2f})")
        print(f"    V1 HMM:   \"{asr_hmm}\" (WER: {hmm_wer:.2f})")
        print(f"  Acoustic MCD (dB): Baseline={base_mcd:.2f} dB | V1 HMM={hmm_mcd:.2f} dB | Oracle={oracle_mcd:.2f} dB")

        eval_records.append({
            "id": uid,
            "text": text,
            "orig_duration": orig_dur,
            # Baseline
            "base_duration": base_dur,
            "base_dur_err": base_dur_err,
            "base_spec_mse": base_spec_mse,
            "base_mcd": base_mcd,
            "base_asr": asr_base,
            "base_wer": base_wer,
            # V1 HMM
            "hmm_duration": hmm_dur,
            "hmm_dur_err": hmm_dur_err,
            "hmm_spec_mse": hmm_spec_mse,
            "hmm_mcd": hmm_mcd,
            "hmm_asr": asr_hmm,
            "hmm_wer": hmm_wer,
            # Oracle
            "oracle_duration": oracle_dur,
            "oracle_dur_err": oracle_dur_err,
            "oracle_spec_mse": oracle_spec_mse,
            "oracle_mcd": oracle_mcd,
            "oracle_asr": asr_oracle,
            "oracle_wer": oracle_wer,
        })

    # Save detailed JSON
    results_json_path = os.path.join(EVAL_OUT_DIR, "final_evaluation_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=2)

    # Compute Summary Aggregates
    summary = {
        "Metric": [
            "Synthesis Success",
            "Duration Error (%)",
            "Global Log-Mel Spectral MSE",
            "DTW-MCD (dB)",
            "Whisper-tiny WER (%)",
            "MOS (1-5 Naturalness)"
        ],
        "Baseline": [
            "20 / 20 (100%)",
            f"{np.mean([r['base_dur_err'] for r in eval_records])*100:.1f} %",
            f"{np.mean([r['base_spec_mse'] for r in eval_records]):.4f}",
            f"{np.mean([r['base_mcd'] for r in eval_records]):.2f} dB",
            f"{np.mean([r['base_wer'] for r in eval_records])*100:.1f} %",
            "Pending formal listening test"
        ],
        "V1 HMM": [
            "20 / 20 (100%)",
            f"{np.mean([r['hmm_dur_err'] for r in eval_records])*100:.1f} %",
            f"{np.mean([r['hmm_spec_mse'] for r in eval_records]):.4f}",
            f"{np.mean([r['hmm_mcd'] for r in eval_records]):.2f} dB",
            f"{np.mean([r['hmm_wer'] for r in eval_records])*100:.1f} %",
            "Pending formal listening test"
        ],
        "Oracle Reference": [
            "20 / 20 (100%)",
            f"{np.mean([r['oracle_dur_err'] for r in eval_records])*100:.1f} %",
            f"{np.mean([r['oracle_spec_mse'] for r in eval_records]):.4f}",
            f"{np.mean([r['oracle_mcd'] for r in eval_records]):.2f} dB",
            f"{np.mean([r['oracle_wer'] for r in eval_records])*100:.1f} %",
            "Reference ceiling"
        ]
    }

    print("\n" + "="*80)
    print(f"{'FINAL EVALUATION COMPARISON TABLE':^80}")
    print("="*80)
    print(f"{'Metric':<30} | {'Baseline':<14} | {'V1 HMM':<14} | {'Oracle Reference':<16}")
    print("-"*80)
    for i in range(len(summary["Metric"])):
        print(f"{summary['Metric'][i]:<30} | {summary['Baseline'][i]:<14} | {summary['V1 HMM'][i]:<14} | {summary['Oracle Reference'][i]:<16}")
    print("="*80)


if __name__ == "__main__":
    run_full_evaluation()
