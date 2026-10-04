"""Gate V2-6: Final Evaluation of Frozen V1 and V2 on Test Set.

Evaluates both frozen V1 and V2 models on all 655 utterances in `data_splits/test.txt`.
Saves detailed JSON with evaluation metrics.
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
from evaluation.run_evaluation_v2 import compute_diagnostics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V1_MODEL_PATH = os.path.join(ROOT, "outputs", "models", "v1_hmm_model.json")
V2_MODEL_PATH = os.path.join(ROOT, "outputs", "models", "v2_hmm_model.json")
WAV_DIR = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "wavs")
META_PATH = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "metadata.csv")
EVAL_OUT_DIR = os.path.join(ROOT, "outputs", "evaluation")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(EVAL_OUT_DIR, "v1_wavs_test"), exist_ok=True)
os.makedirs(os.path.join(EVAL_OUT_DIR, "v2_wavs_test"), exist_ok=True)

TEST_SPLIT_PATH = os.path.join(ROOT, "data_splits", "test.txt")


def load_metadata_map():
    meta = {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                meta[parts[0]] = parts[2]
    return meta

def load_test_ids():
    ids = []
    with open(TEST_SPLIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids

def run_final_test_evaluation():
    print("=== Gate V2-6: Running Final Evaluation on Test Set ===")
    meta = load_metadata_map()
    test_ids = load_test_ids()
    print(f"Loaded {len(test_ids)} test utterances.")

    print("Initializing V1 and V2 HMM synthesizers...")
    v1_synth = HMMV1Synthesizer(V1_MODEL_PATH)
    v2_synth = HMMV2Synthesizer(V2_MODEL_PATH)

    print("Loading Whisper-tiny model for ASR WER evaluation...")
    asr_model = whisper.load_model("tiny", device="cpu")

    eval_records = []

    for idx, uid in enumerate(test_ids):
        text = meta[uid]
        orig_wav_path = os.path.join(WAV_DIR, f"{uid}.wav")
        orig_audio, sr = sf.read(orig_wav_path)
        orig_dur = len(orig_audio) / sr
        orig_mel = wav_to_log_mel(orig_audio)

        print(f"\n[{idx+1}/{len(test_ids)}] Evaluating: {uid} (Orig Dur: {orig_dur:.2f}s)")
        
        # V1 HMM
        v1_out = v1_synth.synthesize(text, n_iter=60, momentum=0.99)
        v1_wav, v1_mel, v1_seq = v1_out["wav"], v1_out["log_mel"], v1_out["state_sequence"]
        v1_dur = v1_out["duration_s"]
        sf.write(os.path.join(EVAL_OUT_DIR, "v1_wavs_test", f"v1_{uid}.wav"), v1_wav, SAMPLE_RATE)

        v1_spec_mse = compute_spectral_mse(orig_mel, v1_mel)
        v1_mcd = compute_dtw_mcd(orig_mel, v1_mel)
        asr_v1 = asr_model.transcribe(os.path.join(EVAL_OUT_DIR, "v1_wavs_test", f"v1_{uid}.wav"), language="en", fp16=False)["text"].strip()
        v1_wer = compute_wer(text, asr_v1)
        v1_diag = compute_diagnostics(v1_mel, v1_seq)

        # V2 HMM
        v2_out = v2_synth.synthesize(text, n_iter=60, momentum=0.99)
        v2_wav, v2_mel, v2_seq = v2_out["wav"], v2_out["log_mel"], v2_out["state_sequence"]
        v2_dur = v2_out["duration_s"]
        sf.write(os.path.join(EVAL_OUT_DIR, "v2_wavs_test", f"v2_{uid}.wav"), v2_wav, SAMPLE_RATE)

        v2_spec_mse = compute_spectral_mse(orig_mel, v2_mel)
        v2_mcd = compute_dtw_mcd(orig_mel, v2_mel)
        asr_v2 = asr_model.transcribe(os.path.join(EVAL_OUT_DIR, "v2_wavs_test", f"v2_{uid}.wav"), language="en", fp16=False)["text"].strip()
        v2_wer = compute_wer(text, asr_v2)
        v2_diag = compute_diagnostics(v2_mel, v2_seq)

        print(f"  WER: V1={v1_wer:.2f} | V2={v2_wer:.2f}")
        print(f"  MCD: V1={v1_mcd:.2f} | V2={v2_mcd:.2f}")

        eval_records.append({
            "id": uid,
            "text": text,
            "orig_duration": orig_dur,
            "v1_duration": v1_dur,
            "v1_dur_err": abs(v1_dur - orig_dur) / orig_dur,
            "v1_spec_mse": v1_spec_mse,
            "v1_mcd": v1_mcd,
            "v1_wer": v1_wer,
            "v1_diag": v1_diag,
            "v2_duration": v2_dur,
            "v2_dur_err": abs(v2_dur - orig_dur) / orig_dur,
            "v2_spec_mse": v2_spec_mse,
            "v2_mcd": v2_mcd,
            "v2_wer": v2_wer,
            "v2_diag": v2_diag,
            "diff_wer": v2_wer - v1_wer,
            "diff_mcd": v2_mcd - v1_mcd,
            "diff_mse": v2_spec_mse - v1_spec_mse
        })

    results_json_path = os.path.join(EVAL_OUT_DIR, "v2_final_test_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=2)
    print(f"\nSaved final test results to {results_json_path}")

if __name__ == "__main__":
    run_final_test_evaluation()
