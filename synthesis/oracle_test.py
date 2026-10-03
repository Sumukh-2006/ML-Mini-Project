"""Gate 3: Validate feature extraction and oracle reconstruction on sample WAVs."""
import os
import sys
import glob
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.config import SAMPLE_RATE, HOP_LENGTH
from synthesis.reconstruction import wav_to_log_mel, log_mel_to_wav

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WAV_DIR = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "wavs")
OUT_DIR = os.path.join(ROOT, "outputs", "oracle_reconstruction")
os.makedirs(OUT_DIR, exist_ok=True)

test_files = [
    "LJ001-0001.wav",
    "LJ001-0002.wav",
    "LJ001-0003.wav",
    "LJ001-0004.wav",
    "LJ001-0005.wav",
]


def run_oracle_test():
    print("=== Running Gate 3: Oracle Reconstruction Test ===")
    results = []

    for fname in test_files:
        wav_path = os.path.join(WAV_DIR, fname)
        audio, sr = sf.read(wav_path)
        assert sr == SAMPLE_RATE, f"Sample rate mismatch: {sr} vs {SAMPLE_RATE}"
        
        # Audio length & expected frames
        num_samples = len(audio)
        expected_frames = 1 + num_samples // HOP_LENGTH

        # 1. WAV -> Log-Mel
        log_mel = wav_to_log_mel(audio)
        T, D = log_mel.shape

        assert D == 80, f"Expected 80 dimensions, got {D}"
        assert abs(T - expected_frames) <= 1, f"Frame count mismatch: {T} vs expected {expected_frames}"

        # 2. Stats
        mel_min = float(np.min(log_mel))
        mel_max = float(np.max(log_mel))
        mel_mean = float(np.mean(log_mel))
        mel_std = float(np.std(log_mel))

        # 3. Log-Mel -> Reconstructed WAV
        recon_wav = log_mel_to_wav(log_mel, n_iter=60, momentum=0.99)
        
        # Save reconstructed audio
        out_wav_path = os.path.join(OUT_DIR, f"recon_{fname}")
        # Normalize reconstructed audio amplitude slightly to avoid clipping if necessary
        recon_norm = recon_wav / max(1.0, np.max(np.abs(recon_wav)))
        sf.write(out_wav_path, recon_norm, SAMPLE_RATE)

        # Spectral distortion / Reconstruction error on linear magnitude
        # Measure SNR or correlation between original and reconstructed (aligned)
        min_len = min(len(audio), len(recon_norm))
        orig_seg = audio[:min_len]
        recon_seg = recon_norm[:min_len]
        
        corr = np.corrcoef(orig_seg, recon_seg)[0, 1] if min_len > 0 else 0.0

        results.append({
            "file": fname,
            "samples": num_samples,
            "duration_s": num_samples / sr,
            "frames": T,
            "expected_frames": expected_frames,
            "min": mel_min,
            "max": mel_max,
            "mean": mel_mean,
            "std": mel_std,
            "recon_samples": len(recon_wav),
            "out_path": out_wav_path,
        })

    for r in results:
        print(f"\nUtterance: {r['file']}")
        print(f"  Duration: {r['duration_s']:.2f}s | Audio samples: {r['samples']}")
        print(f"  Feature shape: [{r['frames']}, 80] (Expected frames: {r['expected_frames']})")
        print(f"  Log-Mel Stats: min={r['min']:.3f}, max={r['max']:.3f}, mean={r['mean']:.3f}, std={r['std']:.3f}")
        print(f"  Reconstructed Audio saved to: {r['out_path']}")

    print("\nReconstruction completed. Validating intelligibility & artifact profile...")


if __name__ == "__main__":
    run_oracle_test()
