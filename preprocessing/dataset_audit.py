"""Gate 1: audit LJSpeech metadata <-> WAV pairing and audio properties (read-only)."""
import csv
import os
import sys
import wave
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET = os.path.join(ROOT, "Dataset", "LJSpeech-1.1")


def main():
    meta_path = os.path.join(DATASET, "metadata.csv")
    wav_dir = os.path.join(DATASET, "wavs")

    # QUOTE_NONE: transcripts contain literal double quotes
    with open(meta_path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE))
    bad_rows = [r for r in rows if len(r) != 3]
    ids = [r[0] for r in rows]
    wav_files = sorted(x[:-4] for x in os.listdir(wav_dir) if x.endswith(".wav"))

    missing = sorted(set(ids) - set(wav_files))
    extra = sorted(set(wav_files) - set(ids))
    dups = [k for k, v in Counter(ids).items() if v > 1]

    print(f"metadata rows: {len(rows)} | malformed rows: {len(bad_rows)}")
    print(f"wav files: {len(wav_files)} | duplicate ids: {len(dups)}")
    print(f"ids without wav: {len(missing)} | wavs without id: {len(extra)}")

    fmt, durs, corrupt, peaks = Counter(), [], [], []
    for i in ids:
        try:
            with wave.open(os.path.join(wav_dir, i + ".wav"), "rb") as w:
                fmt[(w.getnchannels(), w.getsampwidth() * 8, w.getframerate())] += 1
                durs.append(w.getnframes() / w.getframerate())
                x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                peaks.append(np.abs(x.astype(np.int32)).max() / 32768.0)
        except Exception as e:  # noqa: BLE001
            corrupt.append((i, str(e)))

    print(f"(channels, bits, sr) counts: {dict(fmt)}")
    print(f"corrupt files: {len(corrupt)}")
    d = np.array(durs)
    print(f"duration s: min {d.min():.2f} mean {d.mean():.2f} max {d.max():.2f} total {d.sum()/3600:.2f} h")
    p = np.array(peaks)
    print(f"peak amplitude: min {p.min():.3f} max {p.max():.3f} | clipped(>=0.999): {(p >= 0.999).sum()}")
    tl = np.array([len(r[2]) for r in rows if len(r) == 3])
    print(f"normalized transcript chars: min {tl.min()} mean {tl.mean():.1f} max {tl.max()}")
    print(f"non-ASCII transcriptions: {sum(1 for r in rows if len(r) == 3 and not r[1].isascii())}")

    ok = not (bad_rows or missing or extra or dups or corrupt) and len(rows) == 13100
    print("GATE 1:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
