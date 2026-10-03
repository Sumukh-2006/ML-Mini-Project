"""Phase 2 investigation: inventory of non-ASCII characters in LJSpeech metadata (read-only)."""
import csv
import os
import unicodedata
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "metadata.csv")


def main():
    with open(META, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE))

    for col, name in [(1, "Transcription"), (2, "Normalized")]:
        chars, utt = Counter(), Counter()
        n_utts = 0
        for r in rows:
            na = [c for c in r[col] if ord(c) > 127]
            if na:
                n_utts += 1
                for c in set(na):
                    utt[c] += 1
            chars.update(na)
        print(f"\n== {name}: {n_utts} utterances with non-ASCII, {sum(chars.values())} chars ==")
        for c, n in chars.most_common():
            print(f"U+{ord(c):04X} {c!r} {unicodedata.name(c, '?')} | occurrences {n} | utterances {utt[c]}")

    # Also: ASCII character inventory of the normalized column
    inv = Counter(c for r in rows for c in r[2] if ord(c) < 128)
    print("\n== ASCII inventory (Normalized) ==")
    print(sorted(inv.items(), key=lambda kv: -kv[1]))

    print("\n== Sample utterances with non-ASCII (Normalized) ==")
    shown = 0
    for r in rows:
        if any(ord(c) > 127 for c in r[2]) and shown < 12:
            print(r[0], "|", r[2][:150])
            shown += 1


if __name__ == "__main__":
    main()
