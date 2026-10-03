"""Deterministic English G2P (CMU Pronouncing Dictionary / ARPAbet) for the V1 TTS pipeline.

Resolution order per word:
  0. Acronym token ('#fbi' from text_processing) -> spelled letter by letter using the fixed
     LETTER_NAMES table (source 'acronym_spelling').
  1. CMUdict lookup, FIRST listed pronunciation (deterministic; homographs are NOT
     disambiguated by POS in V1 - documented limitation).
  2. Possessive rule: "<base>'s" with <base> in CMUdict -> base phones + {Z | S | IH Z}.
  3. Otherwise OOV -> g2p_en neural predictor (numpy GRU, deterministic, no sampling).
Every non-dictionary word is recorded with its source; nothing is dropped.

Phone inventory: 39 stress-free ARPAbet phones + SIL = 40 symbols, fixed ids (SIL = 0).
Stress digits are removed for the HMM symbols (monophone V1, avoids splitting data 3x);
the stress-marked sequence is kept alongside for reference/V2.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_processing import ACRONYM_PREFIX, SIL, text_to_tokens  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "metadata.csv")

ARPABET_39 = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
SIL_PHONE = "SIL"
PHONE_LIST = [SIL_PHONE] + sorted(ARPABET_39)
PHONE_TO_ID = {p: i for i, p in enumerate(PHONE_LIST)}

# Pronunciation of letter NAMES (CMUdict's entry for a lone letter is not always the letter name,
# e.g. 'a' -> AH0), so this table is fixed explicitly.
LETTER_NAMES = {
    "a": ["EY1"], "b": ["B", "IY1"], "c": ["S", "IY1"], "d": ["D", "IY1"], "e": ["IY1"],
    "f": ["EH1", "F"], "g": ["JH", "IY1"], "h": ["EY1", "CH"], "i": ["AY1"],
    "j": ["JH", "EY1"], "k": ["K", "EY1"], "l": ["EH1", "L"], "m": ["EH1", "M"],
    "n": ["EH1", "N"], "o": ["OW1"], "p": ["P", "IY1"], "q": ["K", "Y", "UW1"],
    "r": ["AA1", "R"], "s": ["EH1", "S"], "t": ["T", "IY1"], "u": ["Y", "UW1"],
    "v": ["V", "IY1"], "w": ["D", "AH1", "B", "AH0", "L", "Y", "UW0"], "x": ["EH1", "K", "S"],
    "y": ["W", "AY1"], "z": ["Z", "IY1"],
}

_VOICELESS = {"P", "T", "K", "F", "TH"}
_SIBILANT = {"S", "Z", "SH", "ZH", "CH", "JH"}
_STRESS_RE = re.compile(r"\d$")


def strip_stress(p):
    return _STRESS_RE.sub("", p)


class Pronouncer:
    def __init__(self):
        from g2p_en import G2p  # imported lazily (downloads NLTK data on first use)
        self._g2p = G2p()
        self._cmu = self._g2p.cmu

    def word_to_phones(self, word):
        """Return (stress-marked ARPAbet phones, source)."""
        if word.startswith(ACRONYM_PREFIX):
            letters = word[len(ACRONYM_PREFIX):]
            return [p for ch in letters for p in LETTER_NAMES[ch]], "acronym_spelling"
        if word in self._cmu:
            return list(self._cmu[word][0]), "cmudict"
        if word.endswith("'s") and word[:-2] in self._cmu:
            base = list(self._cmu[word[:-2]][0])
            last = strip_stress(base[-1])
            if last in _SIBILANT:
                return base + ["IH0", "Z"], "possessive_rule"
            return base + (["S"] if last in _VOICELESS else ["Z"]), "possessive_rule"
        letters = re.sub(r"[^a-z]", "", word)
        if not letters:
            raise ValueError(f"Word without letters cannot be pronounced: {word!r}")
        return list(self._g2p.predict(letters)), "oov_g2p"

    def text_to_phonemes(self, text):
        """Full deterministic text -> phoneme conversion (same call for train and inference)."""
        tokens = text_to_tokens(text)
        phones_stress, phones, word_idx, oov = [], [], [], []
        w = 0
        for tok in tokens:
            if tok == SIL:
                phones_stress.append(SIL_PHONE)
                phones.append(SIL_PHONE)
                word_idx.append(-1)
                continue
            ph, src = self.word_to_phones(tok)
            if src != "cmudict":
                oov.append({"word": tok, "source": src, "phones": " ".join(ph)})
            for p in ph:
                base = strip_stress(p)
                if base not in PHONE_TO_ID:
                    raise ValueError(f"Unknown phoneme {p!r} for word {tok!r}")
                phones_stress.append(p)
                phones.append(base)
                word_idx.append(w)
            w += 1
        return {
            "tokens": tokens,
            "phones": phones,
            "phones_stress": phones_stress,
            "phone_ids": [PHONE_TO_ID[p] for p in phones],
            "word_idx": word_idx,
            "oov": oov,
        }


def load_metadata():
    with open(META, encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of first utterances (0 = all)")
    ap.add_argument("--out", default=None, help="optional JSONL output path")
    ap.add_argument("--show", type=int, default=10, help="how many examples to print")
    args = ap.parse_args()

    rows = load_metadata()
    if args.n > 0:
        rows = rows[: args.n]
    pr = Pronouncer()

    out_f = None
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out_f = open(args.out, "w", encoding="utf-8")

    n_words = n_oov_tokens = 0
    phone_counts = Counter()
    oov_words, src_counts = Counter(), Counter()
    lens, failures = [], []
    for k, (utt_id, _raw, norm) in enumerate(rows):
        try:
            r = pr.text_to_phonemes(norm)
        except ValueError as e:
            failures.append((utt_id, str(e)))
            continue
        words = [t for t in r["tokens"] if t != SIL]
        n_words += len(words)
        n_oov_tokens += len(r["oov"])
        for o in r["oov"]:
            oov_words[(o["word"], o["source"], o["phones"])] += 1
            src_counts[o["source"]] += 1
        phone_counts.update(r["phones"])
        lens.append(len(r["phones"]))
        if k < args.show:
            print(f"\n[{utt_id}] {norm}")
            print("  tokens:", " ".join(r["tokens"]))
            print("  phones:", " ".join(r["phones_stress"]))
            if r["oov"]:
                print("  NON-DICT:", r["oov"])
        if out_f:
            out_f.write(json.dumps({"id": utt_id, "text": norm, **r}) + "\n")
    if out_f:
        out_f.close()

    print("\n=== Statistics ===")
    print(f"utterances processed: {len(rows) - len(failures)} / {len(rows)} | failures: {len(failures)}")
    for f_ in failures[:10]:
        print("  FAIL", f_)
    print(f"word tokens: {n_words} | non-dictionary tokens: {n_oov_tokens} "
          f"({100 * n_oov_tokens / max(n_words, 1):.2f}%) | by source: {dict(src_counts)}")
    print(f"distinct non-dictionary words: {len(oov_words)}")
    for (w, s, p), c in oov_words.most_common(40):
        print(f"  {w:20s} x{c:<4d} {s:16s} {p}")
    if lens:
        print(f"phones/utt: min {min(lens)} mean {sum(lens) / len(lens):.1f} max {max(lens)}")
    unseen = [p for p in PHONE_LIST if p not in phone_counts]
    print(f"phone inventory seen: {len(phone_counts)}/{len(PHONE_LIST)} | unseen: {unseen}")
    print("phone counts:", dict(phone_counts.most_common()))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
