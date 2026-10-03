"""Deterministic text processing for the V1 TTS pipeline.

Input : the LJ Speech *Normalized Transcription* (metadata.csv column 3) or any new sentence.
Output: a list of tokens, each either a word (lowercase, [a-z']+) or the SIL marker.

The same function is used for training and inference.

Rules (derived from the inventory of the 13,100 normalized transcripts):
  1. Typographic quotes/apostrophes -> ASCII  (U+201C/U+201D -> ", U+2019/U+2018 -> ').
  2. Latin letters with diacritics -> base letter via NFKD (u-umlaut -> u, e-acute -> e, ...).
     Lossy for the few foreign words (e.g. French "grace", "Muller"), documented limitation.
  3. Acronyms: a run of >=2 capital letters in the ORIGINAL text (FBI, CIA, USSR, WDSU) is
     detected before lowercasing and emitted as the token "#fbi"; G2P spells it letter by
     letter. (Lone capitals such as the "D" in "LL.D." are ordinary one-letter words.)
     Known ambiguity: "SAC" may be spoken as a word; it is spelled. All-caps input text at
     inference would be spelled out too - use normal sentence case.
  3b. Lowercase everything else.
  4. Phrase punctuation  , ; : . ? ! ( )  -> SIL (consecutive SILs collapsed).
  5. Double quotes, square brackets -> removed (no acoustic realisation of their own).
  6. Hyphen -> word separator (bric-a-brac -> bric a brac).
  7. Apostrophes are kept inside words (cabman's, don't); leading/trailing apostrophes stripped.
  8. Every utterance starts and ends with SIL (clips contain leading/trailing silence).
  9. Any other character (digits, symbols, other scripts) -> ValueError. Nothing is dropped silently.
"""
import re
import unicodedata

SIL = "<sil>"

_CHAR_MAP = {
    "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'",
}
_PHRASE_PUNCT = set(",;:.?!()")
_REMOVED = set('"[]')
_ACRONYM_RE = re.compile(r"(?<![A-Za-z'])[A-Z]{2,}(?![A-Za-z'])")
ACRONYM_PREFIX = "#"


def _to_ascii_letters(text):
    out = []
    for ch in text:
        ch = _CHAR_MAP.get(ch, ch)
        if ord(ch) < 128:
            out.append(ch)
            continue
        decomposed = unicodedata.normalize("NFKD", ch)
        base = "".join(c for c in decomposed if not unicodedata.combining(c))
        out.append(base)
    return "".join(out)


def text_to_tokens(text):
    """Return list of tokens: words (str) and SIL."""
    text = _to_ascii_letters(text)
    acronyms = {m.start(): m.end() for m in _ACRONYM_RE.finditer(text)}
    text_l = text.lower()
    tokens = []
    buf = []

    def flush():
        if buf:
            word = "".join(buf).strip("'")
            buf.clear()
            if word:
                tokens.append(word)

    i = 0
    while i < len(text_l):
        ch = text_l[i]
        if i in acronyms:
            flush()
            tokens.append(ACRONYM_PREFIX + text_l[i:acronyms[i]])
            i = acronyms[i]
            continue
        if "a" <= ch <= "z" or ch == "'":
            buf.append(ch)
        elif ch in _PHRASE_PUNCT:
            flush()
            tokens.append(SIL)
        elif ch.isspace() or ch == "-" or ch in _REMOVED:
            flush()
        else:
            raise ValueError(f"Unsupported character {ch!r} (U+{ord(ch):04X}) in text: {text!r}")
        i += 1
    flush()

    # collapse consecutive SIL, enforce SIL at both ends
    collapsed = []
    for t in tokens:
        if t == SIL and collapsed and collapsed[-1] == SIL:
            continue
        collapsed.append(t)
    if not collapsed or collapsed[0] != SIL:
        collapsed.insert(0, SIL)
    if collapsed[-1] != SIL:
        collapsed.append(SIL)
    return collapsed
