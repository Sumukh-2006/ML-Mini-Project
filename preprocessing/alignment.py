"""Phase 3: forced-alignment glue around Montreal Forced Aligner (MFA).

Subcommands (run in the `tts-hmm` env; MFA itself runs in the separate `aligner` env):

  prepare --name NAME --n N     build MFA corpus (symlinked WAVs + .lab) and a dictionary
                                derived from OUR canonical G2P output (outputs/g2p/phonemes.jsonl)
  check   --name NAME           parse MFA TextGrids, validate them, compare with the G2P
                                sequence, map phones -> frames -> 3 states, write reports

Design notes
  * The raw dataset is never modified: the corpus directory only contains symlinks to the WAVs
    and newly written .lab transcript files.
  * The MFA dictionary is generated from our G2P, so MFA aligns EXACTLY the phone sequence our
    canonical G2P produced. The canonical G2P output is never altered.
  * NOTATION MISMATCH (found by inspecting the model): MFA's english_us_arpa acoustic model uses
    STRESS-MARKED vowels (AA0/AA1/AA2, ...) and has no stress-free vowels, whereas our canonical
    HMM symbols are stress-free. Conversion layer: the dictionary is written from the canonical
    `phones_stress` field (standard CMU ARPAbet), and TextGrid labels are stripped of the stress
    digit when read back. Both the stressed and the stress-free sequences are verified.
  * Acronym tokens ('#fbi') are written to MFA as the dictionary key 'acr_fbi' (deterministic
    conversion; '#' is avoided in transcripts/dictionaries).
  * Punctuation SIL tokens are NOT forced into the transcript. MFA decides where silence occurs;
    its silence intervals become SIL segments. We report how well they match punctuation.
  * Frame mapping uses the approved formula  frame = int(t * SR / HOP).  The STFT (center=True)
    has T = 1 + n_samples // HOP frames, so the final boundary is set to T (the extra last
    frame belongs to the last interval) - this is the only deviation and keeps a full partition.
  * Each phone interval with n frames is split into 3 states by state(i) = (3*i)//n, i=0..n-1.
    Phones with n < 3 frames therefore leave one or more states EMPTY; these are counted and
    reported, not hidden.
"""
import argparse
import json
import os
import re
import sys
import wave
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HOP_LENGTH, N_STATES, SAMPLE_RATE  # noqa: E402
from g2p import PHONE_TO_ID, SIL_PHONE, strip_stress  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WAV_DIR = os.path.join(ROOT, "Dataset", "LJSpeech-1.1", "wavs")
G2P_JSONL = os.path.join(ROOT, "outputs", "g2p", "phonemes.jsonl")
MFA_DIR = os.path.join(ROOT, "outputs", "mfa")
ALIGN_DIR = os.path.join(ROOT, "outputs", "alignments")
SPEAKER = "LJ"

# MFA labels that mean "no speech phone". 'spn' (spoken noise) is NOT silence: it is flagged.
SILENCE_LABELS = {"", "sil", "SIL"}
TIME_TOL = 1e-3  # seconds


def mfa_word(token):
    return "acr_" + token[1:] if token.startswith("#") else token


def load_g2p(n=None, ids=None):
    recs = []
    with open(G2P_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if ids is not None and r["id"] not in ids:
                continue
            recs.append(r)
            if n and len(recs) >= n:
                break
    return recs


def wav_info(utt_id):
    with wave.open(os.path.join(WAV_DIR, utt_id + ".wav"), "rb") as w:
        return w.getnframes(), w.getframerate()


def load_model_phones():
    """Phone symbols of the MFA english_us_arpa acoustic model (read from its zip, no MFA needed)."""
    import zipfile
    zp = os.path.join(MFA_DIR, "mfa_root", "pretrained_models", "acoustic", "english_us_arpa.zip")
    with zipfile.ZipFile(zp) as z:
        txt = z.read("english_us_arpa/phones.txt").decode()
    return {ln.split()[0] for ln in txt.splitlines() if ln.strip()} - {"<eps>", "sil", "spn"}


# ----------------------------------------------------------------------------- prepare
def cmd_prepare(args):
    recs = load_g2p(n=args.n)
    corpus = os.path.join(MFA_DIR, f"corpus_{args.name}", SPEAKER)
    os.makedirs(corpus, exist_ok=True)
    lexicon = {}
    for r in recs:
        words = [t for t in r["tokens"] if t != "<sil>"]
        by_word = {}
        for ph, wi in zip(r["phones_stress"], r["word_idx"]):
            if wi >= 0:
                by_word.setdefault(wi, []).append(ph)
        assert len(by_word) == len(words), r["id"]
        for wi, tok in enumerate(words):
            key, pron = mfa_word(tok), " ".join(by_word[wi])
            if lexicon.setdefault(key, pron) != pron:
                raise RuntimeError(f"Non-deterministic pronunciation for {key!r}")
        with open(os.path.join(corpus, r["id"] + ".lab"), "w", encoding="utf-8") as f:
            f.write(" ".join(mfa_word(t) for t in words) + "\n")
        link = os.path.join(corpus, r["id"] + ".wav")
        if not os.path.lexists(link):
            os.symlink(os.path.join(WAV_DIR, r["id"] + ".wav"), link)
    dict_path = os.path.join(MFA_DIR, f"dict_{args.name}.dict")
    with open(dict_path, "w", encoding="utf-8") as f:
        for w in sorted(lexicon):
            f.write(f"{w}\t{lexicon[w]}\n")
    used = Counter(p for pr in lexicon.values() for p in pr.split())
    model_phones = load_model_phones()
    not_in_model = sorted(set(used) - model_phones)
    print(f"corpus: {corpus} ({len(recs)} utterances)")
    print(f"dictionary: {dict_path} ({len(lexicon)} entries); phones used: {len(used)}")
    print(f"phones not in MFA acoustic model: {not_in_model if not_in_model else 'none'}")
    if not_in_model:
        raise RuntimeError(f"Dictionary phones missing from MFA model: {not_in_model}")


# ----------------------------------------------------------------------------- TextGrid
def parse_textgrid(path):
    """Parse a long-format TextGrid -> {tier_name: [(xmin, xmax, text), ...]}."""
    tiers, cur, cur_iv = {}, None, {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            m = re.match(r'name = "(.*)"$', line)
            if m:
                cur = m.group(1)
                tiers[cur] = []
                continue
            if re.match(r"intervals \[\d+\]:", line):
                cur_iv = {}
                continue
            m = re.match(r"(xmin|xmax) = ([\d.eE+-]+)$", line)
            if m and cur is not None and cur_iv is not None:
                cur_iv[m.group(1)] = float(m.group(2))
                continue
            m = re.match(r'text = "(.*)"$', line)
            if m and cur is not None:
                cur_iv["text"] = m.group(1).replace('""', '"')
                tiers[cur].append((cur_iv["xmin"], cur_iv["xmax"], cur_iv["text"]))
                cur_iv = {}
    return tiers


# ----------------------------------------------------------------------------- frame mapping
def intervals_to_segments(intervals, n_samples):
    """MFA phone intervals -> segments with frame boundaries (approved formula) and 3-state split."""
    T = 1 + n_samples // HOP_LENGTH
    bounds = [int(iv[0] * SAMPLE_RATE / HOP_LENGTH) for iv in intervals] + [T]
    bounds[0] = 0
    segs = []
    for k, (t0, t1, label) in enumerate(intervals):
        s, e = bounds[k], bounds[k + 1]
        n = e - s
        phone = SIL_PHONE if label in SILENCE_LABELS else strip_stress(label)
        # frames per state: count of i in [0,n) with (3*i)//n == state
        counts = [0] * N_STATES
        for i in range(n):
            counts[(N_STATES * i) // n] += 1
        segs.append({"phone": phone, "t0": t0, "t1": t1, "start": s, "end": e,
                     "n_frames": n, "state_frames": counts})
    return segs, T


def segments_to_frame_labels(segs, T):
    """-> (phone_id[T], state[T]) arrays. Every frame gets exactly one phone and one state."""
    phone_id = np.full(T, -1, dtype=np.int16)
    state = np.full(T, -1, dtype=np.int8)
    for sg in segs:
        n = sg["end"] - sg["start"]
        if n <= 0:
            continue
        idx = np.arange(n)
        phone_id[sg["start"]:sg["end"]] = PHONE_TO_ID[sg["phone"]]
        state[sg["start"]:sg["end"]] = (N_STATES * idx) // n
    return phone_id, state


# ----------------------------------------------------------------------------- check
def cmd_check(args):
    recs = load_g2p(n=args.n)
    tg_root = os.path.join(MFA_DIR, f"aligned_{args.name}")
    out_dir = os.path.join(ALIGN_DIR, args.name)
    os.makedirs(out_dir, exist_ok=True)

    failures, ok_rows = [], []
    stats = Counter()
    phone_dur = {}
    sil_internal_total = sil_internal_realised = mfa_extra_sil = 0
    sil_durs, short_phone_frames = [], Counter()

    def fail(uid, reason, detail=""):
        failures.append({"id": uid, "reason": reason, "detail": detail})

    for r in recs:
        uid = r["id"]
        tg = os.path.join(tg_root, SPEAKER, uid + ".TextGrid")
        if not os.path.exists(tg):
            tg = os.path.join(tg_root, uid + ".TextGrid")
        if not os.path.exists(tg):
            fail(uid, "no_textgrid")
            continue
        try:
            tiers = parse_textgrid(tg)
        except Exception as e:  # noqa: BLE001
            fail(uid, "textgrid_parse_error", str(e))
            continue
        if "phones" not in tiers or not tiers["phones"]:
            fail(uid, "no_phone_tier")
            continue
        ivs = tiers["phones"]
        n_samples, sr = wav_info(uid)
        dur = n_samples / sr
        if sr != SAMPLE_RATE:
            fail(uid, "unexpected_sample_rate", str(sr))
            continue

        # timestamp validity: ordered, contiguous, non-overlapping, positive, within WAV duration
        problem = None
        if abs(ivs[0][0]) > TIME_TOL:
            problem = f"first interval starts at {ivs[0][0]}"
        for (a0, a1, _), (b0, _b1, _) in zip(ivs, ivs[1:]):
            if a1 <= a0 or abs(a1 - b0) > 1e-6:
                problem = f"bad/non-contiguous interval near t={a1:.4f}"
                break
        if ivs[-1][1] <= ivs[-1][0]:
            problem = problem or "last interval has non-positive length"
        if ivs[-1][1] > dur + TIME_TOL:
            problem = problem or f"last interval ends {ivs[-1][1]:.4f} > wav duration {dur:.4f}"
        if abs(ivs[-1][1] - dur) > 0.05:
            problem = problem or f"last interval ends {ivs[-1][1]:.4f} != wav duration {dur:.4f}"
        if problem:
            fail(uid, "invalid_timestamps", problem)
            continue

        labels = [iv[2] for iv in ivs]
        if "spn" in labels:
            fail(uid, "spoken_noise_phone", "MFA emitted 'spn' (unknown/OOV region)")
            continue
        unknown = {l for l in labels if l not in SILENCE_LABELS and strip_stress(l) not in PHONE_TO_ID}
        if unknown:
            fail(uid, "unknown_phone_label", ",".join(sorted(unknown)))
            continue

        # phone sequence vs canonical G2P: stress-marked AND stress-free sequences must match exactly
        mfa_st = [l for l in labels if l not in SILENCE_LABELS]
        g2p_st = [p for p in r["phones_stress"] if p != SIL_PHONE]
        mfa_ph = [strip_stress(l) for l in mfa_st]
        g2p_ph = [p for p in r["phones"] if p != SIL_PHONE]
        if mfa_ph != g2p_ph or mfa_st != g2p_st:
            fail(uid, "phone_sequence_mismatch", f"mfa={len(mfa_ph)} g2p={len(g2p_ph)}")
            continue

        segs, T = intervals_to_segments(ivs, n_samples)
        # frame mapping validity
        if any(sg["end"] < sg["start"] for sg in segs):
            fail(uid, "non_monotone_frame_boundaries")
            continue
        pid, st = segments_to_frame_labels(segs, T)
        if (pid < 0).any() or (st < 0).any():
            fail(uid, "unassigned_frames", f"{int((pid < 0).sum())} frames")
            continue
        if sum(sg["n_frames"] for sg in segs) != T:
            fail(uid, "frame_count_mismatch")
            continue

        # silence-vs-punctuation comparison (internal SILs only; ends always exist)
        g_seq = r["phones"]
        word_gap_sil = 0
        internal_g = [i for i, p in enumerate(g_seq) if p == SIL_PHONE and 0 < i < len(g_seq) - 1]
        # map: number of non-sil phones preceding each internal G2P SIL
        g_marks = set()
        for i in internal_g:
            g_marks.add(sum(1 for p in g_seq[:i] if p != SIL_PHONE))
        m_marks = set()
        for k, l in enumerate(labels):
            if l in SILENCE_LABELS and 0 < k < len(labels) - 1:
                m_marks.add(sum(1 for x in labels[:k] if x not in SILENCE_LABELS))
                sil_durs.append(ivs[k][1] - ivs[k][0])
        sil_internal_total += len(g_marks)
        sil_internal_realised += len(g_marks & m_marks)
        mfa_extra_sil += len(m_marks - g_marks)

        for sg in segs:
            phone_dur.setdefault(sg["phone"], []).append(sg["n_frames"])
            if sg["n_frames"] < N_STATES:
                short_phone_frames[sg["n_frames"]] += 1
            stats["empty_states"] += sum(1 for c in sg["state_frames"] if c == 0)
        stats["phone_segments"] += len(segs)
        ok_rows.append({"id": uid, "n_frames": T, "duration": dur, "segments": segs})

    with open(os.path.join(out_dir, "alignments.jsonl"), "w", encoding="utf-8") as f:
        for row in ok_rows:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(out_dir, "failures.jsonl"), "w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row) + "\n")

    n = len(recs)
    print(f"\n=== Alignment check [{args.name}] ===")
    print(f"utterances: {n} | aligned OK: {len(ok_rows)} | failed: {len(failures)}")
    for reason, c in Counter(f_["reason"] for f_ in failures).most_common():
        ex = next(f_ for f_ in failures if f_["reason"] == reason)
        print(f"  failure {reason}: {c} (e.g. {ex['id']} {ex['detail']})")
    if ok_rows:
        print(f"phone segments: {stats['phone_segments']} | empty HMM states: {stats['empty_states']} "
              f"| phone segments with <3 frames: {dict(sorted(short_phone_frames.items()))}")
        print(f"G2P internal SIL (punctuation) realised as MFA silence: "
              f"{sil_internal_realised}/{sil_internal_total} "
              f"({100 * sil_internal_realised / max(sil_internal_total, 1):.1f}%) | "
              f"extra MFA internal silences: {mfa_extra_sil}")
        if sil_durs:
            d = np.array(sil_durs) * 1000
            print(f"internal silence duration ms: min {d.min():.0f} median {np.median(d):.0f} max {d.max():.0f}")
        allf = np.concatenate([np.array(v) for v in phone_dur.values()])
        print(f"phone duration frames: min {allf.min()} median {np.median(allf):.0f} max {allf.max()} "
              f"(1 frame = {1000 * HOP_LENGTH / SAMPLE_RATE:.2f} ms)")
        rows = sorted(((p, np.mean(v), len(v)) for p, v in phone_dur.items()), key=lambda x: -x[2])
        print("mean frames per phone (count): " +
              ", ".join(f"{p}:{m:.1f}({c})" for p, m, c in rows[:12]))
    print(f"reports: {out_dir}")
    return 0 if not failures else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--name", required=True)
    p.add_argument("--n", type=int, default=0)
    p.set_defaults(fn=cmd_prepare)
    c = sub.add_parser("check")
    c.add_argument("--name", required=True)
    c.add_argument("--n", type=int, default=0)
    c.set_defaults(fn=cmd_check)
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
