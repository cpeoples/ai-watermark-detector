#!/usr/bin/env python3
"""Attack / robustness battery for the watermark detectors.

The watermarking surveys (Liu et al. 2024; the MarkLLM toolkit's evaluation module) all
stress the same point: a watermark is only as good as its robustness to real-world edits.
MarkLLM ships a specific set of attacks -- WordDeletion, SynonymSubstitution, RandomWalkAttack,
Back-Translation, paraphrase -- and measures how detection degrades under each.

This tool reproduces the *token-level* core of that battery in a self-contained way (no paid
APIs, no paraphrase model required) and reports SURVIVAL CURVES: the detector's z-score (or
EXP-Edit p-value) as each attack gets stronger. It drives the SAME real detector the rest of
the repo validates, on streams you watermark yourself, so the numbers are authoritative.

Attacks implemented (all operate on the token stream, which is the substrate every text
watermark actually lives in):

  * substitution    - replace a fraction of tokens with random tokens (synonym-swap analogue)
  * deletion        - drop a fraction of tokens (WordDeletion)
  * insertion       - insert random tokens (dual of deletion; breaks fixed-window seeding)
  * random-walk     - repeated local single-token edits (the ICML 2024 random-walk attack idea)
  * block-shuffle   - permute contiguous blocks (reordering; paraphrase-structural analogue)

Interpretation
--------------
KGW/SynthID/Unigram/SWEET use windowed seeding, so insertions/deletions that shift the
context window degrade them faster than pure substitution. Unigram (global split) is the most
edit-robust of the greenlist family. EXP-Edit is explicitly edit-robust; use `--scheme exp-edit`
to see its p-value stay low under insert/delete where the others collapse.

Usage
-----
  python3 tools/attacks.py --scheme kgw --attacks substitution deletion random-walk --trials 100
  python3 tools/attacks.py --scheme unigram --attacks deletion insertion --trials 100 --json
"""

import argparse
import hashlib
import json
import os
import random
import sys

from _common import mean, run_detector as our_score, write_config


DETECTOR_DEFAULT = "target/release/ai-watermark-detector"


# --------------------------------------------------------------------------- attacks

def a_substitution(tokens, frac, vocab, rng):
    out = list(tokens)
    for i in range(len(out)):
        if rng.random() < frac:
            out[i] = rng.randrange(vocab)
    return out


def a_deletion(tokens, frac, vocab, rng):
    return [t for t in tokens if rng.random() >= frac] or [tokens[0]]


def a_insertion(tokens, frac, vocab, rng):
    out = []
    for t in tokens:
        out.append(t)
        if rng.random() < frac:
            out.append(rng.randrange(vocab))
    return out


def a_random_walk(tokens, frac, vocab, rng):
    # Repeated local single-token edits (substitute a random position), n = frac * len steps.
    out = list(tokens)
    steps = int(len(out) * frac)
    for _ in range(steps):
        pos = rng.randrange(len(out))
        out[pos] = rng.randrange(vocab)
    return out


def a_block_shuffle(tokens, frac, vocab, rng):
    # Split into blocks and shuffle a fraction of them (structural reordering).
    n = len(tokens)
    block = max(2, int(n * 0.1))
    blocks = [tokens[i:i + block] for i in range(0, n, block)]
    k = int(len(blocks) * frac)
    if k >= 2:
        idx = list(range(len(blocks)))
        chosen = rng.sample(idx, k)
        vals = [blocks[c] for c in chosen]
        rng.shuffle(vals)
        for c, v in zip(chosen, vals):
            blocks[c] = v
    return [t for b in blocks for t in b]


ATTACKS = {
    "substitution": a_substitution,
    "deletion": a_deletion,
    "insertion": a_insertion,
    "random-walk": a_random_walk,
    "block-shuffle": a_block_shuffle,
}


# --------------------------------------------------------------------------- builders

def build_kgw(args, seeding_scheme="lefthash", context_width=1):
    import torch
    from transformers import WatermarkLogitsProcessor

    vocab = args.vocab
    gamma = args.gamma
    lp = WatermarkLogitsProcessor(
        vocab_size=vocab, device="cpu", greenlist_ratio=gamma, bias=4.0,
        hashing_key=15485863, seeding_scheme=seeding_scheme, context_width=context_width,
    )

    def greens_for(prefix):
        return sorted(int(x) for x in lp._get_greenlist_ids(torch.tensor(prefix)).tolist())

    def gen_wm(length, seed):
        g = torch.Generator().manual_seed(seed)
        seq = [int(torch.randint(0, vocab, (1,), generator=g).item()) for _ in range(context_width)]
        while len(seq) < length:
            greens = greens_for(seq[-context_width:])
            seq.append(greens[int(torch.randint(0, len(greens), (1,), generator=g).item())])
        return seq

    def gen_ctrl(length, seed):
        g = torch.Generator().manual_seed(seed)
        return [int(x) for x in torch.randint(0, vocab, (length,), generator=g).tolist()]

    def build_oracle(streams):
        o = {}
        for ids in streams:
            for i in range(len(ids)):
                lo = max(0, i - context_width + 1)
                prefix = ids[lo:i + 1]
                if len(prefix) < context_width:
                    continue
                key = str(prefix[-1]) if context_width == 1 else " ".join(map(str, prefix))
                if key not in o:
                    o[key] = greens_for(prefix)
        return o

    return vocab, gamma, gen_wm, gen_ctrl, build_oracle


def score_kgw(args, streams, cfg_extra, seeding_scheme, context_width, oracle_cache=None):
    # streams: list of token id lists (already attacked). Returns list of z-scores.
    # oracle_cache: a mutable dict reused across calls so we never recompute a greenlist.
    vocab, gamma, _, _, build_oracle = build_kgw(args, seeding_scheme, context_width)
    if oracle_cache is None:
        oracle = build_oracle(streams)
    else:
        # Extend the shared cache with any prefixes not seen yet.
        import torch
        from transformers import WatermarkLogitsProcessor
        lp = WatermarkLogitsProcessor(
            vocab_size=vocab, device="cpu", greenlist_ratio=gamma, bias=4.0,
            hashing_key=15485863, seeding_scheme=seeding_scheme, context_width=context_width,
        )
        for ids in streams:
            for i in range(len(ids)):
                lo = max(0, i - context_width + 1)
                prefix = ids[lo:i + 1]
                if len(prefix) < context_width:
                    continue
                key = str(prefix[-1]) if context_width == 1 else " ".join(map(str, prefix))
                if key not in oracle_cache:
                    oracle_cache[key] = sorted(
                        int(x) for x in lp._get_greenlist_ids(torch.tensor(prefix)).tolist())
        oracle = oracle_cache
    cfg = {"ngram_len": 5, "keys": [654, 400, 836], "gamma": gamma, "vocab_size": vocab,
           "green_oracle": oracle, "seeding_scheme": seeding_scheme,
           "context_width": context_width}
    cfg.update(cfg_extra)
    path = write_config(cfg)
    zs = [our_score(args.detector, path, "kgw", s)["z"] for s in streams]
    os.unlink(path)
    return zs


def build_unigram(args):
    import numpy as np

    vocab = args.vocab
    gamma = args.gamma
    hash_key = 15485863

    def hash_fn(x):
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], "little")

    mask = np.array([True] * int(gamma * vocab) + [False] * (vocab - int(gamma * vocab)))
    rng = np.random.default_rng(hash_fn(hash_key))
    rng.shuffle(mask)
    greenlist = sorted(int(i) for i in np.nonzero(mask)[0])
    green_set = set(greenlist)

    def gen_wm(length, seed):
        r = random.Random(seed)
        out = []
        for _ in range(length):
            cands = [r.randrange(vocab) for _ in range(8)]
            out.append(next((c for c in cands if c in green_set), cands[0]))
        return out

    def gen_ctrl(length, seed):
        r = random.Random(seed)
        return [r.randrange(vocab) for _ in range(length)]

    cfg = {"ngram_len": 5, "keys": [654], "gamma": gamma, "vocab_size": vocab,
           "unigram_greenlist": greenlist}
    return gen_wm, gen_ctrl, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default=DETECTOR_DEFAULT)
    ap.add_argument("--scheme", default="kgw", choices=["kgw", "unigram"])
    ap.add_argument("--seeding", default="lefthash", choices=["lefthash", "selfhash"])
    ap.add_argument("--context-width", type=int, default=1)
    ap.add_argument("--attacks", nargs="+", default=list(ATTACKS.keys()),
                    choices=list(ATTACKS.keys()))
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--length", type=int, default=300)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        sys.exit('needs the reference implementations.\n  pip install -r tools/requirements.txt')
    if not os.path.exists(args.detector):
        sys.exit(f"detector not found: {args.detector} (cargo build --release)")

    # Generate the base watermarked streams once.
    if args.scheme == "unigram":
        gen_wm, gen_ctrl, cfg_base = build_unigram(args)
        wm_streams = [gen_wm(args.length, 1000 + i) for i in range(args.trials)]

        def score(streams):
            path = write_config({**cfg_base})
            zs = [our_score(args.detector, path, "unigram", s)["z"] for s in streams]
            os.unlink(path)
            return zs
    else:
        _, _, gen_wm, gen_ctrl, _ = build_kgw(args, args.seeding, args.context_width)
        wm_streams = [gen_wm(args.length, 1000 + i) for i in range(args.trials)]
        _oracle_cache = {}

        def score(streams):
            return score_kgw(args, streams, {}, args.seeding, args.context_width,
                             oracle_cache=_oracle_cache)

    baseline = mean(score(wm_streams))

    results = {"scheme": args.scheme, "seeding": f"{args.seeding}/cw{args.context_width}",
               "length": args.length, "trials": args.trials, "baseline_z": baseline,
               "attacks": {}}

    for attack in args.attacks:
        fn = ATTACKS[attack]
        curve = {}
        for frac in args.fracs:
            attacked = []
            for i, wm in enumerate(wm_streams):
                rng = random.Random(50_000 + i)
                attacked.append(fn(wm, frac, args.vocab, rng))
            curve[f"{frac:.2f}"] = mean(score(attacked))
        results["attacks"][attack] = curve

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print()
    print("=" * 72)
    print("WATERMARK ATTACK / ROBUSTNESS BATTERY")
    print("=" * 72)
    print(f"scheme: {args.scheme}  {results['seeding']}   length: {args.length}   "
          f"trials: {args.trials}")
    print(f"baseline mean z (no attack): {baseline:.2f}")
    print()
    header = "attack".ljust(16) + "".join(f"{f:>8.2f}" for f in args.fracs)
    print(header)
    print("-" * len(header))
    for attack, curve in results["attacks"].items():
        row = attack.ljust(16) + "".join(f"{curve[f'{f:.2f}']:>8.1f}" for f in args.fracs)
        print(row)
    print("-" * len(header))
    print("Cells are mean detector z-score at each attack fraction (higher = watermark still")
    print("detectable). A z that falls toward ~0 means the attack has erased the signal. This")
    print("is measured on YOUR watermarked streams with the real detector -- authoritative,")
    print("and makes no claim about production vendor text.")


if __name__ == "__main__":
    main()
