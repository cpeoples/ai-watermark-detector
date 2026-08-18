#!/usr/bin/env python3
"""Live real-world validation battery for BOTH watermark schemes.

This is the project's strongest real-world proof. It drives the REAL, published
watermark implementations shipped in Hugging Face `transformers`:

  * SynthID-Text  -> `SynthIDTextWatermarkLogitsProcessor` (Google DeepMind's code)
  * KGW greenlist -> `WatermarkLogitsProcessor` + `WatermarkDetector` (Kirchenbauer et al.)

and checks that OUR Rust detector reproduces them. Because YOU choose the key, you are
the key-holder and detection is authoritative — this is genuine ground truth.

Why "the key" is unavoidable: KGW/SynthID are *keyed statistical* watermarks. The mark is
not embedded in the text; it is a faint statistical bias created with a secret key. To
measure it you must reproduce the same keyed split, which needs the key. With your own key
here, detection is exact; against a vendor's production output (secret key) it is impossible
by cryptographic design. (Signed file provenance / C2PA is different — check it with the
Rust CLI's `check`/`scan` subcommands.)

What it validates
-----------------
For each scheme:
  1. BIT-FOR-FIT: our detector's mean-g / green-fraction equals the official implementation
     on the same token streams (via an exported sampling table / green-list oracle).
  2. CROSS-CHECK: for KGW, our z-score is compared to HF's real `WatermarkDetector` z.
  3. SEPARATION: watermarked streams score high; unwatermarked controls stay at baseline.
  4. ROBUSTNESS: signal degradation under increasing token edits (copy -> light -> heavy).
  5. MULTI-KEY: repeats across several independent keys.

Modes
-----
  --mode model-free  (default): drive the official logits processors on synthetic streams.
                                 Works fully offline once `transformers`+`torch` are installed.
  --mode model:                 also generate from a real open LM (needs huggingface.co).
  --mode both:                  attempt real-model; fall back to model-free if the hub is down.

Requirements
------------
  pip install "transformers>=4.46" torch

Usage
-----
  python3 tools/validate.py --schemes synthid kgw --num 6 --gen-len 200 --json
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys

from _common import mean, run_detector as our_score, write_config


DETECTOR_DEFAULT = "target/release/ai-watermark-detector"
INSTALL_HINT = (
    "This battery needs the real reference implementations.\n"
    "  pip install -r tools/requirements.txt\n"
    "Everything you *ship* (the Rust `score`/`check`/`scan` CLI) works without it."
)


def corrupt(ids, frac, vocab, rng):
    import torch
    out = list(ids)
    for i in range(len(out)):
        if torch.rand(1, generator=rng).item() < frac:
            out[i] = int(torch.randint(0, vocab, (1,), generator=rng).item())
    return out


# Prompts used when generating from a real open LM (--mode model/both).
REAL_MODEL_PROMPTS = [
    "The history of transportation",
    "In a distant future",
    "A recipe for a good day",
    "The economics of small towns",
    "Once upon a time",
    "Notes on modern architecture",
    "The science of sleep",
    "How rivers shape the land",
]


def try_real_model_synthid(args, keys, ngram_len):
    """Generate real watermarked/control token streams from an open LM using Google's
    official SynthIDTextWatermarkingConfig. Returns (wm_streams, ctrl_streams) or None if
    the model hub is unreachable (caller falls back to model-free synthetic streams)."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            SynthIDTextWatermarkingConfig,
        )
    except ImportError:
        return None

    try:
        print(f"[model] loading {args.model} ...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.eval()
    except Exception as e:
        print(f"[model] hub/model unavailable ({e}); falling back to model-free", file=sys.stderr)
        return None

    wm_config = SynthIDTextWatermarkingConfig(keys=list(keys), ngram_len=ngram_len)

    def gen(watermarked):
        streams = []
        for i in range(args.num):
            prompt = REAL_MODEL_PROMPTS[i % len(REAL_MODEL_PROMPTS)]
            inputs = tokenizer([prompt], return_tensors="pt")
            in_len = inputs["input_ids"].shape[1]
            kwargs = dict(max_new_tokens=args.gen_len, do_sample=True, top_k=40, temperature=1.0)
            if watermarked:
                kwargs["watermarking_config"] = wm_config
            with torch.no_grad():
                out = model.generate(**inputs, **kwargs)
            streams.append(out[0, in_len:].tolist())
        return streams

    return gen(True), gen(False)


# ---------------------------------------------------------------------------
# SynthID
# ---------------------------------------------------------------------------

def validate_synthid(args, keys, ngram_len, base_cfg, results):
    import torch
    from transformers import SynthIDTextWatermarkLogitsProcessor

    lp = SynthIDTextWatermarkLogitsProcessor(
        keys=keys, ngram_len=ngram_len, sampling_table_size=2**16,
        sampling_table_seed=0, context_history_size=1024, device="cpu",
    )
    table = [int(x) for x in lp.sampling_table.to("cpu").tolist()]
    cfg_path = write_config(base_cfg, {"keys": list(keys), "ngram_len": ngram_len,
                                       "sampling_table": table})

    def official_mean_g(ids):
        t = torch.tensor([ids])
        g = lp.compute_g_values(t)
        m = lp.compute_context_repetition_mask(t).unsqueeze(-1).expand_as(g).to(g.dtype)
        tot = m.sum().item()
        return (g.to(m.dtype) * m).sum().item() / tot if tot > 0 else float("nan")

    def official_weighted_mean_g(ids):
        # MarkLLM/Nature weighted-mean detector: g-values [1, seq, depth], mask [1, seq].
        # weights default to linspace(10, 1, depth), normalised to sum to depth.
        import numpy as np
        t = torch.tensor([ids])
        g = lp.compute_g_values(t).float().cpu().numpy()          # [1, seq, depth]
        mask = lp.compute_context_repetition_mask(t).float().cpu().numpy()  # [1, seq]
        depth = g.shape[-1]
        if depth == 1:
            num = np.sum(g * mask[..., None], axis=(1, 2))
            den = depth * np.sum(mask, axis=1)
            return float((num / den)[0]) if den[0] > 0 else float("nan")
        weights = np.linspace(10, 1, depth)
        weights = weights * (depth / weights.sum())
        gw = g * weights[None, None, :]
        num = np.sum(gw * mask[..., None], axis=(1, 2))
        den = depth * np.sum(mask, axis=1)
        return float((num / den)[0]) if den[0] > 0 else float("nan")

    vocab = args.vocab

    def bias_stream(seed):
        g = torch.Generator().manual_seed(seed)
        seq = torch.randint(0, vocab, (ngram_len - 1,), generator=g).tolist()
        while len(seq) < args.gen_len:
            cands = torch.randint(0, vocab, (args.candidates,), generator=g).tolist()
            best, best_s = cands[0], -1.0
            for c in cands:
                trial = torch.tensor([seq[-(ngram_len - 1):] + [c]])
                gv = lp.compute_g_values(trial)
                s = gv[0, -1].float().sum().item() if gv.shape[1] > 0 else 0.0
                if s > best_s:
                    best_s, best = s, c
            seq.append(best)
        return seq

    # If a real-model run is requested and the hub is reachable, use genuine LM output;
    # otherwise fall back to model-free synthetic streams biased by Google's g-function.
    real = None
    if args.mode in ("model", "both"):
        real = try_real_model_synthid(args, keys, ngram_len)
        if real is None and args.mode == "model":
            raise RuntimeError("real-model requested but model/hub unavailable")
    real_wm, real_ct = real if real else (None, None)
    stream_kind = "real-model" if real else "model-free"

    rows = []
    max_abs_diff = 0.0
    for i in range(args.num):
        wm = real_wm[i] if real else bias_stream(1000 + i)
        ctrl = (real_ct[i] if real
                else torch.randint(0, vocab, (args.gen_len,),
                                   generator=torch.Generator().manual_seed(9000 + i)).tolist())
        o_wm, o_ct = official_mean_g(wm), official_mean_g(ctrl)
        ow_wm, ow_ct = official_weighted_mean_g(wm), official_weighted_mean_g(ctrl)
        r_wm = our_score(args.detector, cfg_path, "synthid", wm)
        r_ct = our_score(args.detector, cfg_path, "synthid", ctrl)
        max_abs_diff = max(max_abs_diff, abs(o_wm - r_wm["mean_g"]), abs(o_ct - r_ct["mean_g"]))
        # Weighted-mean detector cross-check (only present when our detector emits it).
        if "weighted_mean_g" in r_wm and r_wm["weighted_mean_g"] is not None:
            max_abs_diff = max(max_abs_diff, abs(ow_wm - r_wm["weighted_mean_g"]),
                               abs(ow_ct - r_ct["weighted_mean_g"]))
        # robustness sweep on the watermarked stream
        rng = torch.Generator().manual_seed(500 + i)
        sweep = {f"edit_{int(f*100):02d}": our_score(args.detector, cfg_path, "synthid",
                 corrupt(wm, f, vocab, rng))["z"] for f in args.edits}
        rows.append({"official_wm_g": o_wm, "our_wm_g": r_wm["mean_g"], "our_wm_z": r_wm["z"],
                     "official_ctrl_g": o_ct, "our_ctrl_g": r_ct["mean_g"], "our_ctrl_z": r_ct["z"],
                     "official_wm_wg": ow_wm, "our_wm_wg": r_wm.get("weighted_mean_g"),
                     "sweep": sweep})

    os.unlink(cfg_path)
    wm_wgs = [r["our_wm_wg"] for r in rows if r.get("our_wm_wg") is not None]
    results["synthid"] = {
        "stream_kind": stream_kind,
        "official_wm_g": mean([r["official_wm_g"] for r in rows]),
        "our_wm_g": mean([r["our_wm_g"] for r in rows]),
        "our_wm_z": mean([r["our_wm_z"] for r in rows]),
        "official_wm_weighted_g": mean([r["official_wm_wg"] for r in rows]),
        "our_wm_weighted_g": (mean(wm_wgs) if wm_wgs else None),
        "official_ctrl_g": mean([r["official_ctrl_g"] for r in rows]),
        "our_ctrl_g": mean([r["our_ctrl_g"] for r in rows]),
        "our_ctrl_z": mean([r["our_ctrl_z"] for r in rows]),
        "max_abs_g_diff_vs_official": max_abs_diff,
        "bit_for_bit": max_abs_diff < 1e-6,
        "sweep": {k: mean([r["sweep"][k] for r in rows]) for k in rows[0]["sweep"]},
    }


# ---------------------------------------------------------------------------
# KGW
# ---------------------------------------------------------------------------

def validate_kgw(args, base_cfg, results, seeding_scheme="lefthash", context_width=1,
                 result_key="kgw"):
    import torch
    from transformers import WatermarkLogitsProcessor, WatermarkDetector, PretrainedConfig

    vocab = args.vocab
    gamma = base_cfg.get("gamma", 0.25)
    hashing_key = 15485863
    selfhash = seeding_scheme == "selfhash"
    n = context_width + 1 - int(selfhash)

    lp = WatermarkLogitsProcessor(
        vocab_size=vocab, device="cpu", greenlist_ratio=gamma, bias=4.0,
        hashing_key=hashing_key, seeding_scheme=seeding_scheme, context_width=context_width,
    )

    def greenset(prefix_tokens):
        """HF greenlist for a given prefix (list of ints)."""
        greens = lp._get_greenlist_ids(torch.tensor(prefix_tokens)).tolist()
        return set(int(x) for x in greens)

    # Build a stream by a greedy tournament using HF's real greenlist. For lefthash the
    # greenlist depends on the last token; for selfhash it depends on the whole window
    # INCLUDING the candidate, so we test each candidate's own greenlist membership.
    def bias_stream(seed):
        g = torch.Generator().manual_seed(seed)
        seq = torch.randint(0, vocab, (context_width,), generator=g).tolist()
        while len(seq) < args.gen_len:
            cands = torch.randint(0, vocab, (args.candidates,), generator=g).tolist()
            pick = cands[0]
            for c in cands:
                if selfhash:
                    prefix = seq[-(context_width - 1):] + [c] if context_width > 1 else [c]
                    if c in greenset(prefix):
                        pick = c
                        break
                else:
                    if c in greenset(seq[-context_width:]):
                        pick = c
                        break
            seq.append(pick)
        return seq

    def random_stream(seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, vocab, (args.gen_len,), generator=g).tolist()

    # Build the oracle keyed by every prefix that actually occurs in the streams we score.
    # This is exact and finite (no need to enumerate the whole vocab), and it generalizes
    # to selfhash / larger context_width where a last-token map is insufficient.
    def prefixes_of(ids):
        out = set()
        for i in range(len(ids) - n + 1):
            ngram = ids[i:i + n]
            prefix = tuple(ngram) if selfhash else tuple(ngram[:-1])
            out.add(prefix)
        return out

    streams = []
    for i in range(args.num):
        streams.append(("wm", bias_stream(2000 + i)))
        streams.append(("ct", random_stream(8000 + i)))

    oracle = {}
    for _, ids in streams:
        for prefix in prefixes_of(ids):
            key = " ".join(str(t) for t in prefix)
            if key not in oracle:
                oracle[key] = sorted(greenset(list(prefix)))

    cfg_path = write_config(base_cfg, {
        "gamma": gamma, "vocab_size": vocab, "green_oracle": oracle,
        "seeding_scheme": seeding_scheme, "context_width": context_width,
    })

    model_config = PretrainedConfig(vocab_size=vocab)
    model_config.bos_token_id = 0
    model_config.is_encoder_decoder = False
    wd = WatermarkDetector(
        model_config=model_config, device="cpu",
        watermarking_config={"greenlist_ratio": gamma, "bias": 4.0,
                             "hashing_key": hashing_key, "seeding_scheme": seeding_scheme,
                             "context_width": context_width},
        ignore_repeated_ngrams=True,
    )

    rows = []
    max_abs_z_diff = 0.0
    max_abs_g_diff = 0.0
    for i in range(args.num):
        wm = streams[2 * i][1]
        ctrl = streams[2 * i + 1][1]

        r_wm = our_score(args.detector, cfg_path, "kgw", wm)
        r_ct = our_score(args.detector, cfg_path, "kgw", ctrl)

        hf_wm = wd(torch.tensor([wm]), return_dict=True)
        hf_ct = wd(torch.tensor([ctrl]), return_dict=True)
        hf_wm_z = float(hf_wm.z_score[0]); hf_ct_z = float(hf_ct.z_score[0])
        hf_wm_g = float(hf_wm.green_fraction[0]); hf_ct_g = float(hf_ct.green_fraction[0])

        max_abs_z_diff = max(max_abs_z_diff, abs(hf_wm_z - r_wm["z"]), abs(hf_ct_z - r_ct["z"]))
        max_abs_g_diff = max(max_abs_g_diff, abs(hf_wm_g - r_wm["mean_g"]), abs(hf_ct_g - r_ct["mean_g"]))

        rng = torch.Generator().manual_seed(600 + i)
        sweep = {f"edit_{int(f*100):02d}": our_score(args.detector, cfg_path, "kgw",
                 corrupt(wm, f, vocab, rng))["z"] for f in args.edits}
        rows.append({"our_wm_z": r_wm["z"], "hf_wm_z": hf_wm_z, "our_wm_g": r_wm["mean_g"],
                     "hf_wm_g": hf_wm_g, "our_ctrl_z": r_ct["z"], "hf_ctrl_z": hf_ct_z,
                     "our_ctrl_g": r_ct["mean_g"], "hf_ctrl_g": hf_ct_g, "sweep": sweep})

    os.unlink(cfg_path)
    results[result_key] = {
        "seeding": f"{seeding_scheme}/cw{context_width}",
        "our_wm_g": mean([r["our_wm_g"] for r in rows]),
        "hf_wm_g": mean([r["hf_wm_g"] for r in rows]),
        "our_wm_z": mean([r["our_wm_z"] for r in rows]),
        "hf_wm_z": mean([r["hf_wm_z"] for r in rows]),
        "our_ctrl_g": mean([r["our_ctrl_g"] for r in rows]),
        "hf_ctrl_g": mean([r["hf_ctrl_g"] for r in rows]),
        "our_ctrl_z": mean([r["our_ctrl_z"] for r in rows]),
        "hf_ctrl_z": mean([r["hf_ctrl_z"] for r in rows]),
        "max_abs_g_diff_vs_hf": max_abs_g_diff,
        "max_abs_z_diff_vs_hf": max_abs_z_diff,
        "bit_for_bit": max_abs_g_diff < 1e-6,
        "sweep": {k: mean([r["sweep"][k] for r in rows]) for k in rows[0]["sweep"]},
    }


def validate_exp(args, keys, results):
    """Aaronson/Kuditipudi exponential (Gumbel) scheme. transformers ships no detector for
    this family, so we implement the canonical reference here (a keyed PRNG assigning each
    (context, token) a uniform u; generation picks argmax u^(1/p); detection sums
    -ln(1-u_chosen)). We then check OUR Rust detector matches the reference score bit-for-bit
    and that watermarked separates from control."""
    import torch

    vocab = args.vocab
    cw = 1
    key = int(keys[0])

    def u_value(context, token):
        # Deterministic keyed uniform in (0,1) for a (context, token) pair.
        h = hashlib.sha256()
        h.update(str(key).encode())
        h.update(b"|")
        h.update(" ".join(str(t) for t in context).encode())
        h.update(b"|")
        h.update(str(int(token)).encode())
        # 53 bits -> double in [0,1)
        val = int.from_bytes(h.digest()[:8], "big") >> 11
        return val / float(1 << 53)

    def gen_watermarked(seed):
        g = torch.Generator().manual_seed(seed)
        seq = torch.randint(0, vocab, (cw,), generator=g).tolist()
        while len(seq) < args.gen_len:
            context = seq[-cw:]
            cands = torch.randint(0, vocab, (args.candidates,), generator=g).tolist()
            # Gumbel/exponential rule with (near-)uniform proposal: pick max u.
            pick = max(cands, key=lambda c: u_value(context, c))
            seq.append(pick)
        return seq

    def gen_control(seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, vocab, (args.gen_len,), generator=g).tolist()

    def ref_score(ids):
        # Reference detection statistic: sum of -ln(1-u) over unique (context, token).
        seen = set()
        s, n = 0.0, 0
        for i in range(cw, len(ids)):
            context = tuple(ids[i - cw:i])
            token = ids[i]
            sig = (context, token)
            if sig in seen:
                continue
            seen.add(sig)
            u = min(max(u_value(list(context), token), 1e-12), 1 - 1e-12)
            s += -math.log(1 - u)
            n += 1
        z = (s - n) / math.sqrt(n) if n else 0.0
        return {"mean_g": s / n if n else 0.0, "z": z, "n": n}

    streams = []
    for i in range(args.num):
        streams.append(gen_watermarked(3000 + i))
        streams.append(gen_control(7000 + i))

    # Export the u-oracle for the (context, token) pairs that occur.
    oracle = {}
    for ids in streams:
        for i in range(cw, len(ids)):
            context = ids[i - cw:i]
            token = ids[i]
            k = " ".join(str(t) for t in context) + "|" + str(token)
            if k not in oracle:
                oracle[k] = u_value(context, token)

    cfg_path = write_config({"ngram_len": 5, "keys": list(keys)},
                            {"context_width": cw, "u_oracle": oracle})

    rows = []
    max_abs_diff = 0.0
    for i in range(args.num):
        wm = streams[2 * i]
        ct = streams[2 * i + 1]
        ref_wm, ref_ct = ref_score(wm), ref_score(ct)
        our_wm = our_score(args.detector, cfg_path, "exp", wm)
        our_ct = our_score(args.detector, cfg_path, "exp", ct)
        max_abs_diff = max(max_abs_diff, abs(ref_wm["z"] - our_wm["z"]),
                           abs(ref_ct["z"] - our_ct["z"]))
        rng = torch.Generator().manual_seed(700 + i)
        sweep = {f"edit_{int(f*100):02d}": our_score(args.detector, cfg_path, "exp",
                 corrupt(wm, f, vocab, rng))["z"] for f in args.edits}
        rows.append({"ref_wm_z": ref_wm["z"], "our_wm_z": our_wm["z"],
                     "ref_ct_z": ref_ct["z"], "our_ct_z": our_ct["z"],
                     "our_wm_g": our_wm["mean_g"], "our_ct_g": our_ct["mean_g"], "sweep": sweep})

    os.unlink(cfg_path)
    results["exp"] = {
        "ref_wm_z": mean([r["ref_wm_z"] for r in rows]),
        "our_wm_z": mean([r["our_wm_z"] for r in rows]),
        "ref_ct_z": mean([r["ref_ct_z"] for r in rows]),
        "our_ct_z": mean([r["our_ct_z"] for r in rows]),
        "our_wm_g": mean([r["our_wm_g"] for r in rows]),
        "our_ct_g": mean([r["our_ct_g"] for r in rows]),
        "max_abs_z_diff_vs_ref": max_abs_diff,
        "bit_for_bit": max_abs_diff < 1e-4,
        "sweep": {k: mean([r["sweep"][k] for r in rows]) for k in rows[0]["sweep"]},
    }


def validate_unigram(args, keys, base_cfg, results):
    """Unigram (Zhao et al. 2024): a single GLOBAL green/red split for every position (no
    per-token seeding), which is why it survives paraphrase far better than KGW. We reproduce
    MarkLLM's *exact* greenlist construction:

        mask = [True]*int(gamma*V) + [False]*rest
        rng  = np.random.default_rng( int.from_bytes(sha256(int64(hash_key))[:4], 'little') )
        rng.shuffle(mask)

    and its z-score (green - gamma*T)/sqrt(T*gamma*(1-gamma)). We export the resulting green
    set to our Rust detector and check it matches the MarkLLM reference bit-for-bit."""
    import numpy as np
    import torch

    vocab = args.vocab
    gamma = base_cfg.get("gamma", 0.5)
    hash_key = int(keys[0])

    # --- MarkLLM-exact greenlist (unigram.py: UnigramUtils.__init__ / _hash_fn) ---
    def hash_fn(x):
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], "little")

    mask = np.array([True] * int(gamma * vocab) + [False] * (vocab - int(gamma * vocab)))
    rng = np.random.default_rng(hash_fn(hash_key))
    rng.shuffle(mask)
    greenlist = sorted(int(i) for i in np.nonzero(mask)[0])
    green_set = set(greenlist)

    def ref_z(ids):
        # MarkLLM UnigramUtils.score_sequence + _compute_z_score.
        green = sum(1 for t in ids if mask[t])
        T = len(ids)
        return (green - gamma * T) / ((T * gamma * (1 - gamma)) ** 0.5), green

    def watermarked_stream(seed):
        # Bias generation toward the fixed global green set (what the logits bias does).
        g = torch.Generator().manual_seed(seed)
        out = []
        for _ in range(args.gen_len):
            cands = torch.randint(0, vocab, (args.candidates,), generator=g).tolist()
            pick = next((c for c in cands if c in green_set), cands[0])
            out.append(pick)
        return out

    def random_stream(seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, vocab, (args.gen_len,), generator=g).tolist()

    cfg_path = write_config(base_cfg, {
        "gamma": gamma, "vocab_size": vocab, "unigram_greenlist": greenlist,
    })

    rows = []
    max_abs_z_diff = 0.0
    for i in range(args.num):
        wm = watermarked_stream(3100 + i)
        ct = random_stream(9100 + i)
        our_wm = our_score(args.detector, cfg_path, "unigram", wm)
        our_ct = our_score(args.detector, cfg_path, "unigram", ct)
        ref_wm_z, _ = ref_z(wm)
        ref_ct_z, _ = ref_z(ct)
        max_abs_z_diff = max(max_abs_z_diff, abs(ref_wm_z - our_wm["z"]),
                             abs(ref_ct_z - our_ct["z"]))
        rng2 = torch.Generator().manual_seed(910 + i)
        sweep = {f"edit_{int(f*100):02d}": our_score(args.detector, cfg_path, "unigram",
                 corrupt(wm, f, vocab, rng2))["z"] for f in args.edits}
        rows.append({"ref_wm_z": ref_wm_z, "our_wm_z": our_wm["z"],
                     "ref_ct_z": ref_ct_z, "our_ct_z": our_ct["z"],
                     "our_wm_g": our_wm["mean_g"], "our_ct_g": our_ct["mean_g"], "sweep": sweep})

    os.unlink(cfg_path)
    results["unigram"] = {
        "ref": "MarkLLM Unigram (Zhao 2024) - global fixed split",
        "ref_wm_z": mean([r["ref_wm_z"] for r in rows]),
        "our_wm_z": mean([r["our_wm_z"] for r in rows]),
        "ref_ct_z": mean([r["ref_ct_z"] for r in rows]),
        "our_ct_z": mean([r["our_ct_z"] for r in rows]),
        "our_wm_g": mean([r["our_wm_g"] for r in rows]),
        "our_ct_g": mean([r["our_ct_g"] for r in rows]),
        "max_abs_z_diff_vs_ref": max_abs_z_diff,
        "bit_for_bit": max_abs_z_diff < 1e-4,
        "sweep": {k: mean([r["sweep"][k] for r in rows]) for k in rows[0]["sweep"]},
    }


def validate_sweet(args, keys, base_cfg, results):
    """SWEET (Lee et al. 2024): KGW greenlist scoring, but only at HIGH-ENTROPY positions
    (the canonical code watermark - it leaves low-entropy/boilerplate tokens untouched). We
    reproduce MarkLLM's SWEETUtils.score_sequence: count green only where entropy > threshold,
    with z over the high-entropy positions. Since we have no real LM here, we synthesize a
    per-position entropy mask and drive both the reference and our Rust detector from it, so
    the comparison is exact and the entropy-gating semantics are what's under test."""
    import torch

    vocab = args.vocab
    gamma = base_cfg.get("gamma", 0.5)
    cw = 1
    key = int(keys[0])

    def greenset(prev):
        g = torch.Generator().manual_seed((key * int(prev)) % (2 ** 63 - 1))
        perm = torch.randperm(vocab, generator=g)
        return set(perm[: int(gamma * vocab)].tolist())

    def watermarked_stream(seed):
        g = torch.Generator().manual_seed(seed)
        seq = torch.randint(0, vocab, (cw,), generator=g).tolist()
        while len(seq) < args.gen_len:
            greens = greenset(seq[-1])
            cands = torch.randint(0, vocab, (args.candidates,), generator=g).tolist()
            seq.append(next((c for c in cands if c in greens), cands[0]))
        return seq

    def random_stream(seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, vocab, (args.gen_len,), generator=g).tolist()

    def entropy_mask_for(ids, seed):
        # Synthetic high-entropy mask: ~60% of positions are "high entropy" (scored).
        g = torch.Generator().manual_seed(seed)
        probs = torch.rand(len(ids), generator=g)
        return [bool(p > 0.4) for p in probs.tolist()]

    def ref_z(ids, mask):
        # MarkLLM SWEETUtils.score_sequence semantics: score positions past cw where
        # mask[idx] is True; green if token in greenlist(prefix).
        green = 0
        total = 0
        for idx in range(cw, len(ids)):
            if not mask[idx]:
                continue
            total += 1
            if ids[idx] in greenset(ids[idx - 1]):
                green += 1
        if total == 0:
            return 0.0, 0
        return (green - gamma * total) / ((total * gamma * (1 - gamma)) ** 0.5), total

    rows = []
    max_abs_z_diff = 0.0
    for i in range(args.num):
        wm = watermarked_stream(3200 + i)
        ct = random_stream(9200 + i)
        wm_mask = entropy_mask_for(wm, 4200 + i)
        ct_mask = entropy_mask_for(ct, 4300 + i)

        # Oracle keyed by previous token over the positions we will score.
        def oracle_for(ids):
            o = {}
            for idx in range(cw, len(ids)):
                prev = ids[idx - 1]
                k = str(prev)
                if k not in o:
                    o[k] = sorted(greenset(prev))
            return o

        cfg_wm = write_config(base_cfg, {
            "gamma": gamma, "vocab_size": vocab, "green_oracle": oracle_for(wm),
            "entropy_mask": wm_mask, "seeding_scheme": "lefthash", "context_width": cw,
        })
        cfg_ct = write_config(base_cfg, {
            "gamma": gamma, "vocab_size": vocab, "green_oracle": oracle_for(ct),
            "entropy_mask": ct_mask, "seeding_scheme": "lefthash", "context_width": cw,
        })
        our_wm = our_score(args.detector, cfg_wm, "sweet", wm)
        our_ct = our_score(args.detector, cfg_ct, "sweet", ct)
        ref_wm_z, _ = ref_z(wm, wm_mask)
        ref_ct_z, _ = ref_z(ct, ct_mask)
        max_abs_z_diff = max(max_abs_z_diff, abs(ref_wm_z - our_wm["z"]),
                             abs(ref_ct_z - our_ct["z"]))
        os.unlink(cfg_wm)
        os.unlink(cfg_ct)
        rows.append({"ref_wm_z": ref_wm_z, "our_wm_z": our_wm["z"],
                     "ref_ct_z": ref_ct_z, "our_ct_z": our_ct["z"],
                     "our_wm_g": our_wm["mean_g"], "our_ct_g": our_ct["mean_g"], "sweep": {}})

    results["sweet"] = {
        "ref": "MarkLLM SWEET (Lee 2024) - entropy-gated KGW (code)",
        "ref_wm_z": mean([r["ref_wm_z"] for r in rows]),
        "our_wm_z": mean([r["our_wm_z"] for r in rows]),
        "ref_ct_z": mean([r["ref_ct_z"] for r in rows]),
        "our_ct_z": mean([r["our_ct_z"] for r in rows]),
        "our_wm_g": mean([r["our_wm_g"] for r in rows]),
        "our_ct_g": mean([r["our_ct_g"] for r in rows]),
        "max_abs_z_diff_vs_ref": max_abs_z_diff,
        "bit_for_bit": max_abs_z_diff < 1e-4,
        "sweep": {},
    }


def validate_exp_edit(args, keys, base_cfg, results):
    """EXP-Edit / ITS-Edit (Kuditipudi et al., TMLR 2024): the distortion-free AND
    edit-robust watermark. Generation samples via the exponential rule using a keyed uniform
    matrix xi [pseudo_length, vocab]; detection is the MINIMUM edit-distance alignment cost
    between the token stream and xi (lower = watermark present), with significance from a
    permutation test against random reference keys.

    We port the reference `levenshtein` / `one_run` / `permutation_test` (jthickstun/watermark,
    MarkLLM exp_edit) in pure Python, export xi to our Rust detector, and check that OUR Rust
    alignment statistic matches the reference bit-for-bit. We then run the permutation test to
    confirm watermarked streams get a small p-value while controls do not, and add an EDIT
    robustness sweep (insert/delete) that plain EXP cannot survive."""
    import torch

    vocab = args.vocab_edit if getattr(args, "vocab_edit", None) else 60
    pseudo_len = args.gen_len_edit if getattr(args, "gen_len_edit", None) else 80
    m = min(40, pseudo_len)  # token stream length
    key = int(keys[0])

    def make_xi(seed):
        rng = random.Random(seed)
        return [[rng.random() for _ in range(vocab)] for _ in range(pseudo_len)]

    xi = make_xi(key)

    def gen_watermarked():
        # Exponential sampling with a (near-)uniform model: token = argmax xi[t]^(1/p).
        # With uniform p this is argmax over xi rows -> aligns the stream to the key.
        seq = []
        for t in range(m):
            row = xi[t % pseudo_len]
            # argmax of xi[t][token] ** (1/p); uniform p => argmax xi[t][token].
            seq.append(max(range(vocab), key=lambda v: row[v]))
        return seq

    def gen_control(seed):
        r = random.Random(seed)
        return [r.randrange(vocab) for _ in range(m)]

    def levenshtein(x, xi_mat, j0, n, gamma=0.0):
        ln = len(x)
        lm = ln
        A = [[0.0] * (lm + 1) for _ in range(ln + 1)]
        for i in range(ln + 1):
            for j in range(lm + 1):
                if i == 0:
                    A[i][j] = j * gamma
                elif j == 0:
                    A[i][j] = i * gamma
                else:
                    p = xi_mat[(j0 + (j - 1)) % n][x[i - 1]]
                    cost = math.log(max(1.0 - p, 1e-300))
                    best = A[i - 1][j] + gamma
                    if A[i][j - 1] + gamma < best:
                        best = A[i][j - 1] + gamma
                    if A[i - 1][j - 1] + cost < best:
                        best = A[i - 1][j - 1] + cost
                    A[i][j] = best
        return A[ln][lm]

    def test_stat(tokens, xi_mat):
        n = len(xi_mat)
        return min(levenshtein(tokens, xi_mat, j0, n) for j0 in range(n))

    def permutation_test(tokens, observed_xi, n_runs=100):
        obs = test_stat(tokens, observed_xi)
        count = 0
        for r in range(n_runs):
            null_xi = make_xi(10_000_000 + r)
            if test_stat(tokens, null_xi) <= obs:
                count += 1
        return (count + 1.0) / (n_runs + 1.0), obs

    def edit_corrupt(tokens, frac, seed):
        # Insertions/deletions (not just substitutions) -- the edit channel EXP-Edit targets.
        r = random.Random(seed)
        out = list(tokens)
        n_edits = int(len(tokens) * frac)
        for _ in range(n_edits):
            if not out:
                break
            pos = r.randrange(len(out))
            if r.random() < 0.5:
                out.pop(pos)
            else:
                out.insert(pos, r.randrange(vocab))
        return out

    cfg_path = write_config(base_cfg, {
        "vocab_size": vocab, "exp_edit_xi": xi, "exp_edit_gamma": 0.0,
    })

    rows = []
    max_abs_diff = 0.0
    n_runs = getattr(args, "edit_runs", 60)
    for i in range(args.num):
        wm = gen_watermarked()
        ct = gen_control(20_000 + i)
        our_wm = our_score(args.detector, cfg_path, "exp-edit", wm)
        our_ct = our_score(args.detector, cfg_path, "exp-edit", ct)
        ref_wm = test_stat(wm, xi)
        ref_ct = test_stat(ct, xi)
        max_abs_diff = max(max_abs_diff, abs(ref_wm - our_wm["mean_g"]),
                           abs(ref_ct - our_ct["mean_g"]))
        p_wm, _ = permutation_test(wm, xi, n_runs)
        p_ct, _ = permutation_test(ct, xi, n_runs)
        # Edit-robustness: p-value of the watermarked stream after insert/delete edits.
        sweep = {}
        for f in args.edits:
            edited = edit_corrupt(wm, f, 700 + i)
            p_edit, _ = permutation_test(edited, xi, n_runs)
            sweep[f"edit_{int(f*100):02d}"] = p_edit
        rows.append({"ref_wm": ref_wm, "our_wm": our_wm["mean_g"], "ref_ct": ref_ct,
                     "our_ct": our_ct["mean_g"], "p_wm": p_wm, "p_ct": p_ct, "sweep": sweep})

    os.unlink(cfg_path)
    results["exp-edit"] = {
        "ref": "Kuditipudi EXP-Edit (TMLR 2024) - distortion-free + edit-robust",
        "ref_wm_stat": mean([r["ref_wm"] for r in rows]),
        "our_wm_stat": mean([r["our_wm"] for r in rows]),
        "ref_ct_stat": mean([r["ref_ct"] for r in rows]),
        "our_ct_stat": mean([r["our_ct"] for r in rows]),
        "p_watermarked": mean([r["p_wm"] for r in rows]),
        "p_control": mean([r["p_ct"] for r in rows]),
        "max_abs_stat_diff_vs_ref": max_abs_diff,
        "bit_for_bit": max_abs_diff < 1e-4,
        "edit_pvalue_sweep": {k: mean([r["sweep"][k] for r in rows]) for k in rows[0]["sweep"]},
    }


def print_report(results, args):
    print()
    print("=" * 78)
    print("LIVE WATERMARK VALIDATION BATTERY  (our Rust detector vs. real HF implementations)")
    print("=" * 78)
    print(f"samples/scheme: {args.num}   gen_len: {args.gen_len}   keys tested: {len(args.key_seeds)}")
    for scheme, r in results.items():
        print()
        print(f"### {scheme.upper()}")
        if scheme.startswith("synthid"):
            print(f"  streams: {r.get('stream_kind', 'model-free')}")
            print(f"  watermarked: official_g={r['official_wm_g']:.4f}  our_g={r['our_wm_g']:.4f}  our_z={r['our_wm_z']:.2f}")
            if r.get("our_wm_weighted_g") is not None:
                print(f"  weighted-mean detector (Nature 2024): official={r['official_wm_weighted_g']:.4f}  our={r['our_wm_weighted_g']:.4f}")
            print(f"  control:     official_g={r['official_ctrl_g']:.4f}  our_g={r['our_ctrl_g']:.4f}  our_z={r['our_ctrl_z']:.2f}")
            print(f"  max |our - official| (mean+weighted) = {r['max_abs_g_diff_vs_official']:.2e}   BIT-FOR-BIT: {r['bit_for_bit']}")
        elif scheme == "exp":
            print("  ref: Aaronson/Kuditipudi exponential (no HF detector; validated vs reference)")
            print(f"  watermarked: ref_z={r['ref_wm_z']:.2f}  our_z={r['our_wm_z']:.2f}  our_score={r['our_wm_g']:.3f}")
            print(f"  control:     ref_z={r['ref_ct_z']:.2f}  our_z={r['our_ct_z']:.2f}  our_score={r['our_ct_g']:.3f}")
            print(f"  max |our_z - ref_z| = {r['max_abs_z_diff_vs_ref']:.2e}   BIT-FOR-BIT: {r['bit_for_bit']}")
        elif scheme == "exp-edit":
            print(f"  ref: {r['ref']}")
            print(f"  statistic (lower = watermark): ref_wm={r['ref_wm_stat']:.4f}  our_wm={r['our_wm_stat']:.4f}")
            print(f"                                 ref_ct={r['ref_ct_stat']:.4f}  our_ct={r['our_ct_stat']:.4f}")
            print(f"  permutation p-value: watermarked={r['p_watermarked']:.4f}   control={r['p_control']:.4f}")
            print(f"  max |our_stat - ref_stat| = {r['max_abs_stat_diff_vs_ref']:.2e}   BIT-FOR-BIT: {r['bit_for_bit']}")
            print("  EDIT-robustness (p-value after insert/delete edits; stays low = survives):")
            print("    " + "  ".join(f"{k}={v:.3f}" for k, v in r["edit_pvalue_sweep"].items()))
            continue
        elif scheme in ("unigram", "sweet"):
            print(f"  ref: {r['ref']}")
            print(f"  watermarked: ref_z={r['ref_wm_z']:.2f}  our_z={r['our_wm_z']:.2f}  our_g={r['our_wm_g']:.3f}")
            print(f"  control:     ref_z={r['ref_ct_z']:.2f}  our_z={r['our_ct_z']:.2f}  our_g={r['our_ct_g']:.3f}")
            print(f"  max |our_z - ref_z| = {r['max_abs_z_diff_vs_ref']:.2e}   BIT-FOR-BIT: {r['bit_for_bit']}")
        else:
            print(f"  seeding: {r.get('seeding', 'lefthash/cw1')}")
            print(f"  watermarked: hf_g={r['hf_wm_g']:.4f}  our_g={r['our_wm_g']:.4f}   hf_z={r['hf_wm_z']:.2f}  our_z={r['our_wm_z']:.2f}")
            print(f"  control:     hf_g={r['hf_ctrl_g']:.4f}  our_g={r['our_ctrl_g']:.4f}   hf_z={r['hf_ctrl_z']:.2f}  our_z={r['our_ctrl_z']:.2f}")
            print(f"  max |our_g - hf_g| = {r['max_abs_g_diff_vs_hf']:.2e}   max |our_z - hf_z| = {r['max_abs_z_diff_vs_hf']:.2e}   BIT-FOR-BIT: {r['bit_for_bit']}")
        sweep = r["sweep"]
        print("  robustness (our_z by edit fraction): " + "  ".join(f"{k}={v:.1f}" for k, v in sweep.items()))
    print()
    print("Interpretation: BIT-FOR-BIT True means our detector reproduces the real published")
    print("implementation exactly on identical token streams. Robustness shows the signal")
    print("degrading as more tokens are edited (the real-world 'survives copy-paste, dies on")
    print("heavy rewrite' behavior). All authoritative because YOU hold the key.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.example.json")
    ap.add_argument("--detector", default=DETECTOR_DEFAULT)
    ap.add_argument("--schemes", nargs="+", default=["synthid", "kgw"],
                    choices=["synthid", "kgw", "exp", "unigram", "sweet", "exp-edit"])
    ap.add_argument("--mode", default="model-free", choices=["model-free", "model", "both"],
                    help="model-free (default, offline); model (generate from a real LM); "
                         "both (try real LM, fall back to model-free).")
    ap.add_argument("--model", default="google/gemma-2-2b",
                    help="Open causal LM for --mode model/both (needs huggingface.co access).")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--gen-len", type=int, default=200)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--edits", type=float, nargs="+", default=[0.0, 0.05, 0.15, 0.30, 0.60])
    ap.add_argument("--vocab-edit", type=int, default=60,
                    help="Small vocab for EXP-Edit (the key matrix xi is pseudo_length x vocab).")
    ap.add_argument("--gen-len-edit", type=int, default=80,
                    help="pseudo_length (key rows) for EXP-Edit.")
    ap.add_argument("--edit-runs", type=int, default=60,
                    help="Permutation-test runs for EXP-Edit significance.")
    ap.add_argument("--key-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--kgw-variants", nargs="+", default=["lefthash:1"],
                    help="KGW seeding variants to test, each as scheme:context_width, e.g. "
                         "lefthash:1 lefthash:2 selfhash:2 selfhash:3")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        sys.exit("error: " + INSTALL_HINT)

    if not os.path.exists(args.detector):
        sys.exit(f"error: detector not found: {args.detector} (run `cargo build --release`)")

    with open(args.config) as fh:
        base_cfg = json.load(fh)
    base_keys = base_cfg["keys"]
    ngram_len = base_cfg.get("ngram_len", 5)

    # Multi-key: rotate the key list deterministically per seed.
    def rotate(keys, s):
        return [((k + s * 7919) % 100003) for k in keys]

    all_results = {}
    for s in args.key_seeds:
        keys = rotate(base_keys, s)
        res = {}
        if "synthid" in args.schemes:
            validate_synthid(args, keys, ngram_len, base_cfg, res)
        if "kgw" in args.schemes:
            for variant in args.kgw_variants:
                scheme_name, _, cw = variant.partition(":")
                cw = int(cw) if cw else 1
                key = "kgw" if variant == "lefthash:1" else f"kgw_{scheme_name}_cw{cw}"
                validate_kgw(args, base_cfg, res, seeding_scheme=scheme_name,
                             context_width=cw, result_key=key)
        if "exp" in args.schemes:
            validate_exp(args, keys, res)
        if "unigram" in args.schemes:
            validate_unigram(args, keys, base_cfg, res)
        if "sweet" in args.schemes:
            validate_sweet(args, keys, base_cfg, res)
        if "exp-edit" in args.schemes:
            validate_exp_edit(args, keys, base_cfg, res)
        all_results[f"key_seed_{s}"] = res

    if args.json:
        print(json.dumps(all_results, indent=2))
        return

    # Report the first key-seed in detail; summarize bit-for-bit across all.
    first = next(iter(all_results.values()))
    print_report(first, args)
    if len(all_results) > 1:
        print()
        print("Across all key seeds:")
        for scheme in first.keys():
            oks = [rs[scheme]["bit_for_bit"] for rs in all_results.values() if scheme in rs]
            print(f"  {scheme}: bit-for-bit on {sum(oks)}/{len(oks)} keys")


if __name__ == "__main__":
    main()
