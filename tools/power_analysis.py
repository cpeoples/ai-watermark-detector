#!/usr/bin/env python3
"""Statistical power / ROC analysis for the watermark detectors.

This answers the two questions a real deployment cares about, empirically and honestly:

  1. DETECTION POWER: given a watermark of a certain strength, how many tokens do you need
     before you can flag it at a target confidence (e.g. FPR <= 1%)? We sweep token length
     and measure the true-positive rate at a z-threshold calibrated to the target FPR on
     genuine unwatermarked controls.
  2. ROC: at a fixed length, sweep the decision threshold and trace TPR vs FPR, reporting
     the area under the curve (AUC). AUC = 0.5 is coin-flip; 1.0 is perfect separation.

It drives the SAME machinery the validation battery uses (our Rust detector fed by an
exported HF green-list oracle for KGW, or by the reference u-oracle for exp), so the curves
reflect the real detector, not a toy. Everything is authoritative because YOU hold the key.

Output is a compact table plus optional JSON; with --plot it also writes PNGs (needs
matplotlib). No claim is made about production vendor text — this measures the detector's
intrinsic statistical power on streams you watermark yourself.

Usage
-----
  python3 tools/power_analysis.py --scheme kgw --lengths 25 50 100 200 400 --trials 200
  python3 tools/power_analysis.py --scheme exp --roc-length 200 --trials 300 --json
"""

import argparse
import hashlib
import json
import math
import os
import sys

from _common import run_detector as our_score, write_config


DETECTOR_DEFAULT = "target/release/ai-watermark-detector"


def build_kgw(args):
    """Return (cfg_path, gen_wm, gen_ctrl) for the KGW scheme using HF's real greenlist."""
    import torch
    from transformers import WatermarkLogitsProcessor

    vocab = args.vocab
    gamma = args.gamma
    lp = WatermarkLogitsProcessor(
        vocab_size=vocab, device="cpu", greenlist_ratio=gamma, bias=4.0,
        hashing_key=15485863, seeding_scheme="lefthash", context_width=1,
    )
    oracle = {}
    for tok in range(vocab):
        oracle[str(tok)] = sorted(int(x) for x in lp._get_greenlist_ids(torch.tensor([tok])).tolist())

    cfg_path = write_config({"ngram_len": 5, "keys": [654, 400, 836],
                             "gamma": gamma, "vocab_size": vocab,
                             "green_oracle": oracle, "seeding_scheme": "lefthash",
                             "context_width": 1})

    def gen_wm(n, seed, strength):
        # strength in [0,1]: probability of actually taking a green token at each step
        # (models a watermark of tunable strength; strength=1 is maximal).
        g = torch.Generator().manual_seed(seed)
        seq = [int(torch.randint(0, vocab, (1,), generator=g).item())]
        while len(seq) < n:
            greens = oracle[str(seq[-1])]
            if torch.rand(1, generator=g).item() < strength and greens:
                idx = int(torch.randint(0, len(greens), (1,), generator=g).item())
                seq.append(greens[idx])
            else:
                seq.append(int(torch.randint(0, vocab, (1,), generator=g).item()))
        return seq

    def gen_ctrl(n, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, vocab, (n,), generator=g).tolist()

    return cfg_path, gen_wm, gen_ctrl, "kgw"


def build_exp(args):
    """Return (cfg_path, gen_wm, gen_ctrl) for the exponential scheme."""
    import torch

    vocab = args.vocab
    key = 654

    def u_value(prev, token):
        h = hashlib.sha256()
        h.update(f"{key}|{prev}|{int(token)}".encode())
        return (int.from_bytes(h.digest()[:8], "big") >> 11) / float(1 << 53)

    # Oracle is built lazily over the pairs we actually produce/score.
    oracle = {}

    def note(prev, token):
        k = f"{prev}|{token}"
        if k not in oracle:
            oracle[k] = u_value(prev, token)

    def gen_wm(n, seed, strength, cand=8):
        g = torch.Generator().manual_seed(seed)
        seq = [int(torch.randint(0, vocab, (1,), generator=g).item())]
        while len(seq) < n:
            prev = seq[-1]
            cands = torch.randint(0, vocab, (cand,), generator=g).tolist()
            if torch.rand(1, generator=g).item() < strength:
                pick = max(cands, key=lambda c: u_value(prev, c))
            else:
                pick = cands[0]
            note(prev, pick)
            seq.append(pick)
        return seq

    def gen_ctrl(n, seed):
        g = torch.Generator().manual_seed(seed)
        seq = torch.randint(0, vocab, (n,), generator=g).tolist()
        for i in range(1, len(seq)):
            note(seq[i - 1], seq[i])
        return seq

    # Config path is (re)written after generation to include the full oracle.
    return oracle, gen_wm, gen_ctrl, "exp"


def z_threshold_for_fpr(control_zs, target_fpr):
    """Empirical z-threshold: the (1 - target_fpr) quantile of control z-scores."""
    xs = sorted(control_zs)
    idx = min(len(xs) - 1, int(math.ceil((1 - target_fpr) * len(xs))) - 1)
    return xs[max(0, idx)]


def auc(tprs, fprs):
    # Trapezoidal AUC with fpr ascending.
    pts = sorted(zip(fprs, tprs))
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return area


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", choices=["kgw", "exp"], default="kgw")
    ap.add_argument("--detector", default=DETECTOR_DEFAULT)
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="Watermark strength in [0,1]; lower = weaker/harder to detect.")
    ap.add_argument("--lengths", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    ap.add_argument("--roc-length", type=int, default=200)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--plot", metavar="DIR", help="Write ROC/power PNGs to DIR (needs matplotlib).")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        sys.exit('error: needs the reference implementations.\n  pip install -r tools/requirements.txt')
    if not os.path.exists(args.detector):
        sys.exit(f"error: detector not found: {args.detector} (run `cargo build --release`)")

    if args.scheme == "kgw":
        cfg_path, gen_wm, gen_ctrl, scheme = build_kgw(args)
        exp_oracle = None
    else:
        exp_oracle, gen_wm, gen_ctrl, scheme = build_exp(args)
        cfg_path = None

    def score_many(streams):
        nonlocal cfg_path
        if scheme == "exp":
            # (Re)write config with the accumulated oracle for exp.
            cfg_path = write_config({"ngram_len": 5, "keys": [654], "context_width": 1,
                                     "u_oracle": dict(exp_oracle)})
        return [our_score(args.detector, cfg_path, scheme, s)["z"] for s in streams]

    # --- Power curve: TPR at target FPR vs. token length ---
    power_rows = []
    for n in args.lengths:
        wm = [gen_wm(n, 1000 + i, args.strength) for i in range(args.trials)]
        ct = [gen_ctrl(n, 5000 + i) for i in range(args.trials)]
        wm_z = score_many(wm)
        ct_z = score_many(ct)
        thr = z_threshold_for_fpr(ct_z, args.target_fpr)
        tpr = sum(1 for z in wm_z if z > thr) / len(wm_z)
        fpr = sum(1 for z in ct_z if z > thr) / len(ct_z)
        power_rows.append({"length": n, "z_threshold": thr, "tpr": tpr, "fpr": fpr,
                           "mean_wm_z": sum(wm_z) / len(wm_z), "mean_ct_z": sum(ct_z) / len(ct_z)})

    # --- ROC at a fixed length ---
    n = args.roc_length
    wm = [gen_wm(n, 2000 + i, args.strength) for i in range(args.trials)]
    ct = [gen_ctrl(n, 6000 + i) for i in range(args.trials)]
    wm_z = score_many(wm)
    ct_z = score_many(ct)
    thresholds = sorted(set([round(z, 3) for z in (wm_z + ct_z)]))
    roc = []
    for thr in thresholds:
        tpr = sum(1 for z in wm_z if z > thr) / len(wm_z)
        fpr = sum(1 for z in ct_z if z > thr) / len(ct_z)
        roc.append({"threshold": thr, "tpr": tpr, "fpr": fpr})
    roc_auc = auc([r["tpr"] for r in roc], [r["fpr"] for r in roc])

    if cfg_path and os.path.exists(cfg_path):
        os.unlink(cfg_path)

    result = {"scheme": scheme, "vocab": args.vocab, "gamma": args.gamma,
              "strength": args.strength, "trials": args.trials,
              "target_fpr": args.target_fpr, "power": power_rows,
              "roc_length": n, "roc_auc": roc_auc}

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print()
        print("=" * 68)
        print(f"POWER / ROC ANALYSIS  [{scheme}]  vocab={args.vocab} gamma={args.gamma} "
              f"strength={args.strength} trials={args.trials}")
        print("=" * 68)
        print(f"Detection power at target FPR <= {args.target_fpr:.0%} (z-threshold calibrated on controls):")
        print(f"  {'tokens':>7}{'z_thresh':>10}{'TPR':>8}{'FPR':>8}{'mean_wm_z':>11}{'mean_ct_z':>11}")
        for r in power_rows:
            print(f"  {r['length']:>7}{r['z_threshold']:>10.2f}{r['tpr']:>8.2%}{r['fpr']:>8.2%}"
                  f"{r['mean_wm_z']:>11.2f}{r['mean_ct_z']:>11.2f}")
        print(f"\nROC at {n} tokens: AUC = {roc_auc:.4f}  (0.5 = chance, 1.0 = perfect)")
        # find smallest length reaching 95% TPR
        good = [r["length"] for r in power_rows if r["tpr"] >= 0.95]
        if good:
            print(f"Tokens needed for >=95% detection at this FPR: ~{min(good)}")
        else:
            print("No tested length reached 95% detection at this FPR (try longer / higher strength).")
        print("\nHonest note: this is the detector's intrinsic power on streams YOU watermark.")
        print("It says nothing about production vendor text, where the secret key is unavailable.")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            sys.exit("error: --plot needs matplotlib (pip install matplotlib)")
        os.makedirs(args.plot, exist_ok=True)
        plt.figure()
        plt.plot([r["fpr"] for r in roc], [r["tpr"] for r in roc], marker=".")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title(f"ROC [{scheme}] AUC={roc_auc:.3f} @ {n} tokens")
        roc_png = os.path.join(args.plot, f"roc_{scheme}.png")
        plt.savefig(roc_png, dpi=120, bbox_inches="tight")
        plt.figure()
        plt.plot([r["length"] for r in power_rows], [r["tpr"] for r in power_rows], marker="o")
        plt.axhline(0.95, color="r", ls="--", alpha=0.5, label="95% TPR")
        plt.xlabel("Tokens"); plt.ylabel(f"TPR at FPR<= {args.target_fpr:.0%}")
        plt.title(f"Detection power vs length [{scheme}]"); plt.legend()
        power_png = os.path.join(args.plot, f"power_{scheme}.png")
        plt.savefig(power_png, dpi=120, bbox_inches="tight")
        print(f"\nWrote {roc_png} and {power_png}")


if __name__ == "__main__":
    main()
