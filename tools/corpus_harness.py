#!/usr/bin/env python3
"""Corpus harness: score labeled text samples and tabulate the watermark signal.

This implements the validation experiment:

    corpus/
      claude/                *.txt   (raw Claude output)
      claude-copy-paste/     *.txt   (Claude output copied & pasted)
      claude-minor-edits/    *.txt
      claude-heavy-edits/    *.txt
      claude-gpt-rewrite/    *.txt
      claude-human-rewrite/  *.txt
      human/                 *.txt
      gpt/                   *.txt
      gemini/                *.txt

For each sample it: tokenizes -> runs the Rust detector -> collects the z-score.
It then prints per-category summary statistics (n, mean/median/min/max z, mean p).

HONESTY NOTE
------------
Without Anthropic's private watermark key + exact tokenizer, this measures whether a
SynthID-compatible signal exists under the *config you provide*. On real Claude text
scored with a guessed key it will (correctly) show no separation. Its intended use is
a CONTROLLED experiment: generate watermarked text with a KNOWN key/tokenizer and
confirm the harness recovers the signal and shows the expected degradation under
copy/paste, edits, and rewrites.

Usage
-----
  python3 tools/corpus_harness.py \
    --corpus corpus \
    --config config.example.json \
    --tiktoken cl100k_base \
    --detector target/release/ai-watermark-detector

  # or with a Hugging Face tokenizer:
  python3 tools/corpus_harness.py --corpus corpus --config config.example.json --hf gpt2
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class Score:
    path: str
    tokens: int
    z: float
    mean_g: float
    p: float
    reliable: bool = True


@dataclass
class CategoryStats:
    name: str
    scores: list[Score] = field(default_factory=list)

    def zs(self) -> list[float]:
        return [s.z for s in self.scores]

    def summary(self) -> dict:
        zs = self.zs()
        if not zs:
            return {"category": self.name, "n": 0}
        return {
            "category": self.name,
            "n": len(zs),
            "mean_z": statistics.mean(zs),
            "median_z": statistics.median(zs),
            "min_z": min(zs),
            "max_z": max(zs),
            "stdev_z": statistics.pstdev(zs) if len(zs) > 1 else 0.0,
            "mean_p": statistics.mean(s.p for s in self.scores),
            "mean_tokens": statistics.mean(s.tokens for s in self.scores),
            "n_reliable": sum(1 for s in self.scores if s.reliable),
        }


def tokenize(args, text: str) -> str:
    """Invoke the tokenizer bridge, returning a whitespace-separated id string."""
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "tokenize_text.py")]
    if args.hf:
        cmd += ["--hf", args.hf]
    else:
        cmd += ["--tiktoken", args.tiktoken]
    proc = subprocess.run(cmd, input=text, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"tokenizer failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def score(args, ids: str) -> dict:
    """Invoke the Rust detector in JSON mode and parse its output."""
    cmd = [args.detector, "score", "--config", args.config, "--scheme", args.scheme, "--json", "--tokens", ids]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"detector failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout.strip())


def iter_samples(corpus_dir: str):
    """Yield (category, filepath) for every *.txt under each category subdir."""
    for entry in sorted(os.listdir(corpus_dir)):
        cat_dir = os.path.join(corpus_dir, entry)
        if not os.path.isdir(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if fname.endswith(".txt"):
                yield entry, os.path.join(cat_dir, fname)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="Directory of labeled category subdirs.")
    ap.add_argument("--config", required=True, help="Detector JSON config (ngram_len + keys).")
    ap.add_argument("--detector", default="target/release/ai-watermark-detector", help="Path to the Rust detector binary.")
    ap.add_argument("--scheme", default="synthid", choices=["synthid", "kgw"], help="Watermark family to score against.")
    kind = ap.add_mutually_exclusive_group(required=True)
    kind.add_argument("--hf", metavar="NAME_OR_PATH")
    kind.add_argument("--tiktoken", metavar="ENCODING")
    kind.add_argument("--pretokenized", action="store_true", help="Corpus files already contain token IDs; skip tokenization.")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table.")
    args = ap.parse_args()

    if not os.path.isdir(args.corpus):
        sys.exit(f"error: corpus dir not found: {args.corpus}")
    if not os.path.exists(args.detector):
        sys.exit(f"error: detector binary not found: {args.detector} (build with `cargo build --release`)")

    cats: dict[str, CategoryStats] = {}
    for category, path in iter_samples(args.corpus):
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        try:
            ids = text.strip() if args.pretokenized else tokenize(args, text)
            if not ids:
                continue
            res = score(args, ids)
        except RuntimeError as e:
            print(f"warning: skipping {path}: {e}", file=sys.stderr)
            continue
        cats.setdefault(category, CategoryStats(category)).scores.append(
            Score(
                path=path,
                tokens=res["tokens"],
                z=res["z"],
                mean_g=res["mean_g"],
                p=res["approx_p_value"],
                reliable=res.get("reliable", True),
            )
        )

    summaries = [cats[c].summary() for c in sorted(cats)]

    if args.json:
        print(json.dumps({"scheme": args.scheme, "categories": summaries}, indent=2))
        return

    if not summaries:
        print("No samples scored. Populate the corpus dir with category subfolders of *.txt files.")
        return

    print(f"Watermark signal by category  (scheme: {args.scheme}, config: {args.config})")
    print("=" * 84)
    print(f"{'category':<24}{'n':>4}{'rel':>5}{'mean_z':>9}{'med_z':>8}{'max_z':>8}{'mean_p':>10}{'mean_tok':>10}")
    print("-" * 84)
    for s in summaries:
        if s["n"] == 0:
            print(f"{s['category']:<24}{0:>4}")
            continue
        print(
            f"{s['category']:<24}{s['n']:>4}{s['n_reliable']:>5}{s['mean_z']:>9.2f}{s['median_z']:>8.2f}"
            f"{s['max_z']:>8.2f}{s['mean_p']:>10.4f}{s['mean_tokens']:>10.0f}"
        )
    print("-" * 84)
    print("rel = samples clearing the ~100-token reliability floor.")
    print("Higher mean_z / lower mean_p => stronger apparent watermark signal.")
    print("Expect (controlled experiment): claude >> other-scheme-watermark ~ human/gpt/gemini,")
    print("with signal degrading across copy-paste < minor-edits < heavy-edits < rewrite.")


if __name__ == "__main__":
    main()
