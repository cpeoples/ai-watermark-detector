#!/usr/bin/env python3
"""Tokenizer bridge: turn raw text into token IDs the Rust detector accepts.

IMPORTANT: WHAT "THE KEY" MEANS
-------------------------------
KGW and SynthID are *keyed statistical* watermarks. There is no visible mark embedded in
the text. Instead, at each token the generator used a SECRET number ("the key") to split
the vocabulary into "preferred" and "not preferred" and leaned slightly toward preferred
tokens. The only trace is a faint statistical bias. To *measure* that bias a detector must
reproduce the exact same secret split - which requires the key. Without the key there is
literally no signal to read; watermarked text is statistically identical to unwatermarked
text. This is a cryptographic design property, not a limitation we can engineer around.
(Contrast: C2PA file provenance is a SIGNED record you can verify with a PUBLIC key - use
the Rust CLI's `check`/`scan` subcommands. Different mechanism entirely.)

Two consequences for this bridge:

  * A detector also needs the EXACT tokenizer used at generation, because the watermark
    lives in token-ID space. There is no universally "correct" tokenizer for arbitrary text.
  * For a CONTROLLED experiment (you generated the text with a known model/key/tokenizer),
    pass that tokenizer here and the resulting IDs are exactly right - detection is then
    authoritative (see tools/validate.py).
  * For arbitrary/real vendor text, any tokenizer choice here is an APPROXIMATION and,
    lacking the vendor's private key, cannot be an authoritative verifier. The detector
    will (correctly) show NO SIGNAL on real vendor text.

Usage
-----
  # Using a Hugging Face tokenizer (recommended for open models):
  python3 tools/tokenize_text.py --hf gpt2 --file sample.txt > ids.txt
  python3 tools/tokenize_text.py --hf gpt2 --text "hello world" > ids.txt

  # Using tiktoken (OpenAI encodings):
  python3 tools/tokenize_text.py --tiktoken cl100k_base --file sample.txt > ids.txt

  # Then score with the Rust detector:
  ai-watermark-detector --config config.example.json --token-file ids.txt

Install one of:
  pip install tokenizers        # for --hf
  pip install tiktoken          # for --tiktoken
"""

import argparse
import sys


def read_text(args) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        with open(args.file, "r", encoding="utf-8") as fh:
            return fh.read()
    return sys.stdin.read()


def encode_hf(model: str, text: str) -> list[int]:
    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit(
            "error: `tokenizers` not installed.\n"
            "       pip install -r tools/requirements.txt   (or: pip install tokenizers)\n"
            "       (or use --tiktoken with `pip install tiktoken`)"
        )
    # Try a pretrained tokenizer by name; fall back to loading a local file.
    try:
        tok = Tokenizer.from_pretrained(model)
    except Exception:
        tok = Tokenizer.from_file(model)
    return tok.encode(text).ids


def encode_tiktoken(encoding: str, text: str) -> list[int]:
    try:
        import tiktoken
    except ImportError:
        sys.exit(
            "error: `tiktoken` not installed.\n"
            "       pip install -r tools/requirements.txt   (or: pip install tiktoken)\n"
            "       (or use --hf with `pip install tokenizers`)"
        )
    enc = tiktoken.get_encoding(encoding)
    return enc.encode(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--text", help="Literal text to tokenize.")
    src.add_argument("--file", help="Path to a UTF-8 text file to tokenize.")

    kind = ap.add_mutually_exclusive_group(required=True)
    kind.add_argument("--hf", metavar="NAME_OR_PATH", help="Hugging Face tokenizer name or local tokenizer.json path.")
    kind.add_argument("--tiktoken", metavar="ENCODING", help="tiktoken encoding name, e.g. cl100k_base, o200k_base.")

    ap.add_argument("--sep", default=" ", help="Separator between IDs in output (default: space).")
    args = ap.parse_args()

    text = read_text(args)
    if not text:
        sys.exit("error: no input text provided (use --text, --file, or pipe via stdin).")

    if args.hf:
        ids = encode_hf(args.hf, text)
    else:
        ids = encode_tiktoken(args.tiktoken, text)

    sys.stdout.write(args.sep.join(str(i) for i in ids))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
