<div align="center">

# AI Watermark Detector

**Is this text or file AI-generated? A research tool that answers in plain English.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Language: Rust](https://img.shields.io/badge/core-Rust-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-blue.svg)]()
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-green.svg)](.github/workflows/ci.yml)

</div>

---

Checks whether a piece of **text** or a **file** (image / video / audio / PDF) carries one
of the known "watermarks" that AI companies hide in their output - and tells you the answer
in plain English. Six text-watermark schemes reproduced **bit-for-bit** against their
official papers, plus **forensic C2PA** file-provenance verification.

## Quick Install

The core is a single Rust binary. First install Rust if `cargo --version` doesn't already
work.

On **macOS/Linux**, run the [rustup](https://rustup.rs) bootstrap:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

On **Windows**, download and run [`rustup-init.exe`](https://rustup.rs), then reopen your
shell so `cargo` is on `PATH` (in PowerShell you can install it via `winget`):

```powershell
winget install --id Rustlang.Rustup -e
```

Then build the release binary:

```bash
cargo build --release
```

The binary lands at `target/release/ai-watermark-detector` (`ai-watermark-detector.exe` on
Windows). Run it with `./target/release/ai-watermark-detector --help`
(`.\target\release\ai-watermark-detector.exe --help` in PowerShell).

To put it on your `PATH`, either `cargo install --path .` or, on macOS/Linux, `make install`
(copies it to `~/.local/bin`). Then:

```bash
ai-watermark-detector --help
make demo    # build a demo corpus and score watermarked vs. human text
```

## What you actually get

Run it on some text and you get a verdict like this:

```text
=== AI watermark screen: WATERMARK SIGNAL FOUND ===
This text carries the 'kgw' statistical watermark for the key you supplied.
Chance a random unwatermarked text would score this high: about less than 1 in a trillion.
```

or, on a file:

```text
file: photo.jpg
  plain answer:    AI-generated, and the provenance is cryptographically TRUSTED (signer verified).
```

Three possible text verdicts, in words:

- **WATERMARK SIGNAL FOUND** - the text matches the watermark for the key you gave it.
- **NO WATERMARK SIGNAL** - no trace of *that* watermark. (This does **not** prove a human
  wrote it - only that this specific watermark+key isn't present.)
- **TOO SHORT TO TELL** - there isn't enough text to make a reliable statistical call
  (you need roughly 100+ words).

## The one thing to understand first (please read)

There are **two completely different** kinds of AI watermark, and they behave differently:

| | **Text watermarks** | **File watermarks (C2PA)** |
|---|---|---|
| What it is | An *invisible statistical bias* in word choice | A *signed receipt* attached to the file |
| Can you check it yourself? | **Only if you have the secret key** | **Yes - verified with a public key** |
| Who has the key? | The AI vendor keeps it private, forever | Anyone (it's public-key crypto, like HTTPS) |
| This tool's role | Research reproduction of the *mechanism* | A **real, forensic** check on real files |

**Why the secret key matters:** a text watermark leaves *no visible mark*. At each word the
AI used a secret number (the "key") to nudge its choices. The only trace is a faint bias
spread across many words. To measure that bias you must reproduce the exact same keyed
nudge - which requires the key. **Without the key there is literally nothing to detect**;
watermarked text is mathematically identical to unwatermarked text. That's a deliberate
cryptographic design choice, not a limitation we can code around. It's why this tool
**cannot** tell you "this ChatGPT/Claude/Gemini paragraph is AI" from raw text alone.
It *can* prove the watermarking mechanism works, on text you watermark yourself with a
known key - which is what real watermarking research does.

Files are different: C2PA is a signed receipt (like the padlock in your browser), so this
tool really does verify it and tell you TRUSTED / UNTRUSTED / TAMPERED.

## Three commands

| Command | What it does | Needs a key? | Needs `c2patool`? |
|---|---|---|---|
| `score` | Scores token IDs against a text watermark (kgw/synthid/exp/unigram/sweet/exp-edit) | Yes (the vendor's, or your own) | No |
| `check` | Verifies C2PA provenance on files/folders/URLs, one verdict per file | No | Yes |
| `scan`  | Same check, aggregated into a per-format coverage table | No | Yes |

Run `ai-watermark-detector <command> --help` for the full argument list; a compact reference
is in [CLI reference](#cli-reference) below.

## Quickstart (60 seconds, no Python needed)

```bash
# 1. Build the one binary (needs Rust: https://rustup.rs)
cargo build --release

# 2. Make a tiny demo corpus watermarked with a KNOWN key, then score it
./target/release/gen_corpus /tmp/demo kgw
./target/release/ai-watermark-detector score \
  --config config.example.json --scheme kgw --token-file /tmp/demo/watermarked/sample_00.txt
# -> WATERMARK SIGNAL FOUND

# 3. Score an unwatermarked sample for contrast
./target/release/ai-watermark-detector score \
  --config config.example.json --scheme kgw --token-file /tmp/demo/human/sample_00.txt
# -> NO WATERMARK SIGNAL

# 4. Check a file's provenance (needs c2patool; see Setup)
./target/release/ai-watermark-detector check photo.jpg
```

On **Windows**, use `target\release\ai-watermark-detector.exe`, a path like
`%TEMP%\demo` instead of `/tmp/demo`, and backslash path separators (or run the commands
verbatim from Git Bash / WSL).

## The watermark families we cover (plain English)

**Text** - six statistical schemes from the two big research families. We reproduce each
one **bit-for-bit** against its official published code, so you can *prove the mechanism*
on text you watermark yourself:

| Scheme (`--scheme`) | Family | Used by / origin | One-line description |
|---|---|---|---|
| `kgw` | green-list | **matches Anthropic's description of Claude** | Splits the vocabulary into "preferred/not" each step and leans toward preferred words. |
| `synthid` | sampling | **the family Gemini uses** | Google's "tournament" g-value scheme (Nature 2024); we match the strong weighted detector. |
| `exp` | sampling | Aaronson / OpenAI-style | Gumbel "exponential" trick; per-word score `-ln(1-u)`. |
| `unigram` | green-list | Zhao et al. 2024 | One fixed global split - the most edit-robust of the green-list schemes. |
| `sweet` | green-list | Lee et al. 2024 | KGW but only on high-information words; the standard **code** watermark. |
| `exp-edit` | sampling | Kuditipudi et al. 2024 | Edit-distance statistic that survives insertions/deletions others don't. |

**Files** - real forensic provenance via **C2PA** on `jpg/png/mp4/wav/pdf`: reads the
signed manifest, validates the signing certificate against a trust list, flags AI-generation
markers and Google **SynthID** assertions, and detects tampering. This is the only check
here that gives an authoritative answer on real vendor output (e.g. Google Imagen/Veo
images that ship with C2PA).

**Images (pixel-domain), contributor tool** - `tools/image_watermark.py` does **Stable
Signature** (Meta, forensic - real 48-bit extractor + exact p-value) and a labelled
**SynthID-Image heuristic** (no public decoder exists, so it's clearly marked *not
forensic*). See the pixel-domain section below.

## Does this "detect every AI"? Honest answer

- **AI text from a real product (Claude / Gemini / ChatGPT):** **No** - the vendor's key is
  private (and ChatGPT/Grok ship no text watermark at all). No public tool can do this from
  raw text; that's the cryptographic wall described above.
- **AI *files* with C2PA (many Google/Adobe/Microsoft pipelines):** **Often yes** - `check`
  gives a real forensic verdict.
- **Any watermark where you *have* the key:** **Yes, exactly** - that's the research use, and
  if a vendor ever publishes a key you can plug it straight in (see "If a key becomes
  public").

## Setup by operating system

You only need **Rust** to run the shipped detector. Python is **contributor-only** (for
proving the Rust engine matches the official reference code).

### 1. Rust CLI (all platforms) - this is the product

Install Rust from <https://rustup.rs>, then:

```bash
cargo build --release
./target/release/ai-watermark-detector --help
```

The binary is self-contained: no Python, no `pip`. Works on macOS, Linux, and Windows
(the same `cargo build --release`; on Windows the binary is
`target\release\ai-watermark-detector.exe`).

### 2. `c2patool` - only needed for the `check` / `scan` (file provenance) commands

| OS | Install |
|---|---|
| macOS | `brew install c2patool` |
| Linux | `cargo install c2patool` |
| Windows | `cargo install c2patool` (or download `c2patool.exe` from [Releases](https://github.com/contentauth/c2patool/releases)) |

The detector finds both `c2patool` and `c2patool.exe` on your `PATH`. If it's missing, the
tool **fails closed** with the exact install command for your platform. The text `score`
command never needs it.

### 3. Python contributor tools (optional) - validation, attacks, images

Use a virtualenv. On **macOS/Linux**:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r tools/requirements.txt
```

On **Windows** (PowerShell):

```powershell
python3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r tools\requirements.txt
```

`tools/requirements.txt` covers all contributor tools (validation, images, tokenizer).

Every Python tool starts with a `#!/usr/bin/env python3` shebang and **fails closed** with a
clear `pip install -r ...` message if a dependency is missing - it never crashes with a raw
`ImportError`.

## Rust vs Python: which file do I run?

Simple rule: **if you *run* it to get an answer, it's Rust. If it *proves* the Rust engine
is correct, it's Python.**

| You want to… | Run | Language | Needs |
|---|---|---|---|
| Score text for a watermark | `ai-watermark-detector score` | Rust | just the binary |
| Check a file's provenance | `ai-watermark-detector check` / `scan` | Rust | binary + `c2patool` |
| Prove the engine matches official code | `tools/validate.py` | Python | `tools/requirements.txt` |
| Measure attack robustness / ROC | `tools/attacks.py`, `tools/power_analysis.py` | Python | `tools/requirements.txt` |
| Detect image watermarks | `tools/image_watermark.py` | Python | `tools/requirements.txt` |
| Turn text into token IDs | `tools/tokenize_text.py` | Python | `tools/requirements.txt` |

Everything under `tools/` is Python and contributor-only; everything you invoke as
`./target/release/...` is Rust and is the shipped product.

**Why two languages?** The detector ships in Rust so it's a single fast, dependency-free
binary you can `curl | install` and run offline. The `tools/` are Python because their job
is to diff our detector against the *official* reference implementations - Hugging Face
`transformers`, MarkLLM, and Meta's Stable Signature model - which exist only as
Python/PyTorch. Reimplementing those in Rust would mean checking our code against our own
code, which proves nothing; running the vendors' actual Python is what makes the validation
trustworthy. Users never touch Python; contributors use it to prove the Rust is correct.

## Output formats

Both Rust subcommands print **human-readable** text by default (verdict first, then
details) and can emit **machine-readable** output with `--format json|xml|yaml`:

```bash
# human (default): verdict-first, plain English
./target/release/ai-watermark-detector score --config config.example.json --scheme kgw --token-file ids.txt

# machine-readable: pick json, xml, or yaml for scripts / pipelines
./target/release/ai-watermark-detector score --config config.example.json --scheme kgw --token-file ids.txt --format json
./target/release/ai-watermark-detector score --config config.example.json --scheme kgw --token-file ids.txt --format yaml
./target/release/ai-watermark-detector score --config config.example.json --scheme kgw --token-file ids.txt --format xml
```

(`--json` is still accepted as a shortcut for `--format json`.) See
[CLI reference](#cli-reference) for the full list of fields each command emits.

## If a key becomes public (or you generated the text yourself): 100% path

This tool becomes an **exact** scorer the moment the scheme, key, and tokenizer all match
the text - the detectors are reproduced bit-for-bit against the official implementations
(validated in the correctness suite and `tools/validate.py`). You already get this on a
corpus you watermark yourself, since you own the key. If a vendor ever publishes their
production key, you plug it straight into the config:

```json
{ "ngram_len": 5, "keys": [<the published key ints>], "gamma": 0.25, "vocab_size": <vocab> }
```

then tokenize with the vendor's exact tokenizer and score:

```bash
python3 tools/tokenize_text.py --hf <vendor/tokenizer> --file suspect.txt > ids.txt
./target/release/ai-watermark-detector score --config config.example.json --scheme <kgw|synthid|...> --token-file ids.txt
```

Honest caveat: "100% accurate" means the *statistic* is computed exactly (bit-for-bit with
the vendor's math). Detection is still statistical - long watermarked text gives
astronomically small p-values (effectively certain), but very short or heavily-edited text
can stay ambiguous no matter what. For **files**, C2PA is already a today-verifiable public-key
check - no waiting on any key.

## Build

```bash
cargo build --release
./target/release/ai-watermark-detector --help
```

The CLI has three subcommands: `score` (text watermark), `check` (file provenance),
`scan` (per-format coverage).

## CLI reference

Every command prints human-readable text by default and accepts `--format text|json|xml|yaml`
for machine-readable output (`--json` is a shortcut for `--format json`).

**`score`** - score token IDs against a text watermark:

| Argument | Description |
|---|---|
| `--tokens <IDS>` | Comma/space-separated token IDs (mutually exclusive with `--token-file`). |
| `--token-file <PATH>` | File of whitespace/comma-separated token IDs. |
| `--config <PATH>` | JSON config: `ngram_len`, `keys`, and scheme extras (see [Config](#config)). Required. |
| `--scheme <NAME>` | `synthid` (default), `kgw`, `exp`, `unigram`, `sweet`, or `exp-edit`. |
| `--threshold <F>` | Minimum confidence for a local "positive" (default `0.95`). |
| `--format <FMT>` | `text` (default), `json`, `xml`, `yaml`. |

**`check`** - verify C2PA provenance for files, folders, or URLs:

| Argument | Description |
|---|---|
| `<FILES>...` | Files, folders (walked recursively), or `http(s)://` URLs. |
| `--fetch-samples` | Download real publicly-signed sample files and check them. |
| `--samples-dir <DIR>` | Where fetched samples are stored (default `samples`). |
| `--trust-anchors <PEM>` | PEM trust-anchor list (path or URL) for real cert-chain validation. |
| `--format <FMT>` | `text` (default), `json`, `xml`, `yaml`. |

**`scan`** - same check, aggregated into a per-format coverage table:

| Argument | Description |
|---|---|
| `<FILES>...` | Files, folders (walked recursively), or `http(s)://` URLs. |
| `--trust-anchors <PEM>` | PEM trust-anchor list (path or URL) for cert-chain validation. |
| `--format <FMT>` | `text` (default), `json`, `xml`, `yaml`. |

The `score` result carries `scheme`, `tokens`, `positions`, `usable_positions`,
`green_or_ones`, `total`, `mean_g`, `weighted_mean_g` (SynthID only), `grounded_p_value`
(schemes with a closed-form null), `z`, `approx_p_value`, `reliable`, `screen_positive`, and
a `warning`. `check`/`scan` emit a `results` list of per-file `verdict`, `has_manifest`,
`ai_source_type`, `algorithmic_source`, `synthid_assertion`, `assertions`, `status`, and
`signals` (each with `confidence`, `source`, `detail`, and an optional `tool`).

## Scoring a token sequence

The detector consumes **token IDs**, not words (see "Why token IDs?" below). Pick the
scheme with `--scheme`:

```bash
# KGW green-list scheme (Kirchenbauer et al.)
./target/release/ai-watermark-detector score \
  --config config.example.json --scheme kgw --token-file tokens.txt

# Gemini-style SynthID scheme (default)
./target/release/ai-watermark-detector score \
  --config config.example.json --scheme synthid --token-file tokens.txt --json
```

### Config

`config.example.json` carries settings for both schemes:

```json
{
  "ngram_len": 5,
  "keys": [654, 400, 836, ...],
  "gamma": 0.25,
  "vocab_size": 8000
}
```

- `ngram_len`, `keys` - used by both schemes.
- `gamma` - KGW green-list fraction (ignored by SynthID).
- `vocab_size` - KGW vocabulary size for the green/red split (ignored by SynthID; if
  omitted it is inferred from the maximum observed token id).
- `sampling_table` (optional, SynthID) - the Bernoulli table exported from Google's
  official processor. When present, the SynthID scheme matches Google bit-for-bit; when
  absent, a standalone reproduction is used. See real-world validation A below.
- `green_oracle` (optional, KGW) - a `last_token -> green_token_ids` map exported from
  HF's real `lefthash` processor. When present, the KGW scheme matches HF bit-for-bit;
  when absent, a standalone reproduction is used.

## Why token IDs?

Both schemes hash *token sequences*, not visible words. The same visible text produces
different token sequences under different tokenizers, so a text-only detector cannot
reproduce a watermark score without the exact tokenizer.

## Checking files & folders (C2PA provenance)

`check` and `scan` both accept any mix of individual files and directories; directories are
walked recursively (hidden entries are skipped). This works on real vendor output today -
no secret key needed - because C2PA is a public-key signature.

```bash
# one file
ai-watermark-detector check photo.jpg

# several files at once
ai-watermark-detector check a.jpg b.png clip.mp4

# a remote file by URL (downloaded, then checked like a local file)
ai-watermark-detector check https://example.com/photo.jpg

# an entire directory, walked recursively - a per-file verdict for each
ai-watermark-detector check ./downloads

# a directory, aggregated into a per-format coverage table (best for many files)
ai-watermark-detector scan ./downloads
```

Use **`check`** when you want a detailed verdict per file, and **`scan`** when you have a
folder full of files and want the overall picture. `scan` prints a table like:

```text
ext         files  w/manifest  AI-marked
jpg             4           4          1
mp4             1           1          0
png             1           0          0
```

Each `check` result is one of: **TRUSTED** (signed and the signer chains to your trust
list), **VALID-UNTRUSTED** (signature is cryptographically valid but the signer isn't on
your list), **TAMPERED** (altered after signing), or **NO-MANIFEST** (no provenance record
- inconclusive, *not* proof a human made it). Add `--trust-anchors <PEM>` to check signers
against a trust list, and `--format json|xml|yaml` for machine-readable output.

### Unsigned metadata signals (lower confidence)

C2PA is the gold standard, but only ~19% of AI media carries it. So alongside the signed
verdict, `check`/`scan` also read **unsigned** hints natively from the file bytes and report
them at **medium/low** confidence - clearly separated from the cryptographic verdict, because
this metadata can be forged or stripped:

- **PNG text chunks** (`parameters`, `prompt`) that Stable Diffusion / ComfyUI / A1111 embed.
- **XMP / IPTC metadata** keys like `AISystemUsed` / `DigitalSourceType` (e.g. DALL-E, Firefly).
- **Container tags** across MP4/MOV, MKV/WebM, AVI, FLV, WAV/FLAC/OGG/AAC audio, and MP3 ID3
  naming AI video/music tools (Sora, Kling, Suno, ElevenLabs…), plus the China **TC260 `AIGC`**
  label.
- **Filename patterns** (e.g. `ElevenLabs_…`, Midjourney/Suno prefixes) as a low-confidence hint.

These never override a C2PA verdict; a signed manifest is always HIGH, metadata is MEDIUM, and
filenames are LOW. In `scan`, the `AI-marked` column counts signed C2PA AI provenance while a
separate `meta-hint` column counts files flagged only by unsigned metadata.

From the signed manifest itself, `check` also surfaces two HIGH-confidence details when present:
the **forensic soft-binding watermark** it declares (e.g. `com.adobe.trustmark.Q`, Digimarc,
Imatag), and an **implied SynthID** pixel watermark - a valid signature from a vendor that always
pairs C2PA with SynthID (Google, OpenAI) implies the watermark even without an explicit
assertion. Both ride on the signature, so they carry the same trust as the verdict. AI markers
are gathered across the **whole manifest chain**, not just the active manifest, since the marker
often lives on a parent/ingredient (an editor re-signing an AI-generated original).

Unsigned formats without C2PA are also covered where a vendor uses a distinctive scheme, such as
the **xAI/Grok** EXIF `Signature` blob paired with a UUID artist field (reported MEDIUM). Text and
source files are scanned too: C2PA 2.4 can embed a signed manifest directly in text (invisible
Unicode variation selectors, an ASCII-armour block, or an inline HTML element), which `c2patool`
does not read - `check` flags the presence of one (signature not verified) as a MEDIUM hint.

When AI provenance sits on a parent/ingredient manifest while the current signer's manifest
declares no AI, `check` reports a **provenance conflict** - a valid signature can assert a
human-only edit over an AI-generated original (the "Integrity Clash").

**Deliberately out of scope.** This stays a deterministic, dependency-light tool, so it does
**not** do visible-logo template matching (e.g. erasing the Gemini sparkle), neural
invisible-watermark decoding (Adobe TrustMark, AudioSeal, WavMark, or SynthID pixel surrogates),
or online verification oracles. Those need a CV/CNN model or a network round-trip and produce
probabilistic or non-reproducible results - they belong to ML-based tools and hosted APIs, not a
local cryptographic verifier.

You can also pass an `http(s)://` URL anywhere a file path is accepted; it is downloaded to
a temp file and checked identically. Note this verifies the fetched file's C2PA
provenance - it is *not* a general "is this image AI?" oracle for unsigned pixels (a
screenshot or re-saved image has no manifest, so it reads as inconclusive).

## Research toolkit

Three components turn this from a bare scorer into a validation harness. None change the
limitation above - they let you *prove the mechanism* on a corpus you watermark with a
**known** key.

### 1. Self-test (correctness anchor)

Proves each scheme's math separates known-watermarked token streams from random ones,
**and** that the two schemes are independent (cross-scheme watermarks look like noise).
No external dependencies.

```bash
cargo test
```

### 2. Tokenizer bridge (text → token IDs)

`tools/tokenize_text.py` converts raw text into token IDs using a real tokenizer library. For
a controlled experiment, pass the *same* tokenizer used at generation; for arbitrary text
this is only an approximation.

```bash
# OpenAI encodings (pip install tiktoken)
python3 tools/tokenize_text.py --tiktoken cl100k_base --file sample.txt > ids.txt

# Hugging Face tokenizers (pip install tokenizers)
python3 tools/tokenize_text.py --hf gpt2 --file sample.txt > ids.txt

./target/release/ai-watermark-detector score --config config.example.json --scheme kgw --token-file ids.txt
```

> Note: the tokenizer bridge is `tools/tokenize_text.py` (named to avoid shadowing
> Python's stdlib `tokenize` module).

### 3. Corpus harness + controlled corpus

Generate a labeled corpus with a **known** key under either scheme and tabulate the signal
across categories, matching the validation matrix (Claude / copy-paste / edits / rewrite /
human / GPT / Gemini), plus an `other-scheme-watermark` category to demonstrate
cross-scheme independence:

```bash
cargo build --release

# KGW green-list corpus + KGW detector
./target/release/gen_corpus corpus kgw
python3 tools/corpus_harness.py --corpus corpus --config config.example.json --pretokenized --scheme kgw

# Gemini-style (SynthID) corpus + SynthID detector
./target/release/gen_corpus corpus synthid
python3 tools/corpus_harness.py --corpus corpus --config config.example.json --pretokenized --scheme synthid
```

Example output (KGW corpus, KGW detector, known key):

```text
category                   n  rel   mean_z   med_z   max_z    mean_p  mean_tok
watermarked               12   12    69.22   69.31   69.91    0.0000      1500
watermarked-copy-paste    12   12    69.22   69.31   69.91    0.0000      1500
watermarked-minor-edits   12   12    53.66   53.56   57.20    0.0000      1500
watermarked-heavy-edits   12   12    11.10   11.21   13.17    0.0000      1500
watermarked-rewrite       12   12     0.15   -0.01    1.60    0.4689      1500
other-scheme-watermark    12   12     0.08   -0.15    1.64    0.4865      1500
gpt                       12   12     0.12   -0.08    2.01    0.4801      1500
gemini                    12   12    -0.13   -0.20    1.80    0.5470      1500
human                     12   12     0.02   -0.18    3.02    0.5320      1500
```

Interpretation:

- **`watermarked` / `watermarked-copy-paste`** - strong signal; survives copy/paste.
- **edits** - signal degrades monotonically (minor -> heavy).
- **`watermarked-rewrite`** - heavy paraphrase washes the watermark out (baseline).
- **`other-scheme-watermark`** - watermarked under the *other* family; sits at baseline,
  proving detectors don't cross schemes.
- **`gpt` / `gemini` / `human`** - unwatermarked baselines at the coin-flip line.

The `rel` column counts samples clearing the ~100-token reliability floor.

To score real (non-pretokenized) text corpora, drop `--pretokenized` and pass
`--tiktoken`/`--hf`.

## Detectable today? A quick reality check

**Text** (keyed statistical watermark - needs the vendor's secret key):

| Provider (text) | Mechanism | Key public? | Authoritative detection here? |
|---|---|---|---|
| Claude | KGW-family green-list | No | No - research reproduction only |
| Gemini | SynthID-Text | No | No - Google tooling only |
| ChatGPT | none shipped | - | Nothing to detect |
| Grok (xAI) | no public evidence | - | Nothing to detect |

We reproduce **six published mechanisms bit-for-bit** - KGW (`lefthash`+`selfhash`, any
context width), SynthID-Text (mean **and** weighted-mean detectors), the exponential/Gumbel
scheme, Unigram (global split), SWEET (entropy-gated, for code), and EXP-Edit (distortion-free
+ edit-robust) - spanning both families in the watermarking literature (KGW/logit-biasing and
Christ/sampling). But detecting any vendor's **production** text needs their secret key. The
"AI detectors" that claim to catch ChatGPT text are stylistic classifiers (perplexity /
burstiness), not watermark readers - this tool deliberately does not pretend to do that.

**Files** (signed provenance - public-key verifiable, so checkable *today*):

| Content type | Provenance | Verifiable here? |
|---|---|---|
| Images (JPEG/PNG/WebP/AVIF/…) | C2PA manifest (+ SynthID for Google) | Yes - `... check` |
| Video (MP4/MOV) | C2PA manifest | Yes |
| Audio (MP3/WAV/M4A) | C2PA manifest (OpenAI audio) | Yes |
| PDF | C2PA (spec-supported, adoption thin) | Yes when signed |
| **.docx / .pptx / .txt / source code** | **none in practice** | **No provenance to check** |

**Pixel-domain image watermarks** (invisible marks *inside* the pixels, not metadata - a
separate axis from C2PA), via `tools/image_watermark.py`:

| Scheme | Mechanism | Decoder public? | Status here |
|---|---|---|---|
| **Stable Signature** (Meta) | k=48-bit CNN-extractable signature | **Yes** (HiDDeN extractor) | **Forensic with the key** - real Meta extractor + exact Binomial(48,0.5) p-value |
| **Tree-Ring** | ring pattern in the noise-latent FFT | needs the diffusion model | Documented, **not faked** - requires DDIM inversion, no standalone decoder |
| **SynthID-Image** (Google) | proprietary pixel watermark | **No** | **Heuristic only** - spectral surrogate (~90% on controlled refs), never forensic |

The key split: *files* can carry a signed, publicly-verifiable manifest; *text content*
(including text pasted into a `.docx` or a code file) can only carry the invisible,
key-gated statistical watermark. That's why AI-generated **code** is the hardest case -
low-entropy and container-less.

## Real-world validation you *can* run

The strongest honest proof available: our Rust detector reproduces the **real, published
implementations** shipped in Hugging Face `transformers` - **bit-for-bit** - across many
conditions. Because you choose the key, you're the key-holder and detection is authoritative.

### A. Live validation battery vs. the real implementations (`tools/validate.py`)

Drives Google's actual `SynthIDTextWatermarkLogitsProcessor` **and** Kirchenbauer's actual
`WatermarkLogitsProcessor` + `WatermarkDetector`, then checks our detector against them.

```bash
pip install "transformers>=4.46" torch      # one-time
python3 tools/validate.py --schemes synthid kgw --num 6 --gen-len 200 --key-seeds 0 1 2

# Optional: also generate from a real open LM via Google's official watermarking config
python3 tools/validate.py --schemes synthid --mode model --model google/gemma-2-2b

# All seeding variants + every extra scheme (KGW family + Christ family):
python3 tools/validate.py --schemes kgw --kgw-variants lefthash:1 lefthash:2 selfhash:2 selfhash:3
python3 tools/validate.py --schemes exp unigram sweet exp-edit --num 4 --gen-len 200
```

Observed result:

```text
### SYNTHID
  watermarked: official_g=0.6277  our_g=0.6277  our_z=17.47
  weighted-mean detector (Nature 2024): official=0.6300  our=0.6300
  control:     official_g=0.5028  our_g=0.5028  our_z=0.38
  max |our - official| (mean+weighted) = 4.70e-07   BIT-FOR-BIT: True

### KGW  (lefthash/cw1, lefthash/cw2, selfhash/cw2, selfhash/cw3 - all BIT-FOR-BIT True)
  watermarked: hf_g=0.9008  our_g=0.9008   hf_z=21.20  our_z=21.20
  control:     hf_g=0.2425  our_g=0.2425   hf_z=-0.25  our_z=-0.25
  max |our_g - hf_g| = 4.02e-07   max |our_z - hf_z| = 3.64e-05   BIT-FOR-BIT: True

### EXP  (Aaronson/Kuditipudi; no HF detector - validated vs the reference formula)
  watermarked: ref_z=24.58  our_z=24.58   control: ref_z=-0.39  our_z=-0.39   BIT-FOR-BIT: True

### UNIGRAM  (Zhao 2024, global fixed split - validated vs the MarkLLM reference)
  watermarked: ref_z=20.84  our_z=20.84   control: ref_z=0.46  our_z=0.46     BIT-FOR-BIT: True

### SWEET  (Lee 2024, entropy-gated KGW for code - validated vs the MarkLLM reference)
  watermarked: ref_z=16.12  our_z=16.12   control: ref_z=-0.32  our_z=-0.32   BIT-FOR-BIT: True

### EXP-EDIT  (Kuditipudi 2024, distortion-free + edit-robust - Levenshtein alignment stat)
  statistic (lower=watermark): ref_wm=-180.66  our_wm=-180.66   ref_ct=-81.17  our_ct=-81.17
  permutation p-value: watermarked=0.016   control=0.60         BIT-FOR-BIT: True
  EDIT-robustness (p after insert/delete): edit_00=0.016  edit_30=0.016  edit_50=0.016  <- survives

Across all key seeds:
  synthid: bit-for-bit on 3/3 keys
  kgw: bit-for-bit on 3/3 keys
```

The battery validates, for every scheme: **bit-for-bit agreement** with the reference
implementation (HF `transformers` for KGW/SynthID incl. the weighted-mean detector; the
MarkLLM / published reference for exp, unigram, sweet, exp-edit), a **cross-check** against
HF's own KGW detector z-score, **separation** (watermarked vs. control), **robustness**
sweeps, and **multi-key** stability. It runs fully offline once `transformers` is installed
(it drives the official logits processors directly); no model download is required.

**How we reach bit-for-bit:** neither torch's sampling table (SynthID), its `torch.randperm`
green-list (KGW), nor the exp keyed PRNG can be recomputed in Rust. So the Python side
exports the exact ground truth into the config - SynthID's `sampling_table`, KGW's
`green_oracle` (keyed by the seeding prefix, so it covers `lefthash` *and* `selfhash` at any
context width), the exp `u_oracle`, Unigram's global `unigram_greenlist`, SWEET's
`entropy_mask`, and EXP-Edit's key matrix `exp_edit_xi`. Our Rust detector consumes these and
matches the real implementations exactly. Note: even Google's open hashing differs from the
**Gemini App's**, so this validates the *mechanism*, not production Gemini/Claude.

### Attack / robustness battery - what actually erases a watermark (`tools/attacks.py`)

The surveys (Liu et al. 2024; MarkLLM's evaluation module) judge a watermark by its survival
under edits. `tools/attacks.py` reproduces the token-level core of MarkLLM's attack suite -
**substitution**, **deletion**, **insertion**, **random-walk** (repeated local edits), and
**block-shuffle** (reordering) - and reports survival curves on the real detector.

```bash
python3 tools/attacks.py --scheme kgw --trials 60 --length 300
python3 tools/attacks.py --scheme unigram --trials 60 --length 300   # contrast robustness
```

Observed (mean detector z at each attack fraction; higher = still detectable):

```text
KGW lefthash/cw1 (baseline z 17.3)      UNIGRAM global split (baseline z 17.2)
attack           0.10  0.30  0.50       attack           0.10  0.30  0.50
substitution     13.9   8.5   4.2       substitution     15.4  11.9   8.7
deletion         14.7  10.1   5.9       deletion         16.3  14.3  12.1
insertion        14.9  10.7   7.0       insertion        16.3  15.0  14.1
random-walk      14.2   9.5   6.3       random-walk      15.6  12.8  10.4
block-shuffle    17.3  17.1  17.0       block-shuffle    17.2  17.2  17.2
```

Two findings that match the literature exactly: (1) **Unigram is far more edit-robust** than
per-token KGW under insertion/deletion - its global split is position-independent, which is
precisely Zhao et al.'s "provable robustness" claim. (2) **Block-shuffle barely dents either**
greenlist scheme, because reordering intact spans preserves local greenlist relationships. For
the scheme designed to survive edits outright, run `validate.py --schemes exp-edit`: its
permutation p-value stays ~0.016 even after 50% insert/delete edits.

### Detection power / ROC - how many tokens do you actually need?

`tools/power_analysis.py` measures the detector's intrinsic statistical power: it sweeps
token length and decision threshold over many watermarked/control trials and reports TPR at a
calibrated FPR, plus a ROC AUC. This empirically quantifies the reliability floor.

```bash
python3 tools/power_analysis.py --scheme kgw --lengths 25 50 100 200 400 800 \
  --strength 0.15 --trials 120 --target-fpr 0.01 --plot out/
```

Observed (a deliberately *weak* watermark, strength 0.15, FPR ≤ 1%):

```text
  tokens  z_thresh     TPR     FPR
      25      2.83   4.17%   0.83%
      50      2.23  27.50%   0.83%
     100      1.68  80.00%   0.83%
     400      3.15  97.50%   0.83%
     800      2.23 100.00%   0.83%
ROC at 200 tokens: AUC = 0.9957
Tokens needed for >=95% detection at this FPR: ~400
```

At full strength the watermark is trivially separable (AUC = 1.0 by ~25 tokens); weakened, it
takes hundreds of tokens - a concrete, measured version of "survives on long text, vanishes on
short/edited spans."

> `tools/validate.py` is the single contributor validation entry point. It covers **all six
> schemes**, multi-key rotation, robustness sweeps, MarkLLM/HF cross-checks, and (with
> `--mode model`) real-LM generation via Google's official `SynthIDTextWatermarkingConfig`.

### Grounded p-values (Three Bricks) - exact tails, not a Gaussian approximation

The Rust `score` command also emits a `grounded_p_value` for the greenlist schemes (KGW,
Unigram, SWEET) and the exponential scheme, following Fernandez et al.'s *Three Bricks to
Consolidate Watermarks* (2023). Instead of the normal z-approximation, it uses the **exact
null distribution**: the Binomial(k, γ) upper tail via the regularized incomplete beta for
greenlist schemes, and the Gamma(n, 1) tail via the regularized upper incomplete gamma for the
exp scheme. These are far better calibrated on short texts. Our Rust `betainc`/`gammaincc`
match `scipy.special` to machine precision (**betainc: ~2e-31, gammaincc: ~5e-16**), covered by
a Rust self-test.

### C. Pixel-domain image watermarks (`tools/image_watermark.py`)

A separate axis from text and C2PA: invisible marks embedded *in the pixels* of diffusion-model
images. Read the honesty labels - they are the point.

**Stable Signature (Meta, ICCV 2023) - forensic when you hold the key.** Every image from a
Stable-Signature model carries a 48-bit signature recoverable by Meta's **public** HiDDeN
extractor CNN. The tool downloads that real extractor, decodes the bits, and reports the exact
false-positive rate from the paper (Eq. 2): `FPR = I_{1/2}(matches, k−matches+1)` - the same
Binomial(48, 0.5) tail (and the same `betainc`) validated above.

```bash
pip install torch torchvision pillow numpy scipy
python3 tools/image_watermark.py stable-signature IMAGE.png --key <48-bit-string> --fetch-extractor
python3 tools/image_watermark.py self-test        # proves both directions through the real CNN
```

Proven end-to-end through Meta's actual extractor (`self-test`):

```text
embedded a random 48-bit key -> 47/48 matches, p=1.7e-13  => WATERMARKED (flagged)
random control image         -> 20/48 matches, p=9.0e-01  => correctly inconclusive
```

The positive path uses the paper's own "unauthorized embedding" procedure (Sec 7.1): gradient
descent through the real extractor until the image decodes to a chosen key. It's a
demonstration of the detector on real weights, not a claim about any vendor's production image.

**SynthID-Image (Google) - heuristic only, never forensic.** Google ships **no public decoder**.
The `synthid-image` subcommand exposes a spectral phase-coherence surrogate (the reverse-SynthID
approach), which reaches ~90% only on *controlled references* and does **not** reliably separate
real content. It is labeled as a weak hint in every output; do not treat it as proof.

**Tree-Ring - deliberately not faked.** Detection requires DDIM-inverting the exact diffusion
model to recover the noise latent (no standalone image decoder exists), so we document it rather
than ship a misleading standalone check.

### B. C2PA content provenance across file formats (`... check` / `... scan`) - forensic

Unlike secret-keyed text watermarks, **C2PA** manifests are *public-key signed* provenance
records embedded in generated files. You can authoritatively verify them today - no private
key needed. C2PA is **container-agnostic**, spanning:

- **Images:** JPEG, PNG, WebP, AVIF, HEIC/HEIF, TIFF, DNG, SVG, GIF
- **Video:** MP4, MOV (BMFF/ISO containers)
- **Audio:** MP3, WAV, M4A
- **Docs:** PDF (spec-supported; vendor adoption thin)

The CLI wraps the official `c2patool`, fetches real signed samples on demand, flags
**AI-generation markers** (the IPTC `digitalSourceType`, e.g. *trainedAlgorithmicMedia* =
fully AI-generated) and any **SynthID** assertion (Google's marker on Gemini/Imagen images
& Veo video), and can produce a per-format coverage report - **all in the Rust binary, no
Python**:

```bash
brew install c2patool                                    # or: cargo install c2patool
./target/release/ai-watermark-detector check --fetch-samples   # real signed samples -> samples/
./target/release/ai-watermark-detector check some_file.jpg --json
./target/release/ai-watermark-detector scan samples           # per-format coverage reality-check

# Real cert-chain validation against the official C2PA trust list:
c2patool init trust                                                # fetch official trust list
./target/release/ai-watermark-detector check some_file.jpg \
  --trust-anchors ~/.config/c2pa/c2pa-trust-list.pem
```

**Forensic verdicts.** With `--trust-anchors`, every file gets one of four verdicts, each
demonstrated on real files:

| Verdict | Meaning | How it's reached |
|---|---|---|
| `TRUSTED` | signature valid *and* signer chains to a provided anchor | signer CA in the trust/allowed list |
| `VALID-UNTRUSTED` | signature cryptographically valid, signer not in the list | e.g. test-signed samples vs. the production list |
| `TAMPERED` | content/manifest altered after signing | `assertion.dataHash.mismatch` after a byte edit |
| `NO-MANIFEST` | no provenance to check | `.txt`, `.docx`, source code, unsigned media |

Observed detail (a real image signed with an AI digital-source-type):

```text
file: ai_generated_marked.jpg
  manifest:        yes
  AI marker:       fully AI-generated (trained algorithmic media)  <-- flagged AI-generated
  status:          manifest present; validation issues: signingCredential.untrusted
```

Observed coverage report (the honest reality check):

```text
ext         files  w/manifest  AI-marked
jpg             3           3          2
mp4             1           1          0
png             1           0          0
wav             1           0          0
txt             1           0          0
py              1           0          0
```

**The cross-format truth:** signed image/video/audio carry checkable provenance; but
`.docx`, `.pptx`, `.txt`, and **source-code files carry NO C2PA manifest in practice**, so
their only possible trace is the key-gated statistical text watermark (which needs the
vendor's secret key). Also: C2PA lives *alongside* the file and is trivially stripped by
format conversion, re-saving, or screenshots - a present valid manifest is strong positive
evidence, but its absence is inconclusive.

### Does this detect Gemini?

- **Gemini text:** No. Same cryptographic wall as Claude - Google's key is private, and
  Google's *own* source notes its open hashing differs from the production Gemini App's. We
  reproduce the SynthID *mechanism* bit-for-bit (validated above), not production Gemini text.
- **Gemini images/video (Imagen, Veo, "Nano Banana"):** Often **yes** - Google attaches
  SynthID and increasingly C2PA provenance. When a C2PA manifest is present, `... check`
  reads and validates it and flags the AI marker / SynthID assertion. That is a real,
  forensic "this came from a generative pipeline" signal on actual files.

## Swapping in a vendor detector later

When a vendor ships a real detection API, the harness can point at it as an alternate
backend (replacing the local `--detector`) while the experimental scorer remains available
for research.

## Development

```bash
make build      # cargo build --release
make test       # run the Rust test suite
make lint       # clippy (if installed)
make demo       # build a demo corpus and score watermarked vs. human
make install    # install the binary to ~/.local/bin
make uninstall  # remove it
make check-tools  # report optional tool availability (c2patool, python)
```

CI (`.github/workflows/ci.yml`) builds, formats, clippy-lints, and tests on macOS, Linux,
and Windows, and syntax-checks every Python tool. Tagging a release (`git tag v0.1.0 &&
git push --tags`) triggers `release.yml`, which publishes prebuilt binaries for all three
platforms.

## Contributing

Contributions welcome:

1. Fork and create a feature branch.
2. Run `make test` (and `make lint`) before opening a PR.
3. If you touch a detector, add/extend a bit-for-bit check in `tools/validate.py`.
4. Keep the honesty rules: label heuristics as heuristics, never fake a forensic verdict.

## License

MIT - see [LICENSE](LICENSE).

---

<div align="center">

**If this tool helped, ⭐ star the repo.**

</div>
