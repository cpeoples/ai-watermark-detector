# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Unsigned metadata AI-generation signals for `check`/`scan` (medium/low confidence,
  separate from the signed C2PA verdict): PNG text chunks, XMP/IPTC keys, MP4/MKV/WebM/AVI/
  FLV video and WAV/FLAC/OGG/AAC audio container tags, the TC260 `AIGC` label, the xAI/Grok
  EXIF signature scheme, embedded C2PA-in-text manifests (C2PA 2.4 A.7/A.8/A.9 presence), and
  filename patterns, with a recognized-tool mapping (Firefly, DALL-E, Midjourney, Stable
  Diffusion, ComfyUI, Apple Image Playground, Microsoft Designer/Copilot, Canva, Claude, and
  more) and the normative IPTC `digitalSourceType` value `trainedAlgorithmicMedia`. Adds a
  `signals` field to JSON/XML/YAML output and a `meta-hint` column to the `scan` table.
  Visible-logo and neural (e.g. TrustMark) detection are out of scope to keep the tool
  deterministic and dependency-light.
- `check` now reports two more details from the signed manifest: the forensic `soft-binding`
  watermark it declares, named from the C2PA Soft Binding Algorithm List (Adobe TrustMark,
  Digimarc, IMATAG, Meta Seal, NexGuard, or the raw registered id), and an implied SynthID pixel
  watermark when a valid signature comes from a vendor that always pairs C2PA with SynthID
  (Google, OpenAI).
  AI markers are collected across the whole manifest chain, so a marker on a parent/ingredient
  manifest (not just the active one) is detected, and a clean active manifest over an AI
  ingredient is surfaced as a provenance conflict. Recognizes the `compositeSynthetic`
  digital-source-type in addition to the trained/composite AI types.
- `--format text|json|xml|yaml` on `score`, `check`, and `scan` for machine-readable output
  (all serialized through `serde`; `--json` kept as a shortcut for `--format json`).
- `check`/`scan` now accept `http(s)://` URLs: a remote file is downloaded and its C2PA
  provenance is verified like a local file.
- Added a real Adobe-signed PDF to the `--fetch-samples` set, covering the PDF path end-to-end.
- MIT `LICENSE`, `Makefile` (`build`/`test`/`install`/`uninstall`/`lint`/`demo`),
  `.editorconfig`, and GitHub Actions CI.
- Plain-English, verdict-first CLI output (`WATERMARK SIGNAL FOUND` /
  `NO WATERMARK SIGNAL` / `TOO SHORT TO TELL`) with p-values translated into human odds.
- Per-OS setup instructions and a single `tools/requirements.txt` for the contributor
  Python tools.

### Fixed
- `c2patool` discovery on Windows now also matches `c2patool.exe`.
- `Cargo.toml` license metadata aligned to MIT.

## [0.1.0]

### Added
- Rust CLI with three subcommands: `score` (text watermark), `check` and `scan`
  (C2PA file provenance).
- Six bit-for-bit text watermark schemes: `kgw`, `synthid`, `exp`, `unigram`, `sweet`,
  `exp-edit`, validated against the official reference implementations.
- Three Bricks grounded p-values, SynthID weighted-mean detector.
- Contributor Python tooling: validation battery, attack/robustness battery,
  power/ROC analysis, tokenizer bridge, and pixel-domain image watermark detection
  (Stable Signature forensic + labelled SynthID-Image heuristic).
