# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `--format text|json|xml|yaml` on `score`, `check`, and `scan` for machine-readable output
  (all serialized through `serde`; `--json` kept as a shortcut for `--format json`).
- `check`/`scan` now accept `http(s)://` URLs: a remote file is downloaded and its C2PA
  provenance is verified like a local file.
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
