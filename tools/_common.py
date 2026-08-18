"""Shared helpers for the contributor validation tools.

Every tool in this directory drives the same Rust detector over JSON configs and averages
z-scores across trials, so the plumbing lives here rather than being copied per script.
Tools are invoked as ``python3 tools/<tool>.py`` from the repo root, which puts this
directory on ``sys.path`` and makes ``import _common`` resolve directly.
"""

import json
import subprocess
import tempfile


def run_detector(detector: str, config_path: str, scheme: str, ids) -> dict:
    """Invoke the Rust detector in JSON mode for one token stream and parse its output."""
    cmd = [detector, "score", "--config", config_path, "--scheme", scheme, "--json",
           "--tokens", " ".join(str(i) for i in ids)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"detector failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout.strip())


def write_config(cfg: dict, extra: dict | None = None) -> str:
    """Write a detector config to a temp JSON file and return its path.

    ``extra`` is merged over ``cfg`` when provided, so callers can layer scheme-specific
    keys onto a shared base without mutating it.
    """
    if extra is not None:
        cfg = {**cfg, **extra}
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def mean(xs) -> float:
    """Mean of the finite values in ``xs`` (NaNs are dropped); NaN if none remain."""
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")
