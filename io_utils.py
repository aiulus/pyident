# pyident/io_utils.py
from __future__ import annotations
import csv
import json
import os
import sys
import platform
import shutil
import subprocess
from typing import Dict, Any, Optional, Iterable

import numpy as np


# -------------------------- paths & atomics ---------------------------

def ensure_dir(path: str) -> None:
    """Create directory if not exists. No-op for '' (current dir)."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _atomic_write_bytes(data: bytes, path: str) -> None:
    tmp = path + ".tmp"
    ensure_dir(os.path.dirname(path))
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _atomic_write_text(text: str, path: str) -> None:
    _atomic_write_bytes(text.encode("utf-8"), path)


# -------------------------- JSON / CSV / NPZ --------------------------

def _np_json_encoder(obj: Any) -> Any:
    """Best-effort JSON encoder for NumPy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def save_json(data: Dict[str, Any], path: str) -> None:
    """Atomically save JSON with pretty indent; handles NumPy scalars."""
    ensure_dir(os.path.dirname(path))
    text = json.dumps(data, indent=2, default=_np_json_encoder)
    _atomic_write_text(text, path)


def save_npz(arrs: Dict[str, np.ndarray], path: str) -> None:
    """Atomically save compressed NPZ."""
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def save_csv(rows: list[dict], path: str, fieldnames: Optional[list[str]] = None) -> None:
    """Atomically save CSV (writes header)."""
    ensure_dir(os.path.dirname(path))
    if fieldnames is None and rows:
        # union of keys across all rows so late-coming fields (e.g., est.moesp.error) are included
        ks = set()
        for r in rows:
            ks.update(r.keys())
        fieldnames = sorted(ks)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})
    os.replace(tmp, path)


# Convenience: append rows without re-writing the entire file
def append_csv(rows: Iterable[dict], path: str, fieldnames: Optional[list[str]] = None) -> None:
    ensure_dir(os.path.dirname(path))
    file_exists = os.path.exists(path)
    if not file_exists and fieldnames is None:
        # infer from first row
        rows = list(rows)
        if not rows:
            return
        fieldnames = list(rows[0].keys())
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------- “light” I/O helpers -----------------------

def maybe_save_npz(
    arrs: Dict[str, np.ndarray],
    path: str,
    light: bool = False,
    keys_keep: Optional[set[str]] = None,
    max_mb: float = 25.0,
) -> None:
    """Save arrays, honoring a 'light' mode.

    - If light=False: save everything.
    - If light=True: only save arrays named in keys_keep OR arrays whose
      compressed size estimate is <= max_mb (very rough heuristic).
    """
    if not light:
        return save_npz(arrs, path)

    keys_keep = set() if keys_keep is None else set(keys_keep)
    slim: Dict[str, np.ndarray] = {}
    budget = max_mb * 1024**2
    for k, v in arrs.items():
        if k in keys_keep:
            slim[k] = v
            continue
        approx = v.size * v.dtype.itemsize
        if approx <= budget:
            slim[k] = v
    if slim:
        save_npz(slim, path)


# -------------------------- versions / manifest -----------------------

def _get_git_commit() -> Optional[str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit
    except Exception:
        return None


def capture_versions(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Capture library versions & environment for reproducibility."""
    ver = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", None),
        "matplotlib": _safe_pkg_version("matplotlib"),
        "scipy": _safe_pkg_version("scipy"),
        "jax": _safe_pkg_version("jax"),
        "git_commit": _get_git_commit(),
    }
    if extra:
        ver.update(extra)
    return ver


def _safe_pkg_version(name: str) -> Optional[str]:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def summarize_array(a: np.ndarray, max_bins: int = 50) -> Dict[str, Any]:
    """Lightweight array summary for manifests."""
    info: Dict[str, Any] = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "size": int(a.size),
    }
    if a.size == 0:
        return info
    try:
        info.update(
            dict(
                mean=float(np.mean(a)),
                std=float(np.std(a)),
                min=float(np.min(a)),
                max=float(np.max(a)),
                frac_zeros=float(np.mean(a == 0)),
                l2=float(np.linalg.norm(a)),
            )
        )
    except Exception:
        pass
    # small histogram for rough distribution
    try:
        counts, edges = np.histogram(a.ravel(), bins=min(max_bins, max(5, int(np.sqrt(a.size)))))
        info["hist_counts"] = counts.tolist()
        info["hist_edges"] = edges.tolist()
    except Exception:
        pass
    return info


def save_manifest(manifest: Dict[str, Any], path: str) -> None:
    """Save a small JSON manifest (config, versions, metrics, shapes…)."""
    save_json(manifest, path)
