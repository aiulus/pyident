# pyident/io_utils.py
from __future__ import annotations
import json, csv, os
import numpy as np
from typing import Dict, Any, Optional

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def save_npz(arrs: Dict[str, np.ndarray], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **arrs)

def save_csv(rows: list[dict], path: str, fieldnames: Optional[list[str]] = None) -> None:
    ensure_dir(os.path.dirname(path))
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
