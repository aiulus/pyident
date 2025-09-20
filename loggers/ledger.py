from __future__ import annotations
from datetime import datetime, UTC
from typing import Any, Dict

from .runtime_banner import runtime_banner

def start_ledger() -> Dict[str, Any]:
    return {
        "started_utc": datetime.now(UTC).isoformat(),
        "env": runtime_banner(),
        "approximations": [],
        "warnings": [],
        "tolerances": {},
    }

def attach_tolerances(ledger: Dict[str, Any], tol) -> None:
    ledger["tolerances"] = dict(vars(tol))

def log_approx(ledger: Dict[str, Any], kind: str, detail: str) -> None:
    ledger["approximations"].append({"kind": kind, "detail": detail})

def log_warning(ledger: Dict[str, Any], msg: str) -> None:
    ledger["warnings"].append(str(msg))
