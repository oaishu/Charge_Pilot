"""Canonical offline-side loader shim (plan layout: ``ml/load_artifacts.py``).

The real serving loader lives in the backend package
(``app.core.ml_artifacts``) so it can read backend settings directly. This
shim makes the plan's documented import path work from the ml/ side (e.g. for
post-training promotion tooling run outside the backend process).

Usage:
    from ml.load_artifacts import load_forecast_artifact   # requires repo-root CWD
"""

from __future__ import annotations

from pathlib import Path

import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = _REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.core.ml_artifacts import (  # noqa: E402,F401
    clear_cache,
    load_fault_artifact,
    load_forecast_artifact,
    staged_versions,
)
