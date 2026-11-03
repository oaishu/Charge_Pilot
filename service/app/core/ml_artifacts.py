"""Load serving artifacts for the standalone ML service (``ml/service``).

Same contract as the backend loader (``backend/app/core/ml_artifacts.py``),
simplified paths resolution for the ML codebase layout:

- ``ml/artifacts/forecast_model_v1.pkl`` / ``fault_model_v1.pkl`` are the
  bundles written by notebooks 02/03.
- Returns ``None`` (never raises) when an artifact is absent/malformed —
  services fall back to their deterministic baselines;
- refuses to stage a forecast artifact that does not beat the baseline
  (``bundle["beats_baseline"]`` flag written by notebook 02);
- caches loaded bundles once per process;
- import-light: joblib/sklearn only load when a bundle actually exists.

Path resolution (first match wins):
1. explicit ``path`` argument;
2. ``settings.ML_ARTIFACTS_DIR`` as given;
3. this file's ancestors: ``ml/service/app/core`` → up to ``ml/`` →
   ``ml/artifacts`` (the canonical location, works regardless of CWD);
4. CWD-anchored ``ml/artifacts`` walk (covers other launch styles).
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("ml.service")

FORECAST_ARTIFACT = "forecast_model_v1.pkl"
FAULT_ARTIFACT = "fault_model_v1.pkl"

_cache: dict[str, Optional[Any]] = {}
_lock = threading.Lock()


def _artifact_dir_candidates(explicit: Optional[Path] = None) -> list[Path]:
    from app.core.config import settings

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(Path(os.path.abspath(settings.ML_ARTIFACTS_DIR)))
    # canonical: walk up from this file to the ml/ package root
    here = Path(__file__).resolve()
    for base in here.parents:
        candidates.append(base / "artifacts")
    # CWD-anchored fallbacks
    for base in (Path(os.getcwd()), *here.parents):
        candidates.append(base / "ml" / "artifacts")
    return candidates


def _artifact_path(name: str, explicit_dir: Optional[Path] = None) -> Optional[Path]:
    for candidate in _artifact_dir_candidates(explicit_dir):
        path = candidate / name
        if path.is_file():
            return path
    return None


def _load_bundle(name: str, explicit_dir: Optional[Path] = None) -> Optional[dict]:
    path = _artifact_path(name, explicit_dir)
    if path is None:
        return None
    try:
        import joblib

        bundle = joblib.load(path)
        if not isinstance(bundle, dict):
            raise TypeError(f"artifact is not a dict: {type(bundle)!r}")
        return bundle
    except Exception:  # noqa: BLE001 - serving must never crash on artifact issues
        logger.exception("failed to load ML artifact %s", path)
        return None


def load_forecast_artifact(explicit_dir: Optional[Path] = None) -> Optional[dict]:
    """Load the demand-forecast bundle, or None when unavailable/demoted."""
    with _lock:
        if "forecast" in _cache:
            return _cache["forecast"]
    bundle = _load_bundle(FORECAST_ARTIFACT, explicit_dir)
    if bundle is not None:
        if not bundle.get("beats_baseline", False):
            logger.info(
                "forecast artifact %s demoted: does not beat baseline",
                bundle.get("model_version"),
            )
            bundle = None
        elif "model_daily_sessions" not in bundle or "model_hourly_demand" not in bundle:
            logger.warning("forecast artifact missing expected models - ignored")
            bundle = None
    with _lock:
        _cache["forecast"] = bundle
    return bundle


def load_fault_artifact(explicit_dir: Optional[Path] = None) -> Optional[dict]:
    """Load the fault-detection bundle (primary classifier + fallback), or None."""
    with _lock:
        if "fault" in _cache:
            return _cache["fault"]
    bundle = _load_bundle(FAULT_ARTIFACT, explicit_dir)
    if bundle is not None and ("primary" not in bundle or "feature_names" not in bundle):
        logger.warning("fault artifact missing expected members - ignored")
        bundle = None
    with _lock:
        _cache["fault"] = bundle
    return bundle


def clear_cache() -> None:
    """Drop cached bundles (used when retraining hot-swaps the .pkl)."""
    with _lock:
        _cache.clear()


def staged_versions() -> dict[str, Optional[str]]:
    """Model versions currently staged, per artifact kind.

    Falls back to the config-version placeholders for the kind whose artifact
    isn't staged (i.e. services are on their deterministic baselines).
    """
    from app.core.config import settings

    forecast = load_forecast_artifact() or {}
    fault = load_fault_artifact() or {}
    return {
        "forecast": forecast.get("model_version", settings.ML_FORECAST_MODEL_VERSION),
        "fault": fault.get("model_version", settings.ML_FAULT_MODEL_VERSION),
    }
