"""Lightweight, dependency-free model registry.

Records every trained/evaluated model as an entry in ``ml/registry/models.json``
so a deployed verdict can always be traced to a concrete artifact: its version,
file checksum, training config, evaluation metrics and recommended thresholds.

This is a pragmatic stand-in for a full registry (MLflow / W&B). It is a plain
JSON manifest with no external services, which is enough for audit + reproducible
deployment and can be swapped for a hosted registry later.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml.config import REGISTRY_DIR, REGISTRY_MANIFEST


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_registry() -> Dict[str, Any]:
    if REGISTRY_MANIFEST.exists():
        with open(REGISTRY_MANIFEST, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {"models": []}


def save_registry(registry: Dict[str, Any]) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, sort_keys=False)


def register_model(
    version: str,
    weights_path: Path,
    task: str = "pcb_defect_detection",
    metrics: Optional[Dict[str, Any]] = None,
    train_config: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Insert or update a model entry keyed by ``version``."""
    weights_path = Path(weights_path)
    entry = {
        "version": version,
        "path": str(weights_path),
        "task": task,
        "sha256": sha256_of(weights_path) if weights_path.exists() else None,
        "metrics": metrics or {},
        "train_config": train_config or {},
        "thresholds": thresholds or {},
        "notes": notes,
        "registered_at": _now(),
    }

    registry = load_registry()
    models: List[Dict[str, Any]] = registry.setdefault("models", [])
    # Replace an existing entry with the same version, else append.
    for index, existing in enumerate(models):
        if existing.get("version") == version:
            models[index] = entry
            break
    else:
        models.append(entry)

    save_registry(registry)
    return entry


def get_model(version: str) -> Optional[Dict[str, Any]]:
    for entry in load_registry().get("models", []):
        if entry.get("version") == version:
            return entry
    return None


def latest_model() -> Optional[Dict[str, Any]]:
    models = load_registry().get("models", [])
    if not models:
        return None
    return sorted(models, key=lambda m: m.get("registered_at", ""))[-1]
