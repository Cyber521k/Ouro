"""
GET /v1/models — returns all installed models known to the registry.
"""
from __future__ import annotations

import time
from datetime import datetime

try:
    from fastapi import APIRouter, Request
except ImportError:  # pragma: no cover
    raise

try:
    from ouro.api.schemas import ModelInfo, ModelListResponse
except ImportError:  # pragma: no cover
    raise

router = APIRouter()


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    """Return all installed models plus the currently loaded model (if any)."""
    models: list[ModelInfo] = []
    seen_ids: set[str] = set()

    # --- Installed models from the registry ---
    try:
        from ouro.registry import storage as registry_storage  # type: ignore

        installed = registry_storage.list_installed_models()
        for entry in installed:
            # entry is a plain dict with keys: id, path, size_mb, modified (ISO string)
            model_id = entry["id"]
            try:
                created = int(datetime.fromisoformat(entry["modified"]).timestamp())
            except Exception:
                created = int(time.time())

            if model_id not in seen_ids:
                models.append(
                    ModelInfo(id=model_id, created=created, owned_by="ouro")
                )
                seen_ids.add(model_id)
    except Exception:
        # Registry not available — continue gracefully
        pass

    # --- Currently loaded model from app state ---
    try:
        loaded_id: str | None = getattr(request.app.state, "model_id", None)
        if loaded_id and loaded_id not in seen_ids:
            models.append(
                ModelInfo(id=loaded_id, created=int(time.time()), owned_by="ouro")
            )
            seen_ids.add(loaded_id)
    except Exception:
        pass

    return ModelListResponse(object="list", data=models)
