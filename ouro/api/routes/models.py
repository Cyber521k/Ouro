"""
GET /v1/models — returns all models currently loaded in the registry.
"""
from __future__ import annotations

import time

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
    """Return all models that are currently loaded and ready to serve."""
    models: list[ModelInfo] = []

    # Primary source: multi-model registry (new path)
    try:
        from ouro.api.server import ModelRegistry
        registry: ModelRegistry = request.app.state.registry
        for model_id in registry.all_ids():
            models.append(
                ModelInfo(
                    id=model_id,
                    created=int(registry.load_time(model_id)),
                    owned_by="ouro",
                )
            )
    except Exception:
        pass

    # Fallback: legacy single-model state (keeps backward compat during transition)
    if not models:
        try:
            loaded_id: str | None = getattr(request.app.state, "model_id", None)
            if loaded_id:
                models.append(
                    ModelInfo(id=loaded_id, created=int(time.time()), owned_by="ouro")
                )
        except Exception:
            pass

    return ModelListResponse(object="list", data=models)
