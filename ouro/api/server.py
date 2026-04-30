"""
Ouro FastAPI application — multi-model server.

All models listed in ~/.ouro/config.yaml are loaded at startup and served
concurrently on a single port.  Requests are routed to the correct model
via the `model` field in each request body, exactly like Ollama.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover
    raise

try:
    import uvicorn
except ImportError:  # pragma: no cover
    uvicorn = None  # type: ignore

try:
    from ouro.api.routes.models import router as models_router
    from ouro.api.routes.chat import router as chat_router
except ImportError:  # pragma: no cover
    raise

log = logging.getLogger("ouro.server")

# ---------------------------------------------------------------------------
# Global model registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Thread-safe registry of loaded (model, tokenizer) pairs."""

    def __init__(self) -> None:
        self._models: Dict[str, Tuple[Any, Any]] = {}
        self._load_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    def get(self, model_id: str) -> Optional[Tuple[Any, Any]]:
        return self._models.get(model_id)

    def all_ids(self) -> list[str]:
        return list(self._models.keys())

    def load_time(self, model_id: str) -> float:
        return self._load_times.get(model_id, time.time())

    async def load(self, model_id: str, model_path: str) -> None:
        """Load a model (blocking in threadpool so event loop stays free)."""
        async with self._lock:
            if model_id in self._models:
                return  # already loaded

        log.info("Loading model '%s' from '%s' …", model_id, model_path)
        t0 = time.monotonic()

        loop = asyncio.get_event_loop()
        model, tokenizer = await loop.run_in_executor(
            None, _load_model_sync, model_path
        )

        elapsed = time.monotonic() - t0
        log.info("Model '%s' ready in %.1f s", model_id, elapsed)

        async with self._lock:
            self._models[model_id] = (model, tokenizer)
            self._load_times[model_id] = time.time()

    def is_empty(self) -> bool:
        return len(self._models) == 0


def _load_model_sync(model_path: str) -> Tuple[Any, Any]:
    """Synchronous mlx_lm load — called in threadpool."""
    try:
        import mlx_lm  # type: ignore
    except ImportError:
        raise RuntimeError(
            "mlx_lm is not installed.  Run: pip install mlx-lm"
        )
    return mlx_lm.load(model_path)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(registry: Optional[ModelRegistry] = None) -> "FastAPI":
    """
    Create and configure the Ouro FastAPI application.

    Args:
        registry: Pre-built ModelRegistry.  If None a new empty one is created
                  (useful for testing).

    Returns:
        A configured FastAPI application.
    """
    if registry is None:
        registry = ModelRegistry()

    app = FastAPI(
        title="Ouro",
        version="0.1.0",
        description="MLX-native multi-model runner with an OpenAI-compatible REST API.",
    )

    # Store registry on app state so routes can access it via request.app.state
    app.state.registry = registry

    # ── CORS ────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ─────────────────────────────────────────────────────────────
    app.include_router(models_router)  # GET  /v1/models
    app.include_router(chat_router)   # POST /v1/chat/completions

    # ── Built-in endpoints ──────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"name": "ouro", "version": "0.1.0"}

    @app.get("/v1/health")
    async def health(request: Request) -> dict:
        reg: ModelRegistry = request.app.state.registry
        return {
            "status": "ok",
            "models_loaded": reg.all_ids(),
            "count": len(reg.all_ids()),
        }

    return app


# ---------------------------------------------------------------------------
# Startup loader — loads all configured models before accepting requests
# ---------------------------------------------------------------------------

async def load_all_models(app: "FastAPI") -> None:
    """
    Called during FastAPI startup lifespan to load every model listed in
    ~/.ouro/config.yaml → models[].

    Models are loaded concurrently (parallel Metal/MLX loads).
    """
    from ouro.config import get_config
    from ouro.registry.storage import resolve_model_path

    cfg = get_config()
    registry: ModelRegistry = app.state.registry

    model_ids: list[str] = list(cfg.models)
    if not model_ids:
        log.warning(
            "No models configured.  Add a 'models' list to ~/.ouro/config.yaml"
        )
        return

    log.info("Loading %d model(s): %s", len(model_ids), model_ids)

    async def _load_one(model_id: str) -> None:
        try:
            model_path = resolve_model_path(model_id)
        except Exception as exc:
            log.error("Cannot resolve path for '%s': %s", model_id, exc)
            return
        try:
            await registry.load(model_id, model_path)
        except Exception as exc:
            log.error("Failed to load '%s': %s", model_id, exc)

    await asyncio.gather(*[_load_one(mid) for mid in model_ids])
    log.info("All models loaded.  Registry: %s", registry.all_ids())


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def run_server(host: str = "127.0.0.1", port: int = 5215) -> None:
    """
    Build registry, load all models, then start Uvicorn.
    This is the function called by `ouro serve` and by launchd on login.
    """
    if uvicorn is None:
        raise RuntimeError("uvicorn is not installed.  Run: pip install uvicorn")

    from contextlib import asynccontextmanager

    registry = ModelRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[valid-type]
        await load_all_models(app)
        yield
        # Shutdown: nothing special needed — Python GC frees MLX arrays

    app = create_app(registry)
    app.router.lifespan_context = lifespan

    log.info("Ouro starting on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
