"""
Ouro FastAPI application — lazy on-demand model server.

Models listed in ~/.ouro/config.yaml are NOT loaded at startup.  They are
loaded on the first request that asks for them, then kept hot in RAM.  When
the number of loaded models would exceed ``max_loaded_models`` (default 1),
the least-recently-used model is evicted from RAM first — freeing Metal/
unified memory before loading the new one.

This makes it safe to list many models in config.yaml regardless of how much
RAM you have — only the model(s) you're actively using occupy memory.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# Global model registry — lazy loading + LRU eviction
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Lazy-loading, LRU-evicting registry of (model, tokenizer) pairs.

    Models are loaded on first request.  When ``max_loaded`` is reached,
    the least-recently-used model is evicted (freed from RAM/Metal memory)
    before loading the new one.

    ``max_loaded=0`` means unlimited — keep everything hot (original behaviour).
    """

    def __init__(self, max_loaded: int = 1) -> None:
        # OrderedDict used as an LRU cache: most-recently-used at the end.
        self._loaded: OrderedDict[str, Tuple[Any, Any]] = OrderedDict()
        self._load_times: Dict[str, float] = {}
        # All model IDs known to the server (from config), whether loaded or not.
        self._known: List[str] = []
        self._max_loaded = max_loaded
        # Per-model asyncio.Lock to prevent concurrent loads of the same model.
        self._model_locks: Dict[str, asyncio.Lock] = {}
        # Global lock protecting _loaded / _known mutations.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> Optional[Tuple[Any, Any]]:
        """Return (model, tokenizer) if loaded, else None. Does NOT touch LRU order."""
        return self._loaded.get(model_id)

    def all_ids(self) -> List[str]:
        """IDs of all *loaded* models (what /v1/models should advertise)."""
        return list(self._loaded.keys())

    def known_ids(self) -> List[str]:
        """IDs of all models known from config (loaded or not)."""
        return list(self._known)

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    def load_time(self, model_id: str) -> float:
        return self._load_times.get(model_id, time.time())

    def is_empty(self) -> bool:
        return len(self._loaded) == 0

    def register_known(self, model_ids: List[str]) -> None:
        """Register model IDs from config without loading them."""
        for mid in model_ids:
            if mid not in self._known:
                self._known.append(mid)
            if mid not in self._model_locks:
                self._model_locks[mid] = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lazy load + LRU eviction
    # ------------------------------------------------------------------

    async def ensure_loaded(self, model_id: str, model_path: str) -> Tuple[Any, Any]:
        """
        Ensure *model_id* is loaded and return (model, tokenizer).

        Thread-safe: concurrent calls for the same model_id will coalesce —
        only one actual load happens; the others wait and then get the result.

        If loading would exceed max_loaded, the LRU model is evicted first.
        """
        # Fast path — already loaded, just bump LRU order
        async with self._lock:
            if model_id in self._loaded:
                self._loaded.move_to_end(model_id)
                return self._loaded[model_id]

        # Register a per-model lock on demand (e.g. for models added at runtime)
        async with self._lock:
            if model_id not in self._model_locks:
                self._model_locks[model_id] = asyncio.Lock()

        # Serialize per-model loads so two simultaneous requests don't double-load
        async with self._model_locks[model_id]:
            # Re-check inside per-model lock (another waiter may have loaded it)
            async with self._lock:
                if model_id in self._loaded:
                    self._loaded.move_to_end(model_id)
                    return self._loaded[model_id]

            # Evict LRU if at capacity
            if self._max_loaded > 0:
                async with self._lock:
                    while len(self._loaded) >= self._max_loaded:
                        lru_id, lru_pair = next(iter(self._loaded.items()))
                        log.info(
                            "Evicting LRU model '%s' to free RAM (max_loaded=%d)",
                            lru_id, self._max_loaded,
                        )
                        del self._loaded[lru_id]
                        _unload_model_sync(lru_id, lru_pair)

            # Load the model
            log.info("Lazy-loading model '%s' from '%s' …", model_id, model_path)
            t0 = time.monotonic()
            loop = asyncio.get_event_loop()
            model, tokenizer = await loop.run_in_executor(
                None, _load_model_sync, model_path
            )
            elapsed = time.monotonic() - t0
            log.info("Model '%s' ready in %.1f s", model_id, elapsed)

            async with self._lock:
                self._loaded[model_id] = (model, tokenizer)
                self._loaded.move_to_end(model_id)
                self._load_times[model_id] = time.time()

            return model, tokenizer

    # ------------------------------------------------------------------
    # Legacy compat shim (used in load_all_models + tests)
    # ------------------------------------------------------------------

    async def load(self, model_id: str, model_path: str) -> None:
        """Compat shim — delegates to ensure_loaded."""
        await self.ensure_loaded(model_id, model_path)


def _load_model_sync(model_path: str) -> Tuple[Any, Any]:
    """Synchronous mlx_lm load — called in threadpool."""
    try:
        import mlx_lm  # type: ignore
    except ImportError:
        raise RuntimeError(
            "mlx_lm is not installed.  Run: pip install mlx-lm"
        )
    return mlx_lm.load(model_path)


def _unload_model_sync(model_id: str, pair: Tuple[Any, Any]) -> None:
    """
    Release Metal/MLX memory for an evicted model.

    MLX arrays are freed when there are no more Python references.  We
    explicitly delete the model object and run mx.metal.clear_cache() so the
    GPU allocator returns memory to the system immediately rather than waiting
    for the next GC cycle.
    """
    try:
        import gc
        import mlx.core as mx  # type: ignore
        model, tokenizer = pair
        del model, tokenizer
        gc.collect()
        try:
            mx.metal.clear_cache()
            log.info("Metal cache cleared after evicting '%s'", model_id)
        except Exception:
            pass  # Non-Metal backends (CPU) don't have this
    except Exception as exc:
        log.warning("Error during model eviction for '%s': %s", model_id, exc)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(registry: Optional[ModelRegistry] = None, lifespan=None) -> "FastAPI":
    """
    Create and configure the Ouro FastAPI application.

    Args:
        registry: Pre-built ModelRegistry.  If None a new empty one is created
                  (useful for testing).
        lifespan: Optional asynccontextmanager lifespan function to pass to FastAPI.

    Returns:
        A configured FastAPI application.
    """
    if registry is None:
        from ouro.config import get_config
        cfg = get_config()
        registry = ModelRegistry(max_loaded=cfg.max_loaded_models)

    app = FastAPI(
        title="Ouro",
        version="0.1.0",
        description="MLX-native multi-model runner with an OpenAI-compatible REST API.",
        lifespan=lifespan,
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
            "models_available": reg.known_ids(),
            "loaded_count": len(reg.all_ids()),
            "max_loaded": reg._max_loaded,
        }

    return app


# ---------------------------------------------------------------------------
# Startup loader — loads all configured models before accepting requests
# ---------------------------------------------------------------------------

async def load_all_models(app: "FastAPI") -> None:
    """
    Called during FastAPI startup lifespan.

    In lazy mode (the default), this simply registers all configured model IDs
    so the server knows about them — no actual loading happens until the first
    request for each model arrives.

    This means startup is instant regardless of how many models are listed in
    config.yaml.
    """
    from ouro.config import get_config

    cfg = get_config()
    registry: ModelRegistry = app.state.registry

    model_ids: list[str] = list(cfg.models)
    if not model_ids:
        log.warning(
            "No models configured.  Add a 'models' list to ~/.ouro/config.yaml"
        )
        return

    registry.register_known(model_ids)
    log.info(
        "Registered %d model(s) for lazy loading (max_loaded=%d): %s",
        len(model_ids), cfg.max_loaded_models, model_ids,
    )


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def run_server(host: str = "127.0.0.1", port: int = 5215) -> None:
    """
    Build registry, register configured models, then start Uvicorn.

    Models are NOT loaded at startup — they load on first request (lazy).
    The LRU eviction policy ensures only max_loaded_models are in RAM at once.
    """
    if uvicorn is None:
        raise RuntimeError("uvicorn is not installed.  Run: pip install uvicorn")

    from contextlib import asynccontextmanager
    from ouro.config import get_config

    cfg = get_config()
    registry = ModelRegistry(max_loaded=cfg.max_loaded_models)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[valid-type]
        await load_all_models(app)
        yield
        # Shutdown: nothing special needed — Python GC frees MLX arrays

    app = create_app(registry, lifespan=lifespan)

    log.info("Ouro starting on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
