"""
Ouro FastAPI application factory and server runner.
"""
from __future__ import annotations

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
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


def create_app(model: object, tokenizer: object, model_id: str) -> "FastAPI":
    """
    Create and configure the Ouro FastAPI application.

    Args:
        model:      The loaded MLX model instance.
        tokenizer:  The associated tokenizer instance.
        model_id:   Human-readable identifier for the loaded model.

    Returns:
        A configured FastAPI application ready to be served.
    """
    app = FastAPI(
        title="Ouro",
        version="0.1.0",
        description="MLX-native model runner with an OpenAI-compatible REST API.",
    )

    # --- Store shared state --------------------------------------------------
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.model_id = model_id

    # --- CORS (allow all origins for local development) ----------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routers -------------------------------------------------------------
    app.include_router(models_router)   # GET  /v1/models
    app.include_router(chat_router)     # POST /v1/chat/completions
                                        # POST /v1/completions

    # --- Built-in endpoints --------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"name": "ouro", "version": "0.1.0"}

    @app.get("/v1/health")
    async def health(request: Request) -> dict:
        loaded: str = getattr(request.app.state, "model_id", "none")
        return {"status": "ok", "model": loaded}

    return app


def run_server(app: "FastAPI", host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Start the Uvicorn server synchronously.

    Args:
        app:  A FastAPI application instance (from :func:`create_app`).
        host: Bind host address.
        port: Bind port number.
    """
    if uvicorn is None:
        raise RuntimeError(
            "uvicorn is not installed. Run `pip install uvicorn` to start the server."
        )
    uvicorn.run(app, host=host, port=port, log_level="info")
