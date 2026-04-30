"""
POST /v1/chat/completions  — OpenAI-compatible chat endpoint.
POST /v1/completions       — Legacy completions endpoint.

Model dispatch: routes each request to the correct loaded model via the
`model` field in the request body.  If `model` is omitted or unrecognised,
falls back to the first loaded model (mirrors Ollama behaviour).
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import AsyncGenerator, Tuple, Any

try:
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import StreamingResponse
except ImportError:  # pragma: no cover
    raise

try:
    from ouro.api.schemas import (
        ChatCompletionChunk,
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatMessage,
        Choice,
        CompletionRequest,
        Delta,
        DeltaFunctionCall,
        DeltaToolCall,
        FunctionCall,
        StreamChoice,
        ToolCall,
        Usage,
    )
except ImportError:  # pragma: no cover
    raise

router = APIRouter()

# Regex to strip <think>…</think> blocks from generated text before returning
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Model resolver
# ---------------------------------------------------------------------------

def _resolve_model(request: Request, requested_model: str | None) -> Tuple[Any, Any, str]:
    """
    Resolve (model, tokenizer, model_id) from the multi-model registry.

    Falls back gracefully to legacy single-model state for backward compat.
    Raises HTTP 503 if no model is available.
    """
    # ── New multi-model registry path ────────────────────────────────────────
    try:
        from ouro.api.server import ModelRegistry
        registry: ModelRegistry = request.app.state.registry

        if not registry.is_empty():
            ids = registry.all_ids()

            # Try exact match first
            if requested_model and requested_model in ids:
                model, tokenizer = registry.get(requested_model)
                return model, tokenizer, requested_model

            # Fuzzy: requested name is a suffix of a loaded ID
            if requested_model:
                for mid in ids:
                    if mid.endswith(requested_model) or requested_model in mid:
                        model, tokenizer = registry.get(mid)
                        return model, tokenizer, mid

            # Fallback to first loaded model
            first_id = ids[0]
            model, tokenizer = registry.get(first_id)
            return model, tokenizer, first_id
    except Exception:
        pass

    # ── Legacy single-model fallback ─────────────────────────────────────────
    model = getattr(request.app.state, "model", None)
    tokenizer = getattr(request.app.state, "tokenizer", None)
    model_id = getattr(request.app.state, "model_id", requested_model or "unknown")

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model loaded.  Add models to ~/.ouro/config.yaml under the "
                "'models' key and restart Ouro."
            ),
        )

    return model, tokenizer, model_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _now() -> int:
    return int(time.time())


def _build_prompt(tokenizer: object, messages: list[ChatMessage], tools: list | None) -> str:
    try:
        from ouro.engine import prompt_builder  # type: ignore
        return prompt_builder.build_prompt(tokenizer, messages, tools)
    except Exception:
        parts: list[str] = []
        for msg in messages:
            role = msg.role or "user"
            content = msg.content or ""
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


def _generate(
    model: object, tokenizer: object, prompt: str,
    max_tokens: int, temperature: float, top_p: float,
) -> str:
    try:
        from ouro.engine import generate  # type: ignore
        return generate.generate(model, tokenizer, prompt, max_tokens, temperature, top_p)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc


def _generate_stream(
    model: object, tokenizer: object, prompt: str,
    max_tokens: int, temperature: float, top_p: float,
) -> object:
    try:
        from ouro.engine import generate  # type: ignore
        return generate.generate_stream(model, tokenizer, prompt, max_tokens, temperature, top_p)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stream generation failed: {exc}") from exc


def _parse_tool_calls(text: str) -> list | None:
    try:
        from ouro.engine import tool_parser  # type: ignore
        return tool_parser.parse_tool_calls(text) or None
    except Exception:
        return None


def _raw_to_tool_call(raw: dict, index: int = 0) -> ToolCall:
    fn = raw.get("function", {})
    return ToolCall(
        index=index,
        id=raw.get("id", f"call_{uuid.uuid4().hex[:24]}"),
        type=raw.get("type", "function"),
        function=FunctionCall(
            name=fn.get("name", ""),
            arguments=fn.get("arguments", "{}"),
        ),
    )


def _raw_to_delta_tool_call(raw: dict, index: int = 0) -> DeltaToolCall:
    fn = raw.get("function", {})
    return DeltaToolCall(
        index=index,
        id=raw.get("id", f"call_{uuid.uuid4().hex[:24]}"),
        type=raw.get("type", "function"),
        function=DeltaFunctionCall(
            name=fn.get("name", ""),
            arguments=fn.get("arguments", "{}"),
        ),
    )


def _sse(data: str) -> str:
    return f"data: {data}\n\n"


# ---------------------------------------------------------------------------
# Non-streaming chat
# ---------------------------------------------------------------------------

def _chat_non_stream(
    request_data: ChatCompletionRequest,
    model: object, tokenizer: object, model_id: str, prompt: str,
) -> ChatCompletionResponse:
    raw_text = _generate(
        model, tokenizer, prompt,
        request_data.max_tokens, request_data.temperature, request_data.top_p,
    )
    text = _strip_thinking(raw_text)
    raw_tool_calls = _parse_tool_calls(text)

    if raw_tool_calls:
        typed_calls = [_raw_to_tool_call(tc, i) for i, tc in enumerate(raw_tool_calls)]
        message = ChatMessage(role="assistant", content=None, tool_calls=typed_calls)
        finish_reason = "tool_calls"
    else:
        message = ChatMessage(role="assistant", content=text)
        finish_reason = "stop"

    return ChatCompletionResponse(
        id=_completion_id(),
        object="chat.completion",
        created=_now(),
        model=model_id,
        choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
        usage=Usage.from_texts(prompt, text),
        system_fingerprint=None,
    )


# ---------------------------------------------------------------------------
# Streaming chat
# ---------------------------------------------------------------------------

async def _chat_stream_generator(
    request_data: ChatCompletionRequest,
    model: object, tokenizer: object, model_id: str, prompt: str,
) -> AsyncGenerator[str, None]:
    completion_id = _completion_id()
    created = _now()

    yield _sse(ChatCompletionChunk(
        id=completion_id, object="chat.completion.chunk", created=created, model=model_id,
        choices=[StreamChoice(index=0, delta=Delta(role="assistant"), finish_reason=None)],
    ).model_dump_json(exclude_none=True))

    token_stream = _generate_stream(
        model, tokenizer, prompt,
        request_data.max_tokens, request_data.temperature, request_data.top_p,
    )

    full_text = ""
    try:
        for token in token_stream:
            full_text += token
    except Exception as exc:
        yield _sse(json.dumps({"error": {"message": str(exc), "type": "server_error"}}))
        yield "data: [DONE]\n\n"
        return

    clean_text = _strip_thinking(full_text)
    raw_tool_calls = _parse_tool_calls(clean_text)

    if raw_tool_calls:
        delta_tool_calls = [_raw_to_delta_tool_call(tc, i) for i, tc in enumerate(raw_tool_calls)]
        yield _sse(ChatCompletionChunk(
            id=completion_id, object="chat.completion.chunk", created=created, model=model_id,
            choices=[StreamChoice(
                index=0,
                delta=Delta(role="assistant", content=None, tool_calls=delta_tool_calls),
                finish_reason="tool_calls",
            )],
        ).model_dump_json(exclude_none=True))
    else:
        yield _sse(ChatCompletionChunk(
            id=completion_id, object="chat.completion.chunk", created=created, model=model_id,
            choices=[StreamChoice(index=0, delta=Delta(content=clean_text), finish_reason=None)],
        ).model_dump_json(exclude_none=True))

        yield _sse(ChatCompletionChunk(
            id=completion_id, object="chat.completion.chunk", created=created, model=model_id,
            choices=[StreamChoice(index=0, delta=Delta(), finish_reason="stop")],
        ).model_dump_json(exclude_none=True))

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.post("/v1/chat/completions")
async def chat_completions(request_data: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completions — dispatches to the requested model."""
    model, tokenizer, model_id = _resolve_model(request, request_data.model)
    prompt = _build_prompt(tokenizer, request_data.messages, request_data.tools)

    if request_data.stream:
        return StreamingResponse(
            _chat_stream_generator(request_data, model, tokenizer, model_id, prompt),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"},
        )
    return _chat_non_stream(request_data, model, tokenizer, model_id, prompt)


@router.post("/v1/completions")
async def legacy_completions(request_data: CompletionRequest, request: Request):
    """Legacy /v1/completions — wraps prompt as a user chat message."""
    model, tokenizer, model_id = _resolve_model(request, request_data.model)
    messages = [ChatMessage(role="user", content=request_data.prompt)]
    prompt = _build_prompt(tokenizer, messages, None)

    chat_req = ChatCompletionRequest(
        model=request_data.model,
        messages=messages,
        stream=request_data.stream,
        temperature=request_data.temperature,
        max_tokens=request_data.max_tokens,
        top_p=request_data.top_p,
    )

    if request_data.stream:
        return StreamingResponse(
            _chat_stream_generator(chat_req, model, tokenizer, model_id, prompt),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"},
        )

    text = _strip_thinking(_generate(
        model, tokenizer, prompt,
        request_data.max_tokens, request_data.temperature, request_data.top_p,
    ))
    return {
        "id": _completion_id(), "object": "text_completion",
        "created": _now(), "model": model_id,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"}],
        "usage": Usage.from_texts(prompt, text).model_dump(),
    }
