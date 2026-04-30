"""
POST /v1/chat/completions  — OpenAI-compatible chat endpoint.
POST /v1/completions       — Legacy completions endpoint.

Model dispatch: routes each request to the correct loaded model via the
`model` field in the request body.  If `model` is omitted or unrecognised,
falls back to the first loaded model (mirrors Ollama behaviour).
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import AsyncGenerator, Tuple, Any

log = logging.getLogger("ouro.chat")

try:
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
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

# Regex to strip <think>…</think> blocks from generated text before returning.
# Two patterns handled:
#   1. Proper:  <think>…</think>  → strip whole block
#   2. Orphan:  …</think>         → model emitted thinking without opening tag;
#               strip everything up to and including the closing tag.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ORPHAN_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    # Pass 1: strip properly-wrapped <think>…</think> blocks
    text = _THINK_RE.sub("", text)
    # Pass 2: strip orphaned …</think> prefix (no opening tag)
    if "</think>" in text:
        text = _ORPHAN_THINK_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Model resolver
# ---------------------------------------------------------------------------

async def _resolve_model(request: Request, requested_model: str | None) -> Tuple[Any, Any, str]:
    """
    Resolve (model, tokenizer, model_id) from the multi-model registry.

    Uses lazy loading: if the model is registered but not yet loaded into RAM,
    it is loaded on first request (and the LRU model is evicted if needed).

    Falls back gracefully to legacy single-model state for backward compat.
    Raises HTTP 503 if no model is available.
    """
    # ── New multi-model registry path ────────────────────────────────────────
    try:
        from ouro.api.server import ModelRegistry
        registry: ModelRegistry = request.app.state.registry

        # Use known_ids() so we can trigger lazy loading for registered-but-not-yet-loaded models
        known = registry.known_ids()

        if known:
            # Resolve which model ID to use
            target_id: str | None = None

            if requested_model:
                # Exact match first
                if requested_model in known:
                    target_id = requested_model
                else:
                    # Suffix / substring match
                    for mid in known:
                        if mid.endswith(requested_model) or requested_model in mid:
                            target_id = mid
                            break

            if target_id is None:
                # Fallback: use first known model
                target_id = known[0]

            # Resolve model path via storage
            from ouro.registry.storage import resolve_model_path
            model_path = resolve_model_path(target_id)

            if model_path is None:
                # Model not downloaded yet — tell the user clearly
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Model '{target_id}' is registered but not downloaded. "
                        f"Run: ouro pull {target_id}"
                    ),
                )

            # ensure_loaded triggers lazy load (or returns cached) + LRU eviction
            model, tokenizer = await registry.ensure_loaded(target_id, str(model_path))
            return model, tokenizer, target_id

    except HTTPException:
        raise  # don't swallow our own HTTP errors
    except Exception as exc:
        log.warning("Multi-model registry path failed, trying legacy fallback: %s", exc)

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

    # ---------------------------------------------------------------------------
    # Streaming state machine with tool-call detection
    #
    # The challenge: Qwen3/Hermes models emit tool calls as XML text
    # (e.g. <tool_call><function=name>...</function></tool_call>), which must be
    # intercepted and converted to proper OpenAI tool_calls format.  If we stream
    # the raw XML as content, Hermes Agent sees garbage text followed by tool_calls
    # and gets confused (empty response error).
    #
    # Solution: buffer ALL output.  Once generation finishes:
    #   - If tool calls are detected → emit them in proper incremental format
    #   - If no tool calls → emit the buffered text as content
    #
    # For latency on non-tool responses: we use a hybrid approach.  We buffer
    # tokens until we can determine whether the output is a tool call.  If the
    # first meaningful tokens indicate a tool call (<tool_call>, or starts with
    # { after think stripping), we buffer everything.  Otherwise we flush and
    # stream normally.
    #
    # States:
    #   "waiting"     — accumulating initial tokens to decide what kind of output
    #   "thinking"    — inside a <think> block; buffer silently
    #   "buffering"   — detected likely tool call; buffer everything silently
    #   "streaming"   — confirmed non-tool output; stream tokens directly
    # ---------------------------------------------------------------------------
    full_text = ""
    state = "waiting"
    buffer = ""

    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"
    TOOL_CALL_OPEN = "<tool_call>"

    # How many characters to buffer before deciding (accounts for <think> + <tool_call>)
    DECISION_THRESHOLD = 20

    try:
        for token in token_stream:
            full_text += token

            if state == "streaming":
                # Fast path — confirmed non-tool output, stream directly
                yield _sse(ChatCompletionChunk(
                    id=completion_id, object="chat.completion.chunk",
                    created=created, model=model_id,
                    choices=[StreamChoice(index=0, delta=Delta(content=token), finish_reason=None)],
                ).model_dump_json(exclude_none=True))

            elif state == "buffering":
                # Silently accumulate — will be emitted as tool_calls at the end
                buffer += token

            elif state == "thinking":
                buffer += token
                if THINK_CLOSE in buffer:
                    # Exited thinking block — check what comes after
                    after = buffer.split(THINK_CLOSE, 1)[1].lstrip("\n")
                    buffer = after
                    # Now decide: is the post-think content a tool call?
                    if buffer.lstrip().startswith(TOOL_CALL_OPEN):
                        state = "buffering"
                    elif len(buffer) >= DECISION_THRESHOLD:
                        # Enough to decide — check for tool call indicators
                        stripped = buffer.lstrip()
                        if stripped.startswith(TOOL_CALL_OPEN) or stripped.startswith("{"):
                            state = "buffering"
                        else:
                            state = "streaming"
                            if buffer:
                                yield _sse(ChatCompletionChunk(
                                    id=completion_id, object="chat.completion.chunk",
                                    created=created, model=model_id,
                                    choices=[StreamChoice(index=0, delta=Delta(content=buffer), finish_reason=None)],
                                ).model_dump_json(exclude_none=True))
                                buffer = ""
                    # else: keep buffering in thinking-exit state (will be caught below)

            else:  # state == "waiting"
                buffer += token
                stripped = buffer.lstrip()

                if stripped.startswith(THINK_OPEN):
                    # Confirmed thinking block
                    state = "thinking"
                elif stripped.startswith(TOOL_CALL_OPEN):
                    # Tool call detected immediately — buffer everything
                    state = "buffering"
                elif stripped.startswith("{"):
                    # Possible bare JSON tool call — buffer to confirm
                    state = "buffering"
                elif len(buffer) >= DECISION_THRESHOLD:
                    # Enough tokens, no tool call indicators — stream normally
                    state = "streaming"
                    yield _sse(ChatCompletionChunk(
                        id=completion_id, object="chat.completion.chunk",
                        created=created, model=model_id,
                        choices=[StreamChoice(index=0, delta=Delta(content=buffer), finish_reason=None)],
                    ).model_dump_json(exclude_none=True))
                    buffer = ""
                # else: keep waiting for more tokens

    except Exception as exc:
        yield _sse(json.dumps({"error": {"message": str(exc), "type": "server_error"}}))
        yield "data: [DONE]\n\n"
        return

    # ---------------------------------------------------------------------------
    # Generation complete — decide how to emit the response
    # ---------------------------------------------------------------------------

    # Strip thinking from the full accumulated text
    clean_full = _strip_thinking(full_text)

    # Check for tool calls
    raw_tool_calls = _parse_tool_calls(clean_full)

    if raw_tool_calls:
        # Emit tool calls in proper OpenAI streaming format:
        # 1. One chunk per tool call with id, type, function.name, and full arguments
        # 2. Final chunk with finish_reason="tool_calls"
        #
        # Note: we emit each tool call's arguments in full (not character-by-character)
        # because Hermes Agent accumulates them anyway and partial arguments can cause
        # JSON parse errors on the client side.
        for i, tc in enumerate(raw_tool_calls):
            delta_tc = _raw_to_delta_tool_call(tc, i)
            yield _sse(ChatCompletionChunk(
                id=completion_id, object="chat.completion.chunk",
                created=created, model=model_id,
                choices=[StreamChoice(
                    index=0,
                    delta=Delta(tool_calls=[delta_tc]),
                    finish_reason=None,
                )],
            ).model_dump_json(exclude_none=True))

        # Final chunk: empty delta + finish_reason
        yield _sse(ChatCompletionChunk(
            id=completion_id, object="chat.completion.chunk",
            created=created, model=model_id,
            choices=[StreamChoice(index=0, delta=Delta(), finish_reason="tool_calls")],
        ).model_dump_json(exclude_none=True))

    else:
        # No tool calls — emit any remaining buffered text as content
        if state != "streaming" and buffer:
            # We were still buffering (e.g. short response that looked like it might
            # be a tool call but wasn't, or thinking-only response)
            clean_text = _strip_thinking(buffer) if state == "thinking" else buffer
            if clean_text:
                yield _sse(ChatCompletionChunk(
                    id=completion_id, object="chat.completion.chunk",
                    created=created, model=model_id,
                    choices=[StreamChoice(index=0, delta=Delta(content=clean_text), finish_reason=None)],
                ).model_dump_json(exclude_none=True))
        elif state == "buffering":
            # Was buffering because it looked like a tool call but parsing found nothing
            # (e.g. model emitted { but it wasn't valid JSON tool call)
            clean_text = _strip_thinking(full_text)
            if clean_text:
                yield _sse(ChatCompletionChunk(
                    id=completion_id, object="chat.completion.chunk",
                    created=created, model=model_id,
                    choices=[StreamChoice(index=0, delta=Delta(content=clean_text), finish_reason=None)],
                ).model_dump_json(exclude_none=True))

        # Finish with stop
        yield _sse(ChatCompletionChunk(
            id=completion_id, object="chat.completion.chunk",
            created=created, model=model_id,
            choices=[StreamChoice(index=0, delta=Delta(), finish_reason="stop")],
        ).model_dump_json(exclude_none=True))

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.post("/v1/chat/completions")
async def chat_completions(request_data: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completions — dispatches to the requested model."""
    model, tokenizer, model_id = await _resolve_model(request, request_data.model)
    prompt = _build_prompt(tokenizer, request_data.messages, request_data.tools)

    if request_data.stream:
        return StreamingResponse(
            _chat_stream_generator(request_data, model, tokenizer, model_id, prompt),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"},
        )
    response = _chat_non_stream(request_data, model, tokenizer, model_id, prompt)
    # Return JSON with null fields excluded — Hermes Agent and other OpenAI clients
    # expect clean responses without null tool_calls/name/tool_call_id fields.
    return JSONResponse(content=json.loads(response.model_dump_json(exclude_none=True)))


@router.post("/v1/completions")
async def legacy_completions(request_data: CompletionRequest, request: Request):
    """Legacy /v1/completions — wraps prompt as a user chat message."""
    model, tokenizer, model_id = await _resolve_model(request, request_data.model)
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
