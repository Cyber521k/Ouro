"""
POST /v1/chat/completions  — OpenAI-compatible chat endpoint.
POST /v1/completions       — Legacy completions endpoint.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import AsyncGenerator

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

# Regex to strip <think>…</think> blocks from generated text before returning via API
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> reasoning blocks from model output for API responses."""
    return _THINK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _now() -> int:
    return int(time.time())


def _build_prompt(tokenizer: object, messages: list[ChatMessage], tools: list | None) -> str:
    """Delegate to engine.prompt_builder if available, else fall back to a simple join."""
    try:
        from ouro.engine import prompt_builder  # type: ignore
        return prompt_builder.build_prompt(tokenizer, messages, tools)
    except Exception:
        # Fallback: concatenate messages as plain text
        parts: list[str] = []
        for msg in messages:
            role = msg.role or "user"
            content = msg.content or ""
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


def _generate(model: object, tokenizer: object, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    """Call engine.generate.generate()."""
    try:
        from ouro.engine import generate  # type: ignore
        return generate.generate(model, tokenizer, prompt, max_tokens, temperature, top_p)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc


def _generate_stream(
    model: object,
    tokenizer: object,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> object:
    """Call engine.generate.generate_stream(); returns an iterable of str tokens."""
    try:
        from ouro.engine import generate  # type: ignore
        return generate.generate_stream(model, tokenizer, prompt, max_tokens, temperature, top_p)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stream generation failed: {exc}") from exc


def _parse_tool_calls(text: str) -> list | None:
    """Try to parse tool calls from generated text."""
    try:
        from ouro.engine import tool_parser  # type: ignore
        return tool_parser.parse_tool_calls(text) or None
    except Exception:
        return None


def _raw_to_tool_call(raw: dict, index: int = 0) -> ToolCall:
    """Convert a raw tool_parser dict to a typed ToolCall schema object."""
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
    """Convert a raw tool_parser dict to a DeltaToolCall for streaming."""
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
# /v1/chat/completions — non-streaming
# ---------------------------------------------------------------------------

def _chat_non_stream(
    request_data: ChatCompletionRequest,
    model: object,
    tokenizer: object,
    model_id: str,
    prompt: str,
) -> ChatCompletionResponse:
    raw_text = _generate(
        model,
        tokenizer,
        prompt,
        request_data.max_tokens,
        request_data.temperature,
        request_data.top_p,
    )
    text = _strip_thinking(raw_text)

    raw_tool_calls = _parse_tool_calls(text)

    if raw_tool_calls:
        # Convert raw dicts → typed ToolCall objects with proper index
        typed_calls = [_raw_to_tool_call(tc, i) for i, tc in enumerate(raw_tool_calls)]
        message = ChatMessage(role="assistant", content=None, tool_calls=typed_calls)
        finish_reason = "tool_calls"
    else:
        message = ChatMessage(role="assistant", content=text)
        finish_reason = "stop"

    usage = Usage.from_texts(prompt, text)

    return ChatCompletionResponse(
        id=_completion_id(),
        object="chat.completion",
        created=_now(),
        model=model_id,
        choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
        usage=usage,
        system_fingerprint=None,
    )


# ---------------------------------------------------------------------------
# /v1/chat/completions — streaming
# ---------------------------------------------------------------------------

async def _chat_stream_generator(
    request_data: ChatCompletionRequest,
    model: object,
    tokenizer: object,
    model_id: str,
    prompt: str,
) -> AsyncGenerator[str, None]:
    completion_id = _completion_id()
    created = _now()

    # First chunk: role announcement
    first_chunk = ChatCompletionChunk(
        id=completion_id,
        object="chat.completion.chunk",
        created=created,
        model=model_id,
        choices=[StreamChoice(index=0, delta=Delta(role="assistant"), finish_reason=None)],
    )
    yield _sse(first_chunk.model_dump_json(exclude_none=True))

    # Stream tokens — buffer full output so we can detect tool calls
    token_stream = _generate_stream(
        model,
        tokenizer,
        prompt,
        request_data.max_tokens,
        request_data.temperature,
        request_data.top_p,
    )

    full_text = ""
    try:
        for token in token_stream:
            full_text += token
    except Exception as exc:
        error_payload = json.dumps({"error": {"message": str(exc), "type": "server_error"}})
        yield _sse(error_payload)
        yield "data: [DONE]\n\n"
        return

    # Strip thinking block, then detect tool calls
    clean_text = _strip_thinking(full_text)
    raw_tool_calls = _parse_tool_calls(clean_text)

    if raw_tool_calls:
        # Tool call path: send structured chunk with index-wrapped DeltaToolCall list
        delta_tool_calls = [_raw_to_delta_tool_call(tc, i) for i, tc in enumerate(raw_tool_calls)]
        tool_chunk = ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=model_id,
            choices=[
                StreamChoice(
                    index=0,
                    delta=Delta(role="assistant", content=None, tool_calls=delta_tool_calls),
                    finish_reason="tool_calls",
                )
            ],
        )
        yield _sse(tool_chunk.model_dump_json(exclude_none=True))
    else:
        # Normal text path: send content then stop
        content_chunk = ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=model_id,
            choices=[StreamChoice(index=0, delta=Delta(content=clean_text), finish_reason=None)],
        )
        yield _sse(content_chunk.model_dump_json(exclude_none=True))

        # Stop chunk
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=model_id,
            choices=[StreamChoice(index=0, delta=Delta(), finish_reason="stop")],
        )
        yield _sse(final_chunk.model_dump_json(exclude_none=True))

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.post("/v1/chat/completions")
async def chat_completions(request_data: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completions endpoint."""
    app_state = request.app.state
    model = getattr(app_state, "model", None)
    tokenizer = getattr(app_state, "tokenizer", None)
    model_id: str = getattr(app_state, "model_id", request_data.model)

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="No model loaded. Start Ouro with a model first.")

    prompt = _build_prompt(tokenizer, request_data.messages, request_data.tools)

    if request_data.stream:
        return StreamingResponse(
            _chat_stream_generator(request_data, model, tokenizer, model_id, prompt),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return _chat_non_stream(request_data, model, tokenizer, model_id, prompt)


# ---------------------------------------------------------------------------
# /v1/completions — legacy endpoint
# ---------------------------------------------------------------------------

@router.post("/v1/completions")
async def legacy_completions(request_data: CompletionRequest, request: Request):
    """Legacy /v1/completions endpoint; wraps the prompt as a user chat message."""
    app_state = request.app.state
    model = getattr(app_state, "model", None)
    tokenizer = getattr(app_state, "tokenizer", None)
    model_id: str = getattr(app_state, "model_id", request_data.model)

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="No model loaded. Start Ouro with a model first.")

    messages = [ChatMessage(role="user", content=request_data.prompt)]
    prompt = _build_prompt(tokenizer, messages, None)

    # Synthetic ChatCompletionRequest for shared logic
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
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming: return legacy text field format
    text = _generate(
        model,
        tokenizer,
        prompt,
        request_data.max_tokens,
        request_data.temperature,
        request_data.top_p,
    )
    text = _strip_thinking(text)
    usage = Usage.from_texts(prompt, text)
    completion_id = _completion_id()

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": _now(),
        "model": model_id,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": usage.model_dump(),
    }
