"""
OpenAI-compatible Pydantic schemas for the Ouro API.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    raise


# ---------------------------------------------------------------------------
# Core message types
# ---------------------------------------------------------------------------

class FunctionCall(BaseModel):
    """Function call object inside a tool_call entry."""
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """Single tool call in OpenAI format."""
    index: int = 0
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    # 'name' is required for tool result messages (role='tool')
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 8192
    top_p: float = 0.9


class CompletionRequest(BaseModel):
    """Legacy /v1/completions endpoint request."""
    model: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


# ---------------------------------------------------------------------------
# Usage / token counting
# ---------------------------------------------------------------------------

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_texts(cls, prompt_text: str, completion_text: str) -> "Usage":
        prompt_tokens = len(prompt_text.split())
        completion_tokens = len(completion_text.split())
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


# ---------------------------------------------------------------------------
# Non-streaming response models
# ---------------------------------------------------------------------------

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str
    # Hermes reads logprobs even if None — include the field
    logprobs: Any | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage
    # Hermes checks this field; always send it
    system_fingerprint: str | None = None


# ---------------------------------------------------------------------------
# Model list response models
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "ouro"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Streaming response models
# ---------------------------------------------------------------------------

class DeltaFunctionCall(BaseModel):
    """Partial function call for streaming deltas."""
    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(BaseModel):
    """Single tool call entry in a streaming delta — must include index."""
    index: int = 0
    id: str | None = None
    type: str | None = "function"
    function: DeltaFunctionCall | None = None


class Delta(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[DeltaToolCall] | None = None


class StreamChoice(BaseModel):
    index: int
    delta: Delta
    finish_reason: str | None = None
    logprobs: Any | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]
    system_fingerprint: str | None = None
