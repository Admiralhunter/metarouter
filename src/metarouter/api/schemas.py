"""OpenAI-compatible API schemas."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single chat message."""

    role: str
    content: str | list[dict]
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Request for chat completion (OpenAI-compatible)."""

    model: Optional[str] = None  # Ignored, router selects model
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[str | list[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Response for chat completion."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information (OpenAI-compatible)."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "lm-studio"


class ModelsListResponse(BaseModel):
    """List of available models."""

    object: str = "list"
    data: list[ModelInfo]


class ErrorResponse(BaseModel):
    """Error response."""

    error: dict[str, Any]
