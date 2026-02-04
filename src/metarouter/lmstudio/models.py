"""Data models for LM Studio API responses."""

from typing import Optional

from pydantic import BaseModel, Field


class Quantization(BaseModel):
    """Model quantization info."""

    name: str
    bits_per_weight: float


class Capabilities(BaseModel):
    """Model capabilities."""

    vision: bool = False
    trained_for_tool_use: bool = False


class LoadedInstance(BaseModel):
    """Information about a loaded model instance."""

    context_length: Optional[int] = None
    eval_batch_size: Optional[int] = None
    flash_attention: Optional[bool] = None


class ModelInfo(BaseModel):
    """Model information from LM Studio API."""

    id: str
    type: str  # "llm" or "embedding"
    publisher: Optional[str] = None
    display_name: Optional[str] = None
    architecture: Optional[str] = None
    quantization: Optional[Quantization] = None
    size_bytes: Optional[int] = None
    params_string: Optional[str] = None  # e.g., "7B", "13B"
    max_context_length: Optional[int] = None
    format: Optional[str] = None  # "gguf", "mlx", etc.
    capabilities: Optional[Capabilities] = None
    description: Optional[str] = None
    state: Optional[str] = None  # "loaded" or not present
    loaded_instances: list[LoadedInstance] = Field(default_factory=list)

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self.state == "loaded" or len(self.loaded_instances) > 0

    def to_context_string(self) -> str:
        """Convert to concise string for router model context."""
        parts = [self.id]

        if self.params_string:
            parts.append(f"[{self.params_string}]")

        if self.quantization:
            parts.append(f"({self.quantization.name})")

        if self.is_loaded:
            parts.append("- LOADED")
        else:
            parts.append("- available")

        if self.max_context_length:
            parts.append(f"context:{self.max_context_length//1024}k")

        # Add capabilities
        caps = []
        if self.capabilities:
            if self.capabilities.vision:
                caps.append("vision")
            if self.capabilities.trained_for_tool_use:
                caps.append("tools")

        if self.architecture:
            caps.append(self.architecture)

        if caps:
            parts.append(f"({', '.join(caps)})")

        return " ".join(parts)


class ModelsResponse(BaseModel):
    """Response from GET /api/v0/models."""

    data: list[ModelInfo]


class CompletionStats(BaseModel):
    """Performance statistics from completion response."""

    tokens_per_second: Optional[float] = None
    time_to_first_token: Optional[float] = None
    generation_time: Optional[float] = None
    stop_reason: Optional[str] = None


class CompletionUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    """Single completion choice."""

    index: int
    message: dict
    finish_reason: Optional[str] = None
    delta: Optional[dict] = None  # For streaming


class CompletionResponse(BaseModel):
    """Response from chat completion."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Optional[CompletionUsage] = None
    stats: Optional[CompletionStats] = None
