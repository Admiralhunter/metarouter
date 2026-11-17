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


class SessionMetadata(BaseModel):
    """Metadata about the routing decision for this session."""

    selected_model: str = Field(description="The model ID selected by the router")
    reason: str = Field(description="Explanation of why this model was selected")
    confidence: float = Field(description="Router's confidence in the selection (0-1)")
    load_required: bool = Field(description="Whether the model needed to be loaded")


class ChatCompletionResponse(BaseModel):
    """Response for chat completion."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None
    session_metadata: Optional[SessionMetadata] = Field(
        default=None, description="Metadata about the routing decision"
    )


class ModelInfo(BaseModel):
    """Model information (OpenAI-compatible)."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "lm-studio"
    instance_name: Optional[str] = Field(None, description="LM Studio instance hosting this model")


class ModelsListResponse(BaseModel):
    """List of available models."""

    object: str = "list"
    data: list[ModelInfo]


class ErrorResponse(BaseModel):
    """Error response."""

    error: dict[str, Any]


class BenchmarkScoresResponse(BaseModel):
    """Benchmark scores for a model."""

    intelligence_index: Optional[float] = Field(None, description="Overall quality score")
    coding_index: Optional[float] = Field(None, description="Coding ability score")
    math_index: Optional[float] = Field(None, description="Math reasoning score")
    mmlu_pro: Optional[float] = Field(None, description="General knowledge (MMLU-Pro)")
    gpqa: Optional[float] = Field(None, description="PhD-level science questions")
    livecodebench: Optional[float] = Field(None, description="Real-world coding tasks")
    math_500: Optional[float] = Field(None, description="Math problem solving")
    aime: Optional[float] = Field(None, description="Competition math")
    humaneval: Optional[float] = Field(None, description="Code generation ability")
    arena_elo: Optional[float] = Field(None, description="Human preference Elo rating")


class PerformanceMetricsResponse(BaseModel):
    """Performance metrics for a model."""

    avg_tokens_per_second: Optional[float] = Field(None, description="Average throughput")
    avg_time_to_first_token: Optional[float] = Field(None, description="Average latency to first token")
    avg_generation_time: Optional[float] = Field(None, description="Average total generation time")
    sample_count: int = Field(0, description="Number of samples tracked")


class ModelMetrics(BaseModel):
    """Combined metrics for a single model."""

    model_id: str = Field(description="Model identifier")
    model_name: Optional[str] = Field(None, description="Display name from benchmark data")
    benchmark_scores: Optional[BenchmarkScoresResponse] = Field(None, description="Benchmark data from Artificial Analysis")
    performance_metrics: Optional[PerformanceMetricsResponse] = Field(None, description="Real-time performance metrics")


class CacheStats(BaseModel):
    """Statistics about the caches."""

    benchmark_cache: dict[str, Any] = Field(description="Benchmark cache statistics")
    performance_cache: dict[str, Any] = Field(description="Performance cache statistics")


class MetricsResponse(BaseModel):
    """Response containing all cached metrics used by phi4 for model routing."""

    description: str = Field(
        default="Metrics cached for phi4 model routing decisions",
        description="Description of what these metrics are used for"
    )
    models: list[ModelMetrics] = Field(description="Metrics for each model")
    cache_stats: CacheStats = Field(description="Cache statistics")
