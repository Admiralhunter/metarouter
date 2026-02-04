"""FastAPI routes for the router."""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..cache.benchmarks import get_benchmark_fetcher
from ..cache.performance import get_performance_cache
from ..routing.router import ModelRouter
from .schemas import (
    BenchmarkScoresResponse,
    CacheStats,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    MetricsResponse,
    ModelInfo,
    ModelMetrics,
    ModelsListResponse,
    PerformanceMetricsResponse,
    SessionMetadata,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_router_from_request(request: Request) -> ModelRouter:
    """Get ModelRouter instance from request state."""
    return request.app.state.router


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models(request: Request) -> ModelsListResponse:
    """List all available models."""
    try:
        model_router = get_router_from_request(request)
        models = await model_router.client.get_models()

        model_list = [
            ModelInfo(
                id=model.id,
                object="model",
                created=0,
                owned_by=model.publisher or "lm-studio",
            )
            for model in models
        ]

        return ModelsListResponse(object="list", data=model_list)

    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    completion_request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """
    Handle chat completion requests with intelligent routing.

    This is the main endpoint - it routes requests to the best model
    based on query content and model availability.
    """
    try:
        model_router = get_router_from_request(request)

        # Convert messages to dict format
        messages = [msg.model_dump() for msg in completion_request.messages]

        # Get additional parameters
        kwargs = {
            k: v
            for k, v in completion_request.to_dict().items()
            if k not in ["model", "messages", "stream"]
        }

        # Route the request
        selection, response = await model_router.route_completion(
            messages=messages,
            stream=completion_request.stream or False,
            **kwargs,
        )

        # Build session metadata from routing decision
        session_metadata = SessionMetadata(
            selected_model=selection.selected_model,
            reason=selection.reason,
            confidence=selection.confidence,
            load_required=selection.load_required,
        )

        # Handle streaming response
        if completion_request.stream:
            async def stream_generator():
                async for chunk in response:
                    yield chunk

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Selected-Model": selection.selected_model,
                    "X-Selection-Reason": selection.reason,
                    "X-Selection-Confidence": str(selection.confidence),
                },
            )

        # Handle non-streaming response
        # Convert LMStudio response to OpenAI format
        return ChatCompletionResponse(
            id=response.id,
            object="chat.completion",
            created=response.created,
            model=selection.selected_model,  # Use the selected model, not requested model
            choices=[
                {
                    "index": choice.index,
                    "message": choice.message,
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            usage=response.usage.model_dump() if response.usage else None,
            session_metadata=session_metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@router.get("/v1/metrics", response_model=MetricsResponse)
async def get_metrics(request: Request) -> MetricsResponse:
    """
    Get cached metrics used by phi4 for model routing decisions.

    Returns benchmark scores (from Artificial Analysis) and real-time
    performance metrics for all models that have been tracked.
    """
    try:
        model_router = get_router_from_request(request)
        benchmark_fetcher = get_benchmark_fetcher()
        performance_cache = get_performance_cache()

        # Get all available models from LM Studio
        models = await model_router.client.get_models()
        model_ids = [m.id for m in models]

        # Get benchmark scores for all models
        benchmark_scores = await benchmark_fetcher.get_scores_batch(model_ids)

        # Get performance metrics for all tracked models
        performance_metrics = performance_cache.get_all_metrics()

        # Build response
        model_metrics_list = []
        for model_id in model_ids:
            scores = benchmark_scores.get(model_id)
            perf = performance_metrics.get(model_id)

            # Convert benchmark scores to response format
            benchmark_response = None
            if scores and scores.has_data():
                benchmark_response = BenchmarkScoresResponse(
                    intelligence_index=scores.intelligence_index,
                    coding_index=scores.coding_index,
                    math_index=scores.math_index,
                    mmlu_pro=scores.mmlu_pro,
                    gpqa=scores.gpqa,
                    livecodebench=scores.livecodebench,
                    math_500=scores.math_500,
                    aime=scores.aime,
                    humaneval=scores.humaneval,
                    arena_elo=scores.arena_elo,
                )

            # Convert performance metrics to response format
            perf_response = None
            if perf and perf.sample_count > 0:
                perf_response = PerformanceMetricsResponse(
                    avg_tokens_per_second=perf.avg_tokens_per_second,
                    avg_time_to_first_token=perf.avg_time_to_first_token,
                    avg_generation_time=perf.avg_generation_time,
                    sample_count=perf.sample_count,
                )

            # Get model name from benchmark cache if available
            model_name = None
            cached_model = benchmark_fetcher._find_model_in_cache(model_id)
            if cached_model:
                model_name = cached_model.model_name

            model_metrics_list.append(
                ModelMetrics(
                    model_id=model_id,
                    model_name=model_name,
                    benchmark_scores=benchmark_response,
                    performance_metrics=perf_response,
                )
            )

        # Get cache statistics
        benchmark_stats = benchmark_fetcher.get_cache_stats()
        perf_stats = {
            "tracked_models": len(performance_metrics),
            "total_samples": sum(m.sample_count for m in performance_metrics.values()),
        }

        return MetricsResponse(
            models=model_metrics_list,
            cache_stats=CacheStats(
                benchmark_cache=benchmark_stats,
                performance_cache=perf_stats,
            ),
        )

    except Exception as e:
        logger.error(f"Error fetching metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": "MetaRouter",
        "version": "0.1.0",
        "description": "LLM-powered intelligent routing for LM Studio",
        "endpoints": {
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "metrics": "/v1/metrics",
            "health": "/health",
        },
    }
