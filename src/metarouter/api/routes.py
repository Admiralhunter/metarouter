"""FastAPI routes for the router."""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..routing.router import ModelRouter
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    ModelInfo,
    ModelsListResponse,
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
        selected_model, response = await model_router.route_completion(
            messages=messages,
            stream=completion_request.stream or False,
            **kwargs,
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
                    "X-Selected-Model": selected_model,
                },
            )

        # Handle non-streaming response
        # Convert LMStudio response to OpenAI format
        return ChatCompletionResponse(
            id=response.id,
            object="chat.completion",
            created=response.created,
            model=selected_model,  # Use the selected model, not requested model
            choices=[
                {
                    "index": choice.index,
                    "message": choice.message,
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            usage=response.usage.model_dump() if response.usage else None,
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
            "health": "/health",
        },
    }
