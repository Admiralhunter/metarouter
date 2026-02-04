"""LM Studio API client."""

import logging
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

import httpx

from ..config.settings import LMStudioSettings
from .models import CompletionResponse, ModelInfo, ModelsResponse

logger = logging.getLogger(__name__)


class LMStudioClient:
    """Client for interacting with LM Studio API."""

    def __init__(self, settings: LMStudioSettings):
        self.settings = settings
        self.base_url = settings.base_url.rstrip("/")
        self.timeout = settings.timeout
        self.refresh_interval = settings.refresh_interval

        self._models_cache: Optional[list[ModelInfo]] = None
        self._cache_timestamp: Optional[datetime] = None

    async def get_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        """
        Get all available models from LM Studio.

        Args:
            force_refresh: Skip cache and fetch fresh data

        Returns:
            List of ModelInfo objects
        """
        # Check cache validity
        if not force_refresh and self._models_cache is not None and self._cache_timestamp:
            age = datetime.now() - self._cache_timestamp
            if age < timedelta(seconds=self.refresh_interval):
                logger.debug(f"Returning cached models (age: {age.total_seconds():.1f}s)")
                return self._models_cache

        # Fetch fresh data
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/v0/models")
                response.raise_for_status()

                models_response = ModelsResponse(**response.json())
                self._models_cache = models_response.data
                self._cache_timestamp = datetime.now()

                logger.info(f"Fetched {len(self._models_cache)} models from LM Studio")
                return self._models_cache

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch models from LM Studio: {e}")
            # Return cached data if available
            if self._models_cache:
                logger.warning("Using stale cached models due to API error")
                return self._models_cache
            raise

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **kwargs,
    ) -> CompletionResponse | AsyncIterator[bytes]:
        """
        Send chat completion request to specific model.

        Args:
            model: Model ID to use
            messages: Chat messages
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            CompletionResponse or async iterator of SSE bytes (if streaming)
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        logger.debug(f"Sending completion request to {model} (stream={stream})")

        if stream:
            return self._stream_completion(url, payload)
        else:
            return await self._non_stream_completion(url, payload)

    async def _non_stream_completion(self, url: str, payload: dict) -> CompletionResponse:
        """Handle non-streaming completion."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

            completion = CompletionResponse(**response.json())
            logger.debug(
                f"Completion finished: {completion.usage.completion_tokens if completion.usage else '?'} tokens"
            )
            return completion

    async def _stream_completion(self, url: str, payload: dict) -> AsyncIterator[bytes]:
        """Handle streaming completion."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

    async def load_model(self, model_id: str) -> bool:
        """
        Request LM Studio to load a model.

        LM Studio supports Just-In-Time (JIT) model loading - when you send a
        chat completion request to a model that isn't loaded, LM Studio will
        automatically load it. This method is informational only.

        Args:
            model_id: Model ID to load

        Returns:
            True - LM Studio handles loading automatically via JIT
        """
        logger.info(
            f"Model {model_id} will be JIT-loaded by LM Studio when the request is made"
        )
        return True

    async def get_loaded_models(self) -> list[ModelInfo]:
        """Get only currently loaded models."""
        all_models = await self.get_models()
        return [m for m in all_models if m.is_loaded]

    def clear_cache(self) -> None:
        """Clear the models cache."""
        self._models_cache = None
        self._cache_timestamp = None
        logger.debug("Models cache cleared")
