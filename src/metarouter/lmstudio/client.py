"""LM Studio API client."""

import logging
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

import httpx

from ..config.settings import LMStudioInstanceConfig, LMStudioSettings
from .models import CompletionResponse, ModelInfo, ModelsResponse

logger = logging.getLogger(__name__)


class LMStudioClient:
    """Client for interacting with a single LM Studio instance."""

    def __init__(self, settings: LMStudioSettings | LMStudioInstanceConfig, instance_name: str = "default"):
        if isinstance(settings, LMStudioSettings):
            self.base_url = settings.base_url.rstrip("/")
            self.timeout = settings.timeout
            self.refresh_interval = settings.refresh_interval
        else:
            self.base_url = settings.base_url.rstrip("/")
            self.timeout = settings.timeout
            self.refresh_interval = settings.refresh_interval

        self.instance_name = instance_name
        self.settings = settings

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
                # Tag each model with the instance name
                for model in models_response.data:
                    model.instance_name = self.instance_name
                self._models_cache = models_response.data
                self._cache_timestamp = datetime.now()

                logger.info(
                    f"Fetched {len(self._models_cache)} models from LM Studio "
                    f"instance '{self.instance_name}' ({self.base_url})"
                )
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
        logger.debug(f"Models cache cleared for instance '{self.instance_name}'")


class MultiInstanceClient:
    """Client that aggregates multiple LM Studio instances.

    Presents a unified view of all models across instances and routes
    requests to the correct instance based on which instance hosts
    a given model.
    """

    def __init__(self, clients: list[LMStudioClient]):
        if not clients:
            raise ValueError("At least one LMStudioClient is required")
        self.clients = clients
        self._instance_map: dict[str, LMStudioClient] = {}
        # The first client is the default (used for router model)
        self.default_client = clients[0]

    @property
    def instance_names(self) -> list[str]:
        """Get names of all configured instances."""
        return [c.instance_name for c in self.clients]

    def _get_client_for_model(self, model_id: str) -> LMStudioClient:
        """Get the client instance that hosts a given model.

        Falls back to the default client if the model hasn't been seen yet.
        """
        client = self._instance_map.get(model_id)
        if client:
            return client
        logger.debug(
            f"Model '{model_id}' not in instance map, using default instance "
            f"'{self.default_client.instance_name}'"
        )
        return self.default_client

    async def get_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        """Get all models from all instances, deduplicated.

        If the same model ID exists on multiple instances, the loaded version
        is preferred. If both are loaded (or both unloaded), the first instance wins.
        """
        import asyncio

        async def _fetch_from(client: LMStudioClient) -> list[ModelInfo]:
            try:
                return await client.get_models(force_refresh=force_refresh)
            except Exception as e:
                logger.error(
                    f"Failed to fetch models from instance '{client.instance_name}' "
                    f"({client.base_url}): {e}"
                )
                return []

        results = await asyncio.gather(*[_fetch_from(c) for c in self.clients])

        # Merge models: prefer loaded versions, track instance mapping
        seen: dict[str, ModelInfo] = {}
        for models in results:
            for model in models:
                existing = seen.get(model.id)
                if existing is None:
                    seen[model.id] = model
                elif model.is_loaded and not existing.is_loaded:
                    # Prefer the loaded version
                    seen[model.id] = model

        # Update instance map for routing
        self._instance_map.clear()
        for model in seen.values():
            if model.instance_name:
                for client in self.clients:
                    if client.instance_name == model.instance_name:
                        self._instance_map[model.id] = client
                        break

        all_models = list(seen.values())
        logger.debug(
            f"Aggregated {len(all_models)} unique models from "
            f"{len(self.clients)} instances"
        )
        return all_models

    async def get_loaded_models(self) -> list[ModelInfo]:
        """Get only currently loaded models across all instances."""
        all_models = await self.get_models()
        return [m for m in all_models if m.is_loaded]

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **kwargs,
    ) -> CompletionResponse | AsyncIterator[bytes]:
        """Route chat completion to the correct instance for the model."""
        client = self._get_client_for_model(model)
        logger.debug(
            f"Routing completion for '{model}' to instance "
            f"'{client.instance_name}' ({client.base_url})"
        )
        return await client.chat_completion(
            model=model, messages=messages, stream=stream, **kwargs
        )

    async def load_model(self, model_id: str) -> bool:
        """Request the appropriate instance to load a model."""
        client = self._get_client_for_model(model_id)
        return await client.load_model(model_id)

    def clear_cache(self) -> None:
        """Clear caches on all instances."""
        for client in self.clients:
            client.clear_cache()
        self._instance_map.clear()
        logger.debug("Cleared caches for all instances")

    async def get_instance_health(self) -> dict[str, dict]:
        """Check connectivity to each instance.

        Returns a dict keyed by instance name with health status.
        """
        import asyncio

        async def _check(client: LMStudioClient) -> tuple[str, dict]:
            try:
                models = await client.get_models(force_refresh=True)
                loaded = [m for m in models if m.is_loaded]
                return client.instance_name, {
                    "status": "healthy",
                    "base_url": client.base_url,
                    "total_models": len(models),
                    "loaded_models": len(loaded),
                }
            except Exception as e:
                return client.instance_name, {
                    "status": "unreachable",
                    "base_url": client.base_url,
                    "error": str(e),
                }

        results = await asyncio.gather(*[_check(c) for c in self.clients])
        return dict(results)
