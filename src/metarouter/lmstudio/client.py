"""LM Studio API client."""

import asyncio
import logging
import re
from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing, contextmanager
from time import monotonic
from typing import Optional

import httpx

from ..config.settings import LMStudioInstanceConfig, LMStudioSettings
from .models import CompletionResponse, ModelInfo, ModelsResponse

logger = logging.getLogger(__name__)

# Default weight when params_string is missing or unparseable
_DEFAULT_MODEL_WEIGHT = 7.0


def _parse_model_weight(params_string: Optional[str]) -> float:
    """Extract a numeric weight from a params_string like '7B', '14B', '70B'.

    Returns a float representing billions of parameters.
    """
    if not params_string:
        return _DEFAULT_MODEL_WEIGHT
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([BM])", params_string.strip(), re.IGNORECASE)
    if not match:
        return _DEFAULT_MODEL_WEIGHT
    value = float(match.group(1))
    # If it says "M" (millions), convert to billions
    if match.group(2).upper() == "M":
        value /= 1000
    return value or _DEFAULT_MODEL_WEIGHT


class LMStudioClient:
    """Client for interacting with a single LM Studio instance."""

    def __init__(self, settings: LMStudioSettings | LMStudioInstanceConfig, instance_name: str = "default"):
        self.base_url = settings.base_url.rstrip("/")
        self.timeout = settings.timeout
        self.refresh_interval = settings.refresh_interval
        self.instance_name = instance_name

        self._models_cache: Optional[list[ModelInfo]] = None
        self._cache_expiration = 0.0

    async def get_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        """
        Get all available models from LM Studio.

        Args:
            force_refresh: Skip cache and fetch fresh data

        Returns:
            List of ModelInfo objects
        """
        # Check cache validity
        if not force_refresh and self._models_cache is not None and monotonic() < self._cache_expiration:
            return self._models_cache

        # Fetch fresh data
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout, 5)) as client:
                response = await client.get(f"{self.base_url}/api/v0/models")
                response.raise_for_status()

                models_response = ModelsResponse.model_validate(response.json())
                # Tag each model with the instance name
                for model in models_response.data:
                    model.instance_name = self.instance_name
                self._models_cache = models_response.data
                self._cache_expiration = monotonic() + self.refresh_interval

                logger.info(
                    f"Fetched {len(self._models_cache)} models from LM Studio "
                    f"instance '{self.instance_name}' ({self.base_url})"
                )
                return self._models_cache

        except (httpx.HTTPError, httpx.InvalidURL, ValueError):
            self.clear_cache()
            raise

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **kwargs,
    ) -> CompletionResponse | AsyncGenerator[bytes, None]:
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

    async def _stream_completion(self, url: str, payload: dict) -> AsyncGenerator[bytes, None]:
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
        self._cache_expiration = 0.0
        logger.debug(f"Models cache cleared for instance '{self.instance_name}'")


class MultiInstanceClient:
    """Aggregate model discovery and route to an available LM Studio replica."""

    def __init__(self, clients: list[LMStudioClient], health_check_interval: int = 30):
        if not clients:
            raise ValueError("At least one LMStudioClient is required")
        self.clients = clients
        self.health_check_interval = health_check_interval
        self._instance_models: dict[str, list[ModelInfo]] = {}
        self._model_clients: dict[str, list[tuple[LMStudioClient, ModelInfo]]] = {}
        self._instance_load: dict[str, float] = {client.instance_name: 0.0 for client in clients}
        self._refresh_lock = asyncio.Lock()
        self._health_task: asyncio.Task | None = None

    async def get_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        """Return one entry per model ID, preferring a loaded replica."""
        async def discover(client: LMStudioClient) -> list[ModelInfo] | None:
            # Background and explicit health checks probe offline replicas.
            if not force_refresh and self._instance_models and client.instance_name not in self._instance_models:
                return None
            try:
                return await client.get_models(force_refresh=force_refresh)
            except (httpx.HTTPError, httpx.InvalidURL, ValueError) as error:
                logger.warning("Model discovery failed for %s: %s", client.instance_name, error)
                return None

        async with self._refresh_lock:
            results = await asyncio.gather(*(discover(client) for client in self.clients))
            instance_models = {}
            model_clients: dict[str, list[tuple[LMStudioClient, ModelInfo]]] = {}
            for client, models in zip(self.clients, results, strict=True):
                if models is None:
                    continue
                instance_models[client.instance_name] = models
                for model in models:
                    model_clients.setdefault(model.id, []).append((client, model))
            self._instance_models = instance_models
            self._model_clients = model_clients
            return [
                min(replicas, key=lambda entry: not entry[1].is_loaded)[1]
                for replicas in model_clients.values()
            ]

    @contextmanager
    def _reserve_replica(self, model_id: str) -> Iterator[LMStudioClient]:
        replicas = self._model_clients.get(model_id)
        if replicas:
            client, model = min(
                replicas,
                key=lambda entry: (not entry[1].is_loaded, self._instance_load[entry[0].instance_name]),
            )
            weight = _parse_model_weight(model.params_string)
        else:
            client = next(
                (client for client in self.clients if client.instance_name in self._instance_models),
                self.clients[0],
            )
            weight = _DEFAULT_MODEL_WEIGHT
        name = client.instance_name
        logger.info("Routing %s to LM Studio instance %s", model_id, name)
        self._instance_load[name] += weight
        try:
            yield client
        finally:
            self._instance_load[name] -= weight

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **kwargs,
    ) -> CompletionResponse | AsyncGenerator[bytes, None]:
        if stream:
            return self._tracked_stream(model, messages, **kwargs)
        with self._reserve_replica(model) as client:
            return await client.chat_completion(model=model, messages=messages, **kwargs)

    async def _tracked_stream(
        self, model: str, messages: list[dict], **kwargs
    ) -> AsyncGenerator[bytes, None]:
        # The reservation begins with the first read and ends on an explicit close.
        with self._reserve_replica(model) as client:
            raw = await client.chat_completion(model=model, messages=messages, stream=True, **kwargs)
            async with aclosing(raw) as chunks:
                async for chunk in chunks:
                    yield chunk

    def start_health_loop(self) -> None:
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_loop())

    async def stop_health_loop(self) -> None:
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            await asyncio.gather(self._health_task, return_exceptions=True)

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.health_check_interval)
            await self.get_models(force_refresh=True)

    async def get_instance_health(self) -> dict[str, dict]:
        """Probe each instance and update the routing map from the same results."""
        await self.get_models(force_refresh=True)
        health = {}
        for client in self.clients:
            name = client.instance_name
            models = self._instance_models.get(name)
            info = {
                "status": "unreachable" if models is None else "healthy",
                "base_url": client.base_url,
                "current_load": self._instance_load[name],
            }
            if models is not None:
                loaded = [model.id for model in models if model.is_loaded]
                info.update(total_models=len(models), loaded_models=len(loaded), loaded_model_ids=loaded)
            health[name] = info
        return health
