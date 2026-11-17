"""LM Studio API client."""

import asyncio
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

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
    match = re.search(r"([\d.]+)\s*[BbMm]", params_string)
    if not match:
        return _DEFAULT_MODEL_WEIGHT
    value = float(match.group(1))
    # If it says "M" (millions), convert to billions
    if params_string.upper().endswith("M"):
        value /= 1000
    return value


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
    requests to the correct instance.  Uses weighted least-connections
    to prefer instances with the lowest current load, where each
    in-flight request is weighted by the model's parameter count
    (a 70B completion counts ~10x more than a 7B one).

    Runs a background health check loop to track which instances are
    reachable and update the model map automatically.
    """

    def __init__(self, clients: list[LMStudioClient], health_check_interval: int = 30):
        if not clients:
            raise ValueError("At least one LMStudioClient is required")
        self.clients = clients
        self.health_check_interval = health_check_interval

        # model_id -> list of clients that have it (loaded or available)
        self._model_clients: dict[str, list[LMStudioClient]] = {}
        # model_id -> ModelInfo (representative, for weight lookup)
        self._model_info: dict[str, ModelInfo] = {}
        # instance_name -> current weighted load (sum of in-flight weights)
        self._instance_load: dict[str, float] = defaultdict(float)
        # Track which instances are currently healthy
        self._healthy: set[str] = {c.instance_name for c in clients}
        # The first client is the default (used for router model fallback)
        self.default_client = clients[0]
        # Background task handle
        self._health_task: Optional[asyncio.Task] = None

    @property
    def instance_names(self) -> list[str]:
        """Get names of all configured instances."""
        return [c.instance_name for c in self.clients]

    # ── load tracking ────────────────────────────────────────────────

    def _model_weight(self, model_id: str) -> float:
        """Get the weight for a model based on its parameter count."""
        info = self._model_info.get(model_id)
        if info:
            return _parse_model_weight(info.params_string)
        return _DEFAULT_MODEL_WEIGHT

    def _add_load(self, instance_name: str, weight: float) -> None:
        self._instance_load[instance_name] += weight

    def _remove_load(self, instance_name: str, weight: float) -> None:
        self._instance_load[instance_name] = max(
            0.0, self._instance_load[instance_name] - weight
        )

    def get_instance_loads(self) -> dict[str, float]:
        """Get current weighted load per instance (for diagnostics)."""
        return dict(self._instance_load)

    # ── routing ──────────────────────────────────────────────────────

    def _get_client_for_model(self, model_id: str) -> LMStudioClient:
        """Pick the least-loaded healthy instance that has this model.

        Falls back to the default client if the model is unknown.
        """
        candidates = self._model_clients.get(model_id)
        if not candidates:
            logger.debug(
                f"Model '{model_id}' not in instance map, using default instance "
                f"'{self.default_client.instance_name}'"
            )
            return self.default_client

        # Filter to healthy instances
        healthy_candidates = [
            c for c in candidates if c.instance_name in self._healthy
        ]
        if not healthy_candidates:
            # All unhealthy — try them all anyway
            healthy_candidates = candidates

        # Pick the one with lowest weighted load
        best = min(
            healthy_candidates,
            key=lambda c: self._instance_load.get(c.instance_name, 0.0),
        )
        return best

    # ── model aggregation ────────────────────────────────────────────

    async def get_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        """Get all models from all instances, merged.

        If the same model ID exists on multiple instances, a single entry
        is returned (preferring loaded). The internal routing map tracks
        all instances per model for load balancing.
        """
        async def _fetch_from(client: LMStudioClient) -> tuple[LMStudioClient, list[ModelInfo]]:
            try:
                models = await client.get_models(force_refresh=force_refresh)
                return client, models
            except Exception as e:
                logger.error(
                    f"Failed to fetch models from instance '{client.instance_name}' "
                    f"({client.base_url}): {e}"
                )
                return client, []

        results = await asyncio.gather(*[_fetch_from(c) for c in self.clients])

        # Build per-model client lists and pick a representative ModelInfo
        new_model_clients: dict[str, list[LMStudioClient]] = {}
        representative: dict[str, ModelInfo] = {}

        for client, models in results:
            for model in models:
                new_model_clients.setdefault(model.id, []).append(client)

                existing = representative.get(model.id)
                if existing is None:
                    representative[model.id] = model
                elif model.is_loaded and not existing.is_loaded:
                    representative[model.id] = model

        self._model_clients = new_model_clients
        self._model_info = representative

        all_models = list(representative.values())
        logger.debug(
            f"Aggregated {len(all_models)} unique models from "
            f"{len(self.clients)} instances"
        )
        return all_models

    async def get_loaded_models(self) -> list[ModelInfo]:
        """Get only currently loaded models across all instances."""
        all_models = await self.get_models()
        return [m for m in all_models if m.is_loaded]

    # ── completions ──────────────────────────────────────────────────

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **kwargs,
    ) -> CompletionResponse | AsyncIterator[bytes]:
        """Route chat completion to the least-loaded instance for the model."""
        client = self._get_client_for_model(model)
        weight = self._model_weight(model)
        instance = client.instance_name

        logger.info(
            f"Routing '{model}' (weight={weight:.0f}) to instance "
            f"'{instance}' ({client.base_url}) "
            f"[load before: {self._instance_load.get(instance, 0):.0f}]"
        )

        self._add_load(instance, weight)

        if stream:
            # Wrap the stream so load is released when iteration ends
            raw_stream = await client.chat_completion(
                model=model, messages=messages, stream=True, **kwargs
            )
            return self._tracked_stream(raw_stream, instance, weight)
        else:
            try:
                response = await client.chat_completion(
                    model=model, messages=messages, stream=False, **kwargs
                )
                return response
            finally:
                self._remove_load(instance, weight)

    async def _tracked_stream(
        self,
        raw: AsyncIterator[bytes],
        instance_name: str,
        weight: float,
    ) -> AsyncIterator[bytes]:
        """Wrap an async byte stream to release load when done."""
        try:
            async for chunk in raw:
                yield chunk
        finally:
            self._remove_load(instance_name, weight)

    async def load_model(self, model_id: str) -> bool:
        """Request the appropriate instance to load a model."""
        client = self._get_client_for_model(model_id)
        return await client.load_model(model_id)

    # ── health monitoring ────────────────────────────────────────────

    def start_health_loop(self) -> None:
        """Start the background health-check loop.

        Should be called once during application startup.
        """
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_loop())
            logger.info(
                f"Started instance health monitor (interval={self.health_check_interval}s)"
            )

    def stop_health_loop(self) -> None:
        """Cancel the background health-check loop."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            logger.info("Stopped instance health monitor")

    async def _health_loop(self) -> None:
        """Periodically refresh models from every instance.

        This serves double duty:
        - Updates _healthy set so routing avoids dead instances
        - Refreshes the model list so new/removed models are detected
        """
        while True:
            await asyncio.sleep(self.health_check_interval)
            try:
                health = await self.get_instance_health()
                newly_healthy = {
                    name for name, info in health.items()
                    if info.get("status") == "healthy"
                }
                came_back = newly_healthy - self._healthy
                went_down = self._healthy - newly_healthy
                if came_back:
                    logger.info(f"Instance(s) recovered: {', '.join(came_back)}")
                if went_down:
                    logger.warning(f"Instance(s) unreachable: {', '.join(went_down)}")
                self._healthy = newly_healthy

                # Refresh models to pick up any changes
                await self.get_models(force_refresh=True)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    async def get_instance_health(self) -> dict[str, dict]:
        """Check connectivity to each instance.

        Returns a dict keyed by instance name with health status and load.
        """
        async def _check(client: LMStudioClient) -> tuple[str, dict]:
            try:
                models = await client.get_models(force_refresh=True)
                loaded = [m for m in models if m.is_loaded]
                return client.instance_name, {
                    "status": "healthy",
                    "base_url": client.base_url,
                    "total_models": len(models),
                    "loaded_models": len(loaded),
                    "loaded_model_ids": [m.id for m in loaded],
                    "current_load": self._instance_load.get(client.instance_name, 0.0),
                }
            except Exception as e:
                return client.instance_name, {
                    "status": "unreachable",
                    "base_url": client.base_url,
                    "error": str(e),
                    "current_load": self._instance_load.get(client.instance_name, 0.0),
                }

        results = await asyncio.gather(*[_check(c) for c in self.clients])
        return dict(results)

    # ── cache management ─────────────────────────────────────────────

    def clear_cache(self) -> None:
        """Clear caches on all instances."""
        for client in self.clients:
            client.clear_cache()
        self._model_clients.clear()
        self._model_info.clear()
        logger.debug("Cleared caches for all instances")
