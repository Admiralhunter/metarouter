"""Main routing logic coordinator."""

import logging
from typing import AsyncIterator

from ..cache.performance import get_performance_cache
from ..config.settings import Settings
from ..lmstudio.client import LMStudioClient, MultiInstanceClient
from ..lmstudio.models import CompletionResponse
from .router_selector import ModelSelection, RouterModelSelector

logger = logging.getLogger(__name__)


class ModelRouter:
    """Coordinates model selection and request routing."""

    def __init__(self, settings: Settings):
        self.settings = settings

        # Build per-instance clients from config
        instances = settings.lm_studio.get_instances()
        clients = [
            LMStudioClient(inst, instance_name=inst.name)
            for inst in instances
        ]
        self.multi_client = MultiInstanceClient(
            clients,
            health_check_interval=settings.lm_studio.health_check_interval,
        )

        # Backward-compatible: expose .client pointing to the multi-instance client
        self.client = self.multi_client

        self.selector = RouterModelSelector(
            client=self.multi_client,
            router_model=settings.router.model,
            prefer_loaded_bonus=settings.router.prefer_loaded_bonus,
        )
        self.performance_cache = get_performance_cache()

    async def route_completion(
        self,
        messages: list[dict],
        stream: bool = False,
        **kwargs,
    ) -> tuple[ModelSelection, CompletionResponse | AsyncIterator[bytes]]:
        """
        Route a completion request to the best model.

        Args:
            messages: Chat messages
            stream: Whether to stream response
            **kwargs: Additional completion parameters

        Returns:
            Tuple of (model_selection, completion_response)
        """
        # Extract user query for routing decision
        user_query = self._extract_query(messages)

        # Select best model using the router model
        selection = await self.selector.select_model(user_query)

        # Log if model needs to be loaded (LM Studio handles JIT loading automatically)
        if selection.load_required:
            if self.settings.router.auto_load_models:
                logger.info(f"Model {selection.selected_model} not loaded - LM Studio will JIT load it")
            else:
                logger.warning(
                    f"Model {selection.selected_model} not loaded and auto_load disabled - "
                    "request may fail if LM Studio JIT loading is disabled"
                )

        # Forward request to selected model (multi-client routes to correct instance)
        logger.info(f"Routing request to: {selection.selected_model}")
        response = await self.multi_client.chat_completion(
            model=selection.selected_model,
            messages=messages,
            stream=stream,
            **kwargs,
        )

        # Track performance for non-streaming responses
        if not stream and isinstance(response, CompletionResponse):
            if response.stats and self.settings.performance_tracking.enabled:
                self.performance_cache.record_inference(selection.selected_model, response.stats)

        return selection, response

    def _extract_query(self, messages: list[dict]) -> str:
        """
        Extract the most relevant query text from messages.

        Prioritizes the last user message, but includes context if needed.
        """
        if not messages:
            return ""

        # Get last user message
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle multimodal content
                    text_parts = [
                        item["text"] for item in content if isinstance(item, dict) and "text" in item
                    ]
                    return " ".join(text_parts)

        # Fallback: return all message content
        all_content = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                all_content.append(content)

        return " ".join(all_content)
