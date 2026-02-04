"""Main routing logic coordinator."""

import logging
from typing import AsyncIterator

from ..cache.performance import get_performance_cache
from ..config.settings import Settings
from ..lmstudio.client import LMStudioClient
from ..lmstudio.models import CompletionResponse
from .router_selector import RouterModelSelector

logger = logging.getLogger(__name__)


class ModelRouter:
    """Coordinates model selection and request routing."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = LMStudioClient(settings.lm_studio)
        self.selector = RouterModelSelector(
            client=self.client,
            router_model=settings.router.model,
            prefer_loaded_bonus=settings.router.prefer_loaded_bonus,
        )
        self.performance_cache = get_performance_cache()

    async def route_completion(
        self,
        messages: list[dict],
        stream: bool = False,
        **kwargs,
    ) -> tuple[str, CompletionResponse | AsyncIterator[bytes]]:
        """
        Route a completion request to the best model.

        Args:
            messages: Chat messages
            stream: Whether to stream response
            **kwargs: Additional completion parameters

        Returns:
            Tuple of (selected_model_id, completion_response)
        """
        # Extract user query for routing decision
        user_query = self._extract_query(messages)

        # Select best model using the router model
        selection = await self.selector.select_model(user_query)

        # Check if model needs to be loaded
        if selection.load_required:
            if self.settings.router.auto_load_models:
                logger.info(f"Auto-loading model: {selection.selected_model}")
                loaded = await self.client.load_model(selection.selected_model)
                if not loaded:
                    logger.warning(
                        f"Failed to load {selection.selected_model}, attempting anyway"
                    )
            else:
                logger.warning(
                    f"Model {selection.selected_model} not loaded, but auto_load disabled"
                )

        # Forward request to selected model
        logger.info(f"Routing request to: {selection.selected_model}")
        response = await self.client.chat_completion(
            model=selection.selected_model,
            messages=messages,
            stream=stream,
            **kwargs,
        )

        # Track performance for non-streaming responses
        if not stream and isinstance(response, CompletionResponse):
            if response.stats and self.settings.performance_tracking.enabled:
                self.performance_cache.record_inference(selection.selected_model, response.stats)

        return selection.selected_model, response

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
