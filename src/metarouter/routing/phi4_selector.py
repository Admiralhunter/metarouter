"""Model selection using phi-4."""

import json
import logging
from typing import Optional

from pydantic import BaseModel

from ..cache.performance import get_performance_cache
from ..lmstudio.client import LMStudioClient
from ..lmstudio.models import ModelInfo

logger = logging.getLogger(__name__)


class ModelSelection(BaseModel):
    """Model selection result from phi-4."""

    selected_model: str
    reason: str
    load_required: bool
    confidence: float


class Phi4Selector:
    """Model selector using phi-4 for intelligent routing."""

    def __init__(self, client: LMStudioClient, router_model: str, prefer_loaded_bonus: int):
        self.client = client
        self.router_model = router_model
        self.prefer_loaded_bonus = prefer_loaded_bonus
        self.performance_cache = get_performance_cache()

    def _build_model_context(self, models: list[ModelInfo]) -> str:
        """Build concise model list for phi-4 context."""
        lines = ["Available models:"]

        for i, model in enumerate(models, 1):
            # Basic model info
            line_parts = [f"{i}. {model.to_context_string()}"]

            # Add performance data if available
            metrics = self.performance_cache.get_metrics(model.id)
            if metrics.sample_count > 0:
                line_parts.append(f" | {metrics.to_string()}")

            lines.append("".join(line_parts))

        return "\n".join(lines)

    def _build_selection_prompt(self, query: str, model_context: str) -> str:
        """Build the prompt for phi-4 to select a model."""
        return f"""You are a model router for LM Studio. Select the best model for this query.

USER QUERY:
{query}

{model_context}

ROUTING GUIDELINES:
1. Match query requirements (code, math, reasoning, chat, vision, general knowledge)
2. Prefer LOADED models (much faster, no load time) - this is very important
3. Balance quality vs speed (simple queries don't need large models)
4. Consider context length if query is long
5. Only recommend loading a new model if significantly better for the task
6. For code queries, prefer models with "coder" in the name or tool-use capability
7. For vision queries, only select models with vision capability
8. For simple conversational queries, prefer small fast models (0.5B-3B)
9. For complex reasoning/math, prefer larger models (70B+)

Respond ONLY with valid JSON in this exact format:
{{
  "selected_model": "exact-model-id-from-list",
  "reason": "Brief explanation",
  "load_required": true,
  "confidence": 0.95
}}"""

    async def select_model(self, query: str) -> ModelSelection:
        """
        Select the best model for a query using phi-4.

        Args:
            query: User's query text

        Returns:
            ModelSelection with chosen model and reasoning
        """
        # Get all available models
        models = await self.client.get_models()

        if not models:
            raise RuntimeError("No models available in LM Studio")

        logger.info(f"Selecting model for query (length: {len(query)} chars)")

        # Build context
        model_context = self._build_model_context(models)
        prompt = self._build_selection_prompt(query, model_context)

        # Call phi-4 for selection
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat_completion(
                model=self.router_model,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent routing
                max_tokens=200,
            )

            # Extract and parse response
            content = response.choices[0].message.get("content", "")
            logger.debug(f"phi-4 response: {content}")

            # Try to extract JSON from response
            selection_data = self._extract_json(content)
            selection = ModelSelection(**selection_data)

            # Validate selected model exists
            model_ids = [m.id for m in models]
            if selection.selected_model not in model_ids:
                logger.warning(
                    f"phi-4 selected unknown model: {selection.selected_model}, "
                    f"falling back to first available"
                )
                # Fallback to first loaded model or first model
                loaded_models = [m for m in models if m.is_loaded]
                fallback = loaded_models[0] if loaded_models else models[0]
                selection.selected_model = fallback.id
                selection.load_required = not fallback.is_loaded
                selection.reason = f"Fallback due to invalid selection: {selection.reason}"

            logger.info(
                f"Selected: {selection.selected_model} "
                f"(confidence: {selection.confidence:.2f}, "
                f"load_required: {selection.load_required})"
            )
            logger.info(f"Reason: {selection.reason}")

            return selection

        except Exception as e:
            logger.error(f"Error during model selection: {e}", exc_info=True)
            # Fallback: use first loaded model
            loaded = [m for m in models if m.is_loaded]
            if loaded:
                fallback_model = loaded[0].id
                logger.warning(f"Falling back to loaded model: {fallback_model}")
                return ModelSelection(
                    selected_model=fallback_model,
                    reason=f"Fallback due to selection error: {str(e)}",
                    load_required=False,
                    confidence=0.5,
                )
            else:
                # No loaded models, use first available
                fallback_model = models[0].id
                logger.warning(f"Falling back to first available model: {fallback_model}")
                return ModelSelection(
                    selected_model=fallback_model,
                    reason=f"Fallback due to selection error: {str(e)}",
                    load_required=True,
                    confidence=0.5,
                )

    def _extract_json(self, content: str) -> dict:
        """
        Extract JSON from phi-4 response.

        Handles cases where JSON is wrapped in markdown code blocks or has extra text.
        """
        # Try to find JSON in markdown code block
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()

        # Try to find JSON object
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            content = content[start:end]

        return json.loads(content)
