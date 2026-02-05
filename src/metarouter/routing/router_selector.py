"""Model selection using the router model."""

import json
import logging
from typing import Optional

from pydantic import BaseModel

from ..cache.benchmarks import BenchmarkFetcher, BenchmarkScores, get_benchmark_fetcher
from ..cache.performance import get_performance_cache
from ..config.settings import get_settings
from ..lmstudio.client import LMStudioClient, MultiInstanceClient
from ..lmstudio.models import ModelInfo

logger = logging.getLogger(__name__)


class ModelSelection(BaseModel):
    """Model selection result from the router model."""

    selected_model: str
    reason: str
    load_required: bool
    confidence: float


class RouterModelSelector:
    """Model selector using the router model for intelligent routing."""

    def __init__(self, client: LMStudioClient | MultiInstanceClient, router_model: str, prefer_loaded_bonus: int):
        self.client = client
        self.router_model = router_model
        self.prefer_loaded_bonus = prefer_loaded_bonus
        self.performance_cache = get_performance_cache()
        self.benchmark_fetcher = get_benchmark_fetcher()
        self.settings = get_settings()

    def _find_matching_model(
        self, selected: str, models: list[ModelInfo]
    ) -> Optional[ModelInfo]:
        """
        Find a model that matches the selected string.

        Handles cases where the router returns variations like:
        - Exact match: "qwen2.5-0.5b-instruct"
        - With quantization: "qwen2.5-0.5b-instruct (Q8_0)"
        - With extra text: "qwen2.5-0.5b-instruct [0.5B]"

        Returns:
            Matching ModelInfo or None if no match found
        """
        model_ids = {m.id: m for m in models}

        # 1. Exact match
        if selected in model_ids:
            return model_ids[selected]

        # 2. Normalize and try again (lowercase, strip whitespace)
        selected_normalized = selected.lower().strip()
        for model_id, model in model_ids.items():
            if model_id.lower() == selected_normalized:
                return model

        # 3. Check if selected starts with or contains a known model ID
        # Sort by length descending to match longest IDs first
        for model_id in sorted(model_ids.keys(), key=len, reverse=True):
            # Selected starts with the model ID
            if selected_normalized.startswith(model_id.lower()):
                logger.debug(f"Fuzzy match: '{selected}' -> '{model_id}' (prefix match)")
                return model_ids[model_id]
            # Model ID is contained in selected (handles "model (Q8_0)" format)
            if model_id.lower() in selected_normalized:
                logger.debug(f"Fuzzy match: '{selected}' -> '{model_id}' (contains match)")
                return model_ids[model_id]

        # 4. Check if any model ID starts with selected (partial input)
        for model_id, model in model_ids.items():
            if model_id.lower().startswith(selected_normalized):
                logger.debug(f"Fuzzy match: '{selected}' -> '{model_id}' (partial match)")
                return model

        return None

    async def _build_model_context(
        self,
        models: list[ModelInfo],
        benchmark_scores: dict[str, Optional[BenchmarkScores]],
    ) -> str:
        """Build concise model list for router model context."""
        lines = ["Available models:"]

        for i, model in enumerate(models, 1):
            # Basic model info
            line_parts = [f"{i}. {model.to_context_string()}"]

            # Add benchmark scores if available and enabled
            if self.settings.benchmarks.enabled:
                scores = benchmark_scores.get(model.id)
                if scores and scores.has_data():
                    line_parts.append(f"\n   Benchmarks: {scores.to_context_string()}")

            # Add performance data if available
            metrics = self.performance_cache.get_metrics(model.id)
            if metrics.sample_count > 0:
                line_parts.append(f"\n   Performance: {metrics.to_string()}")

            lines.append("".join(line_parts))

        return "\n".join(lines)

    def _build_selection_prompt(self, query: str, model_context: str) -> str:
        """Build the prompt for the router model to select a model."""
        benchmark_instructions = ""
        if self.settings.benchmarks.enabled:
            benchmark_instructions = """
BENCHMARK SCORES (when available):
- quality: Overall intelligence/capability score (higher = smarter)
- coding: Code generation and understanding ability
- math: Mathematical reasoning ability
- elo: Human preference rating from Chatbot Arena

Use these scores to make informed decisions about model quality for specific tasks.
For coding tasks, prefer higher 'coding' scores.
For math/reasoning, prefer higher 'math' scores.
"""

        return f"""You are a model router for LM Studio. Select the best model for this query.

USER QUERY:
{query}

{model_context}
{benchmark_instructions}
ROUTING GUIDELINES:
1. Match query requirements (code, math, reasoning, chat, vision, general knowledge)
2. STRONGLY prefer LOADED models (instant response, no load time)
3. Balance quality vs speed (simple queries don't need large models)
4. Consider context length if query is long
5. Only recommend loading a new model if significantly better for the task
6. For code queries, prefer models with high 'coding' benchmark or "coder" in the name
7. For vision queries, only select models with vision capability
8. For simple conversational queries, prefer small fast models (0.5B-3B)
9. For complex reasoning/math, prefer models with high 'math' benchmark or larger models (70B+)
10. Use benchmark scores as quantitative evidence when available
11. Models may be hosted on different machines (@instance_name). The router handles routing automatically - just select the best model regardless of instance

IMPORTANT: The "selected_model" MUST be the EXACT string inside the ID="..." quotes from the model list above. Do not include any other text like quantization or parameters.

Respond ONLY with valid JSON:
{{
  "selected_model": "exact-value-from-ID-quotes",
  "reason": "Brief explanation",
  "load_required": false,
  "confidence": 0.95
}}"""

    async def select_model(self, query: str) -> ModelSelection:
        """
        Select the best model for a query using the router model.

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

        # Fetch benchmark scores for all models (uses cache, fetches if needed)
        benchmark_scores: dict[str, Optional[BenchmarkScores]] = {}
        if self.settings.benchmarks.enabled:
            model_ids = [m.id for m in models]
            benchmark_scores = await self.benchmark_fetcher.get_scores_batch(model_ids)
            models_with_scores = sum(1 for s in benchmark_scores.values() if s and s.has_data())
            logger.debug(f"Benchmark data available for {models_with_scores}/{len(models)} models")

        # Build context
        model_context = await self._build_model_context(models, benchmark_scores)
        prompt = self._build_selection_prompt(query, model_context)

        # Call router model for selection
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
            logger.debug(f"Router model response: {content}")

            # Try to extract JSON from response
            selection_data = self._extract_json(content)
            selection = ModelSelection(**selection_data)

            # Find matching model (with fuzzy matching)
            matched_model = self._find_matching_model(selection.selected_model, models)

            if matched_model:
                # Update to canonical model ID
                if matched_model.id != selection.selected_model:
                    logger.debug(
                        f"Resolved model ID: '{selection.selected_model}' -> '{matched_model.id}'"
                    )
                selection.selected_model = matched_model.id
                selection.load_required = not matched_model.is_loaded
            else:
                logger.warning(
                    f"Router model selected unknown model: {selection.selected_model}, "
                    f"falling back to loaded model"
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
        Extract JSON from router model response.

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
