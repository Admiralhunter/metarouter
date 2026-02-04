"""Benchmark data cache for model quality metrics."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default cache file location
DEFAULT_CACHE_PATH = Path(__file__).parent.parent.parent.parent / "data" / "benchmarks_cache.json"


class BenchmarkScores(BaseModel):
    """Benchmark scores for a model."""

    # Composite indices from Artificial Analysis
    intelligence_index: Optional[float] = None  # Overall quality score
    coding_index: Optional[float] = None  # Coding ability
    math_index: Optional[float] = None  # Math reasoning

    # Individual benchmark scores
    mmlu_pro: Optional[float] = None  # General knowledge
    gpqa: Optional[float] = None  # PhD-level science
    livecodebench: Optional[float] = None  # Real-world coding
    math_500: Optional[float] = None  # Math problems
    aime: Optional[float] = None  # Competition math
    humaneval: Optional[float] = None  # Code generation

    # Arena ratings
    arena_elo: Optional[float] = None  # Human preference Elo

    def to_context_string(self) -> str:
        """Convert to concise string for router model context."""
        parts = []

        if self.intelligence_index is not None:
            parts.append(f"quality={self.intelligence_index:.0f}")
        if self.coding_index is not None:
            parts.append(f"coding={self.coding_index:.0f}")
        if self.math_index is not None:
            parts.append(f"math={self.math_index:.0f}")
        if self.arena_elo is not None:
            parts.append(f"elo={self.arena_elo:.0f}")

        return ", ".join(parts) if parts else ""

    def has_data(self) -> bool:
        """Check if any benchmark data is available."""
        return any([
            self.intelligence_index,
            self.coding_index,
            self.math_index,
            self.arena_elo,
        ])


class CachedModel(BaseModel):
    """Cached benchmark data for a single model."""

    model_id: str  # Canonical model ID from API
    model_name: Optional[str] = None  # Display name
    scores: BenchmarkScores = Field(default_factory=BenchmarkScores)
    # Aliases for fuzzy matching (lowercase variations, common names)
    aliases: list[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkCache(BaseModel):
    """Cache for benchmark data with metadata."""

    models: dict[str, CachedModel] = Field(default_factory=dict)
    last_fetch: Optional[datetime] = None
    source: str = "artificial_analysis"
    version: str = "1.0"


class BenchmarkFetcher:
    """Fetches and caches benchmark data from external sources."""

    ARTIFICIAL_ANALYSIS_API = "https://artificialanalysis.ai/api/v2/data/llms/models"

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        cache_ttl_hours: int = 24,
        api_timeout: int = 30,
    ):
        """
        Initialize benchmark fetcher.

        Args:
            cache_path: Path to cache file (defaults to data/benchmarks_cache.json)
            cache_ttl_hours: Hours before cache is considered stale
            api_timeout: HTTP timeout for API requests
        """
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.api_timeout = api_timeout
        self._cache: Optional[BenchmarkCache] = None

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_cache(self) -> BenchmarkCache:
        """Load cache from disk."""
        if self._cache is not None:
            return self._cache

        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    data = json.load(f)
                self._cache = BenchmarkCache(**data)
                logger.debug(f"Loaded benchmark cache with {len(self._cache.models)} models")
            except Exception as e:
                logger.warning(f"Failed to load benchmark cache: {e}")
                self._cache = BenchmarkCache()
        else:
            self._cache = BenchmarkCache()

        return self._cache

    def _save_cache(self) -> None:
        """Save cache to disk."""
        if self._cache is None:
            return

        try:
            self._ensure_cache_dir()
            with open(self.cache_path, "w") as f:
                json.dump(self._cache.model_dump(mode="json"), f, indent=2, default=str)
            logger.debug(f"Saved benchmark cache with {len(self._cache.models)} models")
        except Exception as e:
            logger.error(f"Failed to save benchmark cache: {e}")

    def _generate_aliases(self, model_id: str, model_name: Optional[str] = None) -> list[str]:
        """Generate search aliases for fuzzy matching."""
        aliases = set()

        # Add lowercase versions
        aliases.add(model_id.lower())
        if model_name:
            aliases.add(model_name.lower())

        # Extract base model name (remove version suffixes, quantization)
        base = model_id.lower()
        # Remove common suffixes
        for suffix in ["-instruct", "-chat", "-base", "-hf", "-gguf", "-mlx"]:
            base = base.replace(suffix, "")
        aliases.add(base)

        # Handle common naming patterns
        # "meta-llama/llama-3.1-70b" -> "llama-3.1-70b", "llama3.1", "llama-3"
        if "/" in model_id:
            short_name = model_id.split("/")[-1].lower()
            aliases.add(short_name)
            # Remove size suffix for broader matching
            for size in ["0.5b", "1b", "3b", "7b", "8b", "13b", "14b", "30b", "32b", "70b", "72b", "120b", "235b"]:
                if size in short_name:
                    aliases.add(short_name.replace(f"-{size}", "").replace(size, ""))

        return list(aliases)

    def _parse_artificial_analysis_response(self, data: list[dict]) -> dict[str, CachedModel]:
        """Parse Artificial Analysis API response into cached models."""
        models = {}

        for item in data:
            model_id = item.get("model_id") or item.get("id")
            if not model_id:
                continue

            scores = BenchmarkScores(
                intelligence_index=item.get("artificial_analysis_intelligence_index"),
                coding_index=item.get("artificial_analysis_coding_index"),
                math_index=item.get("artificial_analysis_math_index"),
                mmlu_pro=item.get("mmlu_pro"),
                gpqa=item.get("gpqa"),
                livecodebench=item.get("livecodebench"),
                math_500=item.get("math_500"),
                aime=item.get("aime"),
                humaneval=item.get("humaneval"),
                arena_elo=item.get("arena_elo") or item.get("chatbot_arena_elo"),
            )

            model_name = item.get("name") or item.get("display_name")
            aliases = self._generate_aliases(model_id, model_name)

            cached = CachedModel(
                model_id=model_id,
                model_name=model_name,
                scores=scores,
                aliases=aliases,
            )

            # Store by canonical ID and all aliases
            models[model_id.lower()] = cached
            for alias in aliases:
                if alias not in models:
                    models[alias] = cached

        return models

    async def fetch_from_api(self) -> bool:
        """
        Fetch fresh benchmark data from Artificial Analysis API.

        Returns:
            True if fetch was successful
        """
        logger.info("Fetching benchmark data from Artificial Analysis API...")

        try:
            async with httpx.AsyncClient(timeout=self.api_timeout) as client:
                response = await client.get(self.ARTIFICIAL_ANALYSIS_API)
                response.raise_for_status()
                data = response.json()

            # Handle both direct array and wrapped response
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            elif isinstance(data, dict) and "models" in data:
                data = data["models"]

            if not isinstance(data, list):
                logger.error(f"Unexpected API response format: {type(data)}")
                return False

            # Parse and cache
            models = self._parse_artificial_analysis_response(data)
            self._cache = BenchmarkCache(
                models=models,
                last_fetch=datetime.utcnow(),
            )
            self._save_cache()

            logger.info(f"Fetched benchmark data for {len(models)} model entries")
            return True

        except httpx.HTTPStatusError as e:
            logger.error(f"API request failed: {e.response.status_code} - {e.response.text[:200]}")
            return False
        except httpx.RequestError as e:
            logger.error(f"Network error fetching benchmarks: {e}")
            return False
        except Exception as e:
            logger.error(f"Error fetching benchmarks: {e}", exc_info=True)
            return False

    def _find_model_in_cache(self, model_id: str) -> Optional[CachedModel]:
        """Find a model in cache using fuzzy matching."""
        cache = self._load_cache()

        # Normalize the search ID
        search_id = model_id.lower().strip()

        # Direct lookup
        if search_id in cache.models:
            return cache.models[search_id]

        # Try removing common suffixes from LM Studio model IDs
        # e.g., "qwen2.5-coder-32b-instruct" -> match "qwen2.5-coder-32b"
        base_id = search_id
        for suffix in ["-instruct", "-chat", "-base", "-gguf", "-mlx"]:
            base_id = base_id.replace(suffix, "")

        if base_id in cache.models:
            return cache.models[base_id]

        # Partial match: check if any cached model ID is contained in search ID
        for cached_id, cached_model in cache.models.items():
            if cached_id in search_id or search_id in cached_id:
                return cached_model

        # Check aliases
        for cached_model in cache.models.values():
            for alias in cached_model.aliases:
                if alias in search_id or search_id in alias:
                    return cached_model

        return None

    async def get_scores(
        self,
        model_id: str,
        force_refresh: bool = False,
    ) -> Optional[BenchmarkScores]:
        """
        Get benchmark scores for a model.

        Args:
            model_id: Model identifier (will use fuzzy matching)
            force_refresh: Force fetch from API even if cached

        Returns:
            BenchmarkScores or None if not found
        """
        cache = self._load_cache()

        # Check if we need to refresh
        needs_refresh = force_refresh or cache.last_fetch is None
        if not needs_refresh and cache.last_fetch:
            age = datetime.utcnow() - cache.last_fetch
            needs_refresh = age > self.cache_ttl

        # Try cache first (unless force refresh)
        if not force_refresh:
            cached = self._find_model_in_cache(model_id)
            if cached:
                return cached.scores

        # Model not in cache or needs refresh
        if needs_refresh or not self._find_model_in_cache(model_id):
            await self.fetch_from_api()
            # Try again after refresh
            cached = self._find_model_in_cache(model_id)
            if cached:
                return cached.scores

        return None

    async def get_scores_batch(
        self,
        model_ids: list[str],
        force_refresh: bool = False,
    ) -> dict[str, Optional[BenchmarkScores]]:
        """
        Get benchmark scores for multiple models.

        Efficiently fetches from API only if needed.

        Args:
            model_ids: List of model identifiers
            force_refresh: Force fetch from API

        Returns:
            Dict mapping model_id to BenchmarkScores (or None if not found)
        """
        cache = self._load_cache()
        results = {}
        missing = []

        # Check cache for each model
        if not force_refresh:
            for model_id in model_ids:
                cached = self._find_model_in_cache(model_id)
                if cached:
                    results[model_id] = cached.scores
                else:
                    missing.append(model_id)
        else:
            missing = model_ids

        # Fetch if we have missing models or forced refresh
        if missing or force_refresh:
            # Check if cache is stale
            needs_refresh = force_refresh or cache.last_fetch is None
            if not needs_refresh and cache.last_fetch:
                age = datetime.utcnow() - cache.last_fetch
                needs_refresh = age > self.cache_ttl

            if needs_refresh or missing:
                await self.fetch_from_api()

                # Retry lookups for missing models
                for model_id in missing:
                    cached = self._find_model_in_cache(model_id)
                    results[model_id] = cached.scores if cached else None

        # Ensure all requested models are in results
        for model_id in model_ids:
            if model_id not in results:
                results[model_id] = None

        return results

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        cache = self._load_cache()
        unique_models = len(set(m.model_id for m in cache.models.values()))
        return {
            "total_entries": len(cache.models),
            "unique_models": unique_models,
            "last_fetch": cache.last_fetch.isoformat() if cache.last_fetch else None,
            "cache_path": str(self.cache_path),
        }

    def clear_cache(self) -> None:
        """Clear the benchmark cache."""
        self._cache = BenchmarkCache()
        if self.cache_path.exists():
            self.cache_path.unlink()
        logger.info("Benchmark cache cleared")


# Global benchmark fetcher instance
_benchmark_fetcher: Optional[BenchmarkFetcher] = None


def get_benchmark_fetcher() -> BenchmarkFetcher:
    """Get or create global benchmark fetcher instance."""
    global _benchmark_fetcher
    if _benchmark_fetcher is None:
        from ..config.settings import get_settings

        settings = get_settings()
        _benchmark_fetcher = BenchmarkFetcher(
            cache_ttl_hours=getattr(settings, "benchmark_cache_ttl_hours", 24),
        )
    return _benchmark_fetcher
