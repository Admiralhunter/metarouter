"""Performance tracking for models."""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from ..lmstudio.models import CompletionStats

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics for a model."""

    avg_tokens_per_second: Optional[float] = None
    avg_time_to_first_token: Optional[float] = None
    avg_generation_time: Optional[float] = None
    sample_count: int = 0

    def to_string(self) -> str:
        """Convert to readable string for router model context."""
        parts = []
        if self.avg_tokens_per_second:
            parts.append(f"~{self.avg_tokens_per_second:.1f} tokens/sec")
        if self.avg_time_to_first_token:
            parts.append(f"TTFT: {self.avg_time_to_first_token:.2f}s")
        return ", ".join(parts) if parts else "no data"


class PerformanceCache:
    """Cache for tracking model performance metrics."""

    def __init__(self, sample_size: int = 10):
        """
        Initialize performance cache.

        Args:
            sample_size: Number of recent samples to track per model
        """
        self.sample_size = sample_size
        # Store recent stats for each model
        self._stats: dict[str, deque[CompletionStats]] = defaultdict(
            lambda: deque(maxlen=sample_size)
        )

    def record_inference(self, model_id: str, stats: CompletionStats) -> None:
        """
        Record inference statistics for a model.

        Args:
            model_id: Model identifier
            stats: Completion statistics from LM Studio
        """
        if stats:
            self._stats[model_id].append(stats)
            logger.debug(
                f"Recorded stats for {model_id}: "
                f"{stats.tokens_per_second:.1f} t/s, "
                f"TTFT: {stats.time_to_first_token:.3f}s"
                if stats.tokens_per_second and stats.time_to_first_token
                else f"Recorded stats for {model_id}"
            )

    def get_metrics(self, model_id: str) -> PerformanceMetrics:
        """
        Get aggregated performance metrics for a model.

        Args:
            model_id: Model identifier

        Returns:
            PerformanceMetrics with averaged values
        """
        samples = self._stats.get(model_id, deque())
        if not samples:
            return PerformanceMetrics()

        # Calculate averages
        tokens_per_sec = [s.tokens_per_second for s in samples if s.tokens_per_second]
        time_to_first = [s.time_to_first_token for s in samples if s.time_to_first_token]
        gen_time = [s.generation_time for s in samples if s.generation_time]

        return PerformanceMetrics(
            avg_tokens_per_second=sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else None,
            avg_time_to_first_token=sum(time_to_first) / len(time_to_first) if time_to_first else None,
            avg_generation_time=sum(gen_time) / len(gen_time) if gen_time else None,
            sample_count=len(samples),
        )

    def get_all_metrics(self) -> dict[str, PerformanceMetrics]:
        """Get metrics for all tracked models."""
        return {model_id: self.get_metrics(model_id) for model_id in self._stats.keys()}

    def clear(self, model_id: Optional[str] = None) -> None:
        """
        Clear performance data.

        Args:
            model_id: Specific model to clear, or None to clear all
        """
        if model_id:
            if model_id in self._stats:
                del self._stats[model_id]
                logger.debug(f"Cleared stats for {model_id}")
        else:
            self._stats.clear()
            logger.debug("Cleared all performance stats")


# Global performance cache instance
_performance_cache: Optional[PerformanceCache] = None


def get_performance_cache() -> PerformanceCache:
    """Get or create global performance cache instance."""
    global _performance_cache
    if _performance_cache is None:
        from ..config.settings import get_settings

        settings = get_settings()
        _performance_cache = PerformanceCache(
            sample_size=settings.performance_tracking.sample_size
        )
    return _performance_cache
