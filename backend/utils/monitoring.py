"""
Production-ready monitoring and metrics collection
Provides performance tracking, error tracking, and usage analytics
"""

import time
import psutil
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Centralized metrics collection
    Tracks counters, gauges, and histograms
    """

    def __init__(self):
        self._lock = Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._timeseries: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._start_time = datetime.utcnow()

    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric
        
        Args:
            name: Metric name
            value: Increment value
            labels: Optional labels
        """
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
            logger.debug("counter_incremented", metric=name, value=value, labels=labels)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels
        """
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
            logger.debug("gauge_set", metric=name, value=value, labels=labels)

    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Observe a value for histogram metric
        
        Args:
            name: Metric name
            value: Observed value
            labels: Optional labels
        """
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)
            logger.debug("histogram_observed", metric=name, value=value, labels=labels)

    def record_timeseries(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record a time-series data point
        
        Args:
            name: Metric name
            value: Value
            labels: Optional labels
        """
        with self._lock:
            key = self._make_key(name, labels)
            point = MetricPoint(timestamp=datetime.utcnow(), value=value, labels=labels or {})
            self._timeseries[key].append(point)
            
            # Keep only last 24 hours
            cutoff = datetime.utcnow() - timedelta(hours=24)
            self._timeseries[key] = [p for p in self._timeseries[key] if p.timestamp > cutoff]

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value"""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value"""
        key = self._make_key(name, labels)
        return self._gauges.get(key)

    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """
        Get histogram statistics
        
        Returns:
            Dictionary with min, max, avg, p50, p95, p99
        """
        key = self._make_key(name, labels)
        values = list(self._histograms.get(key, []))
        
        if not values:
            return {}

        sorted_values = sorted(values)
        count = len(sorted_values)

        return {
            "count": count,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.50)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)] if count >= 100 else sorted_values[-1],
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: self.get_histogram_stats(name)
                    for name in self._histograms.keys()
                },
                "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
            }

    def reset(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timeseries.clear()
            self._start_time = datetime.utcnow()

    @staticmethod
    def _make_key(name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create metric key from name and labels"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    return _metrics_collector


class PerformanceTracker:
    """Track performance metrics for operations"""

    def __init__(self, operation: str, labels: Optional[Dict[str, str]] = None):
        self.operation = operation
        self.labels = labels or {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.metrics = get_metrics_collector()

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000

        # Record metrics
        self.metrics.observe_histogram(
            f"{self.operation}_duration_ms",
            duration_ms,
            self.labels
        )
        self.metrics.increment_counter(
            f"{self.operation}_total",
            labels=self.labels
        )

        if exc_type is not None:
            self.metrics.increment_counter(
                f"{self.operation}_errors",
                labels={**self.labels, "error_type": exc_type.__name__}
            )

        logger.info(
            f"{self.operation}_completed",
            duration_ms=duration_ms,
            success=exc_type is None,
            **self.labels
        )

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000


class ErrorTracker:
    """Track errors and exceptions"""

    def __init__(self):
        self.metrics = get_metrics_collector()

    def track_error(
        self,
        error: Exception,
        context: str,
        severity: str = "error",
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track an error
        
        Args:
            error: Exception instance
            context: Context where error occurred
            severity: Error severity (error, warning, critical)
            extra: Additional context
        """
        error_type = type(error).__name__
        
        # Increment error counter
        self.metrics.increment_counter(
            "errors_total",
            labels={
                "context": context,
                "error_type": error_type,
                "severity": severity,
            }
        )

        # Log error
        logger.error(
            "error_tracked",
            context=context,
            error_type=error_type,
            error_message=str(error),
            severity=severity,
            **(extra or {})
        )


class UsageAnalytics:
    """Track usage analytics"""

    def __init__(self):
        self.metrics = get_metrics_collector()

    def track_query(
        self,
        query_type: str,
        user_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        result_count: Optional[int] = None,
    ) -> None:
        """
        Track a query
        
        Args:
            query_type: Type of query (search, rag, recommendation)
            user_id: Optional user ID
            duration_ms: Query duration
            result_count: Number of results
        """
        labels = {"query_type": query_type}
        if user_id:
            labels["user_id"] = user_id

        self.metrics.increment_counter("queries_total", labels=labels)

        if duration_ms is not None:
            self.metrics.observe_histogram(
                "query_duration_ms",
                duration_ms,
                labels=labels
            )

        if result_count is not None:
            self.metrics.observe_histogram(
                "query_results",
                result_count,
                labels=labels
            )

    def track_cache_access(self, hit: bool, cache_type: str = "default") -> None:
        """
        Track cache access
        
        Args:
            hit: Whether cache hit or miss
            cache_type: Type of cache
        """
        labels = {"cache_type": cache_type, "result": "hit" if hit else "miss"}
        self.metrics.increment_counter("cache_accesses_total", labels=labels)

    def track_token_usage(self, tokens: int, model: str, operation: str) -> None:
        """
        Track LLM token usage
        
        Args:
            tokens: Number of tokens used
            model: Model name
            operation: Operation type
        """
        labels = {"model": model, "operation": operation}
        self.metrics.increment_counter("tokens_used_total", value=tokens, labels=labels)
        self.metrics.observe_histogram("tokens_per_request", tokens, labels=labels)


class SystemMonitor:
    """Monitor system resources"""

    def __init__(self):
        self.metrics = get_metrics_collector()
        self.process = psutil.Process(os.getpid())

    def collect_system_metrics(self) -> Dict[str, float]:
        """
        Collect current system metrics
        
        Returns:
            Dictionary of system metrics
        """
        # CPU usage
        cpu_percent = self.process.cpu_percent(interval=0.1)
        self.metrics.set_gauge("cpu_usage_percent", cpu_percent)

        # Memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        self.metrics.set_gauge("memory_usage_mb", memory_mb)

        # Disk usage
        disk_usage = psutil.disk_usage("/")
        disk_percent = disk_usage.percent
        self.metrics.set_gauge("disk_usage_percent", disk_percent)

        # Thread count
        thread_count = self.process.num_threads()
        self.metrics.set_gauge("thread_count", thread_count)

        # Open file descriptors (Unix only)
        try:
            fd_count = self.process.num_fds()
            self.metrics.set_gauge("open_file_descriptors", fd_count)
        except AttributeError:
            fd_count = None

        return {
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "disk_percent": disk_percent,
            "thread_count": thread_count,
            "open_fds": fd_count,
        }


# Global instances
_error_tracker = ErrorTracker()
_usage_analytics = UsageAnalytics()
_system_monitor = SystemMonitor()


def get_error_tracker() -> ErrorTracker:
    """Get global error tracker instance"""
    return _error_tracker


def get_usage_analytics() -> UsageAnalytics:
    """Get global usage analytics instance"""
    return _usage_analytics


def get_system_monitor() -> SystemMonitor:
    """Get global system monitor instance"""
    return _system_monitor
