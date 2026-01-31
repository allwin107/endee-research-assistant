"""
Prometheus metrics integration
Exposes /metrics endpoint for Prometheus scraping
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
)
from fastapi import APIRouter, Response
from typing import Dict, Any
import time

from backend.utils.monitoring import get_metrics_collector, get_system_monitor

# Create custom registry
registry = CollectorRegistry()

# Define metrics

# Info metric
app_info = Info(
    "app",
    "Application information",
    registry=registry
)
app_info.info({
    "name": "ai_research_assistant",
    "version": "1.0.0",
    "environment": "production"
})

# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=registry
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry
)

# Query metrics
query_duration_seconds = Histogram(
    "query_duration_seconds",
    "Query duration in seconds",
    ["query_type"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    registry=registry
)

query_total = Counter(
    "query_total",
    "Total queries",
    ["query_type"],
    registry=registry
)

query_errors_total = Counter(
    "query_errors_total",
    "Total query errors",
    ["query_type", "error_type"],
    registry=registry
)

# Cache metrics
cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["cache_type"],
    registry=registry
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["cache_type"],
    registry=registry
)

cache_size = Gauge(
    "cache_size",
    "Current cache size",
    ["cache_type"],
    registry=registry
)

# LLM metrics
llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens used",
    ["model", "operation"],
    registry=registry
)

llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["model"],
    registry=registry
)

# System metrics
system_cpu_usage = Gauge(
    "system_cpu_usage_percent",
    "CPU usage percentage",
    registry=registry
)

system_memory_usage = Gauge(
    "system_memory_usage_mb",
    "Memory usage in MB",
    registry=registry
)

system_disk_usage = Gauge(
    "system_disk_usage_percent",
    "Disk usage percentage",
    registry=registry
)

system_thread_count = Gauge(
    "system_thread_count",
    "Number of threads",
    registry=registry
)

# Database metrics
db_connections_active = Gauge(
    "db_connections_active",
    "Active database connections",
    registry=registry
)

# Endee metrics
endee_search_duration_seconds = Histogram(
    "endee_search_duration_seconds",
    "Endee search duration in seconds",
    ["index"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    registry=registry
)

endee_operations_total = Counter(
    "endee_operations_total",
    "Total Endee operations",
    ["operation", "index"],
    registry=registry
)


# Router
router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    
    Returns:
        Metrics in Prometheus format
    """
    # Update system metrics
    system_monitor = get_system_monitor()
    system_metrics = system_monitor.collect_system_metrics()
    
    system_cpu_usage.set(system_metrics["cpu_percent"])
    system_memory_usage.set(system_metrics["memory_mb"])
    system_disk_usage.set(system_metrics["disk_percent"])
    system_thread_count.set(system_metrics["thread_count"])
    
    # Update cache metrics from internal collector
    metrics_collector = get_metrics_collector()
    app_metrics = metrics_collector.get_all_metrics()
    
    # Sync internal metrics to Prometheus
    for metric_name, value in app_metrics.get("gauges", {}).items():
        if "cache_size" in metric_name:
            cache_type = metric_name.split("{")[1].split("}")[0].split("=")[1] if "{" in metric_name else "default"
            cache_size.labels(cache_type=cache_type).set(value)
    
    # Generate Prometheus format
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


# Helper functions for instrumenting code

class PrometheusInstrumentor:
    """Helper class to instrument code with Prometheus metrics"""
    
    @staticmethod
    def track_request(method: str, endpoint: str, status: int, duration: float):
        """Track HTTP request"""
        http_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
    
    @staticmethod
    def track_query(query_type: str, duration: float, error: str = None):
        """Track query execution"""
        query_total.labels(query_type=query_type).inc()
        query_duration_seconds.labels(query_type=query_type).observe(duration)
        
        if error:
            query_errors_total.labels(query_type=query_type, error_type=error).inc()
    
    @staticmethod
    def track_cache_access(cache_type: str, hit: bool):
        """Track cache access"""
        if hit:
            cache_hits_total.labels(cache_type=cache_type).inc()
        else:
            cache_misses_total.labels(cache_type=cache_type).inc()
    
    @staticmethod
    def track_llm_usage(model: str, operation: str, tokens: int):
        """Track LLM usage"""
        llm_requests_total.labels(model=model).inc()
        llm_tokens_total.labels(model=model, operation=operation).inc(tokens)
    
    @staticmethod
    def track_endee_operation(operation: str, index: str, duration: float):
        """Track Endee operation"""
        endee_operations_total.labels(operation=operation, index=index).inc()
        if operation == "search":
            endee_search_duration_seconds.labels(index=index).observe(duration)


# Export instrumentor
instrumentor = PrometheusInstrumentor()
