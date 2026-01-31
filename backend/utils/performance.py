"""
Performance Monitoring Utilities
Track query times, memory usage, and identify slow operations
"""

import time
import functools
import psutil
import structlog
from typing import Callable, Any
from contextlib import contextmanager

logger = structlog.get_logger()

# Performance thresholds
SLOW_QUERY_THRESHOLD_MS = 1000  # 1 second
MEMORY_WARNING_THRESHOLD_MB = 500


def performance_monitor(operation_name: str = None):
    """
    Decorator to monitor function performance
    
    Tracks:
    - Execution time
    - Memory usage
    - Logs slow queries
    
    Args:
        operation_name: Name of the operation (defaults to function name)
    
    Usage:
        @performance_monitor("semantic_search")
        def search_papers(query):
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Get initial memory
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Start timer
            start_time = time.time()
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Calculate metrics
                duration_ms = (time.time() - start_time) * 1000
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                mem_delta = mem_after - mem_before
                
                # Log performance
                log_data = {
                    "operation": op_name,
                    "duration_ms": round(duration_ms, 2),
                    "memory_mb": round(mem_after, 2),
                    "memory_delta_mb": round(mem_delta, 2),
                }
                
                # Warn on slow queries
                if duration_ms > SLOW_QUERY_THRESHOLD_MS:
                    logger.warning("slow_query_detected", **log_data)
                else:
                    logger.info("operation_complete", **log_data)
                
                # Warn on high memory usage
                if mem_delta > MEMORY_WARNING_THRESHOLD_MB:
                    logger.warning("high_memory_usage", **log_data)
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    "operation_failed",
                    operation=op_name,
                    duration_ms=round(duration_ms, 2),
                    error=str(e)
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Get initial memory
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Start timer
            start_time = time.time()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate metrics
                duration_ms = (time.time() - start_time) * 1000
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                mem_delta = mem_after - mem_before
                
                # Log performance
                log_data = {
                    "operation": op_name,
                    "duration_ms": round(duration_ms, 2),
                    "memory_mb": round(mem_after, 2),
                    "memory_delta_mb": round(mem_delta, 2),
                }
                
                # Warn on slow queries
                if duration_ms > SLOW_QUERY_THRESHOLD_MS:
                    logger.warning("slow_query_detected", **log_data)
                else:
                    logger.info("operation_complete", **log_data)
                
                # Warn on high memory usage
                if mem_delta > MEMORY_WARNING_THRESHOLD_MB:
                    logger.warning("high_memory_usage", **log_data)
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    "operation_failed",
                    operation=op_name,
                    duration_ms=round(duration_ms, 2),
                    error=str(e)
                )
                raise
        
        # Return appropriate wrapper based on function type
        if functools.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@contextmanager
def track_time(operation_name: str):
    """
    Context manager to track execution time
    
    Usage:
        with track_time("database_query"):
            # Your code here
            pass
    """
    start_time = time.time()
    
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        
        log_data = {
            "operation": operation_name,
            "duration_ms": round(duration_ms, 2)
        }
        
        if duration_ms > SLOW_QUERY_THRESHOLD_MS:
            logger.warning("slow_operation", **log_data)
        else:
            logger.debug("operation_timed", **log_data)


class PerformanceMetrics:
    """
    Collect and aggregate performance metrics
    """
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "slow_queries": 0,
            "total_time_ms": 0,
            "avg_time_ms": 0,
            "max_time_ms": 0,
            "min_time_ms": float('inf'),
        }
    
    def record_query(self, duration_ms: float):
        """Record a query execution time"""
        self.metrics["total_queries"] += 1
        self.metrics["total_time_ms"] += duration_ms
        
        if duration_ms > SLOW_QUERY_THRESHOLD_MS:
            self.metrics["slow_queries"] += 1
        
        if duration_ms > self.metrics["max_time_ms"]:
            self.metrics["max_time_ms"] = duration_ms
        
        if duration_ms < self.metrics["min_time_ms"]:
            self.metrics["min_time_ms"] = duration_ms
        
        # Update average
        self.metrics["avg_time_ms"] = (
            self.metrics["total_time_ms"] / self.metrics["total_queries"]
        )
    
    def get_metrics(self) -> dict:
        """Get current metrics"""
        return self.metrics.copy()
    
    def reset(self):
        """Reset all metrics"""
        self.__init__()


# Global metrics instance
_global_metrics = PerformanceMetrics()


def get_performance_metrics() -> dict:
    """Get global performance metrics"""
    return _global_metrics.get_metrics()


def record_query_time(duration_ms: float):
    """Record a query time in global metrics"""
    _global_metrics.record_query(duration_ms)


def get_system_stats() -> dict:
    """
    Get current system resource usage
    
    Returns:
        Dictionary with CPU, memory, and disk stats
    """
    process = psutil.Process()
    
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "memory_percent": process.memory_percent(),
        "threads": process.num_threads(),
        "open_files": len(process.open_files()),
    }
