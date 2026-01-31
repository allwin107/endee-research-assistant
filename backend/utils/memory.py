"""
Memory Optimization Utilities
Provides tools for monitoring and optimizing memory usage
"""

import gc
import psutil
import os
import structlog
import time
from typing import Dict, Any, Generator, Iterable, TypeVar, List
from threading import Thread, Event

from backend.utils.monitoring import get_metrics_collector

logger = structlog.get_logger(__name__)

T = TypeVar("T")

class MemoryOptimizer:
    """
    Manages memory usage through proactive garbage collection
    and monitoring
    """
    
    def __init__(self, threshold_mb: int = 1024, check_interval: int = 60):
        """
        Args:
            threshold_mb: Memory threshold in MB to trigger cleanup
            check_interval: Interval in seconds to check memory
        """
        self.threshold_mb = threshold_mb
        self.check_interval = check_interval
        self._stop_event = Event()
        self._monitor_thread = None
        self.metrics = get_metrics_collector()
        
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
            
        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True, name="MemoryMonitor")
        self._monitor_thread.start()
        logger.info("memory_monitor_started", threshold_mb=self.threshold_mb)

    def stop_monitoring(self):
        """Stop background monitoring"""
        if self._monitor_thread:
            self._stop_event.set()
            self._monitor_thread.join(timeout=2.0)
            logger.info("memory_monitor_stopped")

    def _monitor_loop(self):
        """Background loop to check memory usage"""
        process = psutil.Process(os.getpid())
        
        while not self._stop_event.is_set():
            try:
                mem_info = process.memory_info()
                rss_mb = mem_info.rss / 1024 / 1024
                
                # Record metric
                self.metrics.set_gauge("memory_rss_mb", rss_mb)
                
                # Check threshold
                if rss_mb > self.threshold_mb:
                    logger.warning("memory_threshold_exceeded", current=rss_mb, limit=self.threshold_mb)
                    self.optimize()
                    
            except Exception as e:
                logger.error("memory_monitor_error", error=str(e))
                
            time.sleep(self.check_interval)

    def optimize(self):
        """Force garbage collection and clear caches"""
        before_mb = self.get_current_usage()
        
        # 1. Force GC
        gc.collect()
        
        # 2. Clear known caches (if integrated)
        # TODO: Clear custom service caches if needed (e.g. embedding cache)
        
        after_mb = self.get_current_usage()
        freed = before_mb - after_mb
        
        logger.info("memory_optimization_complete", freed_mb=freed, current_mb=after_mb)
        
        # Alert if still high
        if after_mb > self.threshold_mb:
            logger.error("critical_memory_usage", current_mb=after_mb)

    def get_current_usage(self) -> float:
        """Get current RSS memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
        
    @staticmethod
    def chunk_generator(data: Iterable[T], chunk_size: int = 32) -> Generator[List[T], None, None]:
        """
        Yields chunks from an iterable to process large datasets without
        loading everything into memory.
        """
        chunk = []
        for item in data:
            chunk.append(item)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

# Global instance
_memory_optimizer = None

def get_memory_optimizer() -> MemoryOptimizer:
    global _memory_optimizer
    if _memory_optimizer is None:
        # Default 2GB limit for now
        _memory_optimizer = MemoryOptimizer(threshold_mb=2048)
    return _memory_optimizer
