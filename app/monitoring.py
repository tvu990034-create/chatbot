"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

"""
Monitoring and metrics collection for Cloud AI Chatbot
Provides performance metrics, health checks, and observability features
"""

import time
import threading
from typing import Dict, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, field
import os


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates performance metrics for monitoring
    Thread-safe implementation for production use
    """
    
    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
        
    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value for a histogram metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)
            # Keep only last 1000 values to prevent memory bloat
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value"""
        with self._lock:
            key = self._make_key(name, labels)
            return self._counters.get(key, 0.0)
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value"""
        with self._lock:
            key = self._make_key(name, labels)
            return self._gauges.get(key)
    
    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics"""
        with self._lock:
            key = self._make_key(name, labels)
            values = self._histograms.get(key, [])
            if not values:
                return {}
            
            sorted_values = sorted(values)
            n = len(sorted_values)
            return {
                "count": n,
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "mean": sum(sorted_values) / n,
                "p50": sorted_values[int(n * 0.5)],
                "p95": sorted_values[int(n * 0.95)],
                "p99": sorted_values[int(n * 0.99)],
            }
    
    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for a metric with labels"""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Export counters
        for key, value in self._counters.items():
            lines.append(f"# TYPE {key} counter")
            lines.append(f"{key} {value}")
        
        # Export gauges
        for key, value in self._gauges.items():
            lines.append(f"# TYPE {key} gauge")
            lines.append(f"{key} {value}")
        
        # Export histograms
        for key, values in self._histograms.items():
            if values:
                stats = self.get_histogram_stats(key)
                lines.append(f"# TYPE {key} histogram")
                lines.append(f"{key}_count {stats['count']}")
                lines.append(f"{key}_sum {stats['mean'] * stats['count']}")
                lines.append(f"{key}_bucket {{le=\"0.5\"}} {stats['p50']}")
                lines.append(f"{key}_bucket {{le=\"0.95\"}} {stats['p95']}")
                lines.append(f"{key}_bucket {{le=\"0.99\"}} {stats['p99']}")
                lines.append(f"{key}_bucket {{le=\"+Inf\"}} {stats['max']}")
        
        return "\n".join(lines)


class PerformanceMonitor:
    """
    High-level performance monitoring for chatbot operations
    Tracks TTFB, total response time, cache hit rate, etc.
    """
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self._enabled = os.getenv("MONITORING_ENABLED", "false").lower() == "true"
    
    def record_request(self, ttfb_ms: float, total_ms: float, cache_hit: bool):
        """Record a request's performance metrics"""
        if not self._enabled:
            return
        
        self.metrics.observe_histogram("chat_ttfb_ms", ttfb_ms)
        self.metrics.observe_histogram("chat_total_ms", total_ms)
        self.metrics.increment("chat_requests_total")
        if cache_hit:
            self.metrics.increment("chat_cache_hits")
        else:
            self.metrics.increment("chat_cache_misses")
    
    def record_retrieval(self, latency_ms: float, results_count: int):
        """Record retrieval performance"""
        if not self._enabled:
            return
        
        self.metrics.observe_histogram("retrieval_latency_ms", latency_ms)
        self.metrics.observe_histogram("retrieval_results_count", results_count)
    
    def record_inference(self, latency_ms: float, tokens_generated: int):
        """Record inference performance"""
        if not self._enabled:
            return
        
        self.metrics.observe_histogram("inference_latency_ms", latency_ms)
        self.metrics.observe_histogram("inference_tokens_generated", tokens_generated)
        self.metrics.increment("inference_tokens_total", tokens_generated)
    
    def get_cache_hit_rate(self) -> float:
        """Calculate current cache hit rate"""
        hits = self.metrics.get_counter("chat_cache_hits")
        misses = self.metrics.get_counter("chat_cache_misses")
        total = hits + misses
        return hits / total if total > 0 else 0.0
    
    def get_average_ttfb(self) -> Optional[float]:
        """Get average TTFB"""
        stats = self.metrics.get_histogram_stats("chat_ttfb_ms")
        return stats.get("mean")
    
    def get_average_total_time(self) -> Optional[float]:
        """Get average total response time"""
        stats = self.metrics.get_histogram_stats("chat_total_ms")
        return stats.get("mean")


# Global metrics instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


class RequestContext:
    """
    Context manager for tracking request performance
    Usage:
        with RequestContext("chat") as ctx:
            # ... process request ...
            ctx.record_ttfb(50.0)
            ctx.record_total(100.0)
            ctx.record_cache_hit(True)
    """
    
    def __init__(self, operation: str):
        self.operation = operation
        self.start_time = time.perf_counter()
        self.monitor = get_monitor()
        self.ttfb_time: Optional[float] = None
        self.cache_hit = False
    
    def record_ttfb(self):
        """Record time to first byte"""
        if self.ttfb_time is None:
            self.ttfb_time = (time.perf_counter() - self.start_time) * 1000
    
    def record_cache_hit(self, hit: bool):
        """Record cache hit/miss"""
        self.cache_hit = hit
    
    def record_total(self):
        """Record total request time"""
        total_time = (time.perf_counter() - self.start_time) * 1000
        if self.ttfb_time is None:
            self.ttfb_time = total_time
        
        if self.operation == "chat":
            self.monitor.record_request(self.ttfb_time, total_time, self.cache_hit)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.record_total()
        return False
