import collections, logging, os, statistics, time
import requests
from typing import Optional

logger = logging.getLogger("anomaly")

class SLOAnomalyDetector:
    def __init__(self, slo_ms=1000.0, window=200, min_samples=50, k=3.0,
                 cooldown=300, webhook_url:Optional[str]=None, dry_run=False, enabled=True):
        self.slo = slo_ms; self.window = window; self.min_samples = min_samples
        self.k = k; self.cooldown = cooldown; self.url = webhook_url
        self.dry_run = dry_run; self.enabled = enabled
        self.latencies = collections.deque(maxlen=window)
        self.last_alert = 0.0

    def observe(self, latency_ms: float) -> bool:
        if not self.enabled: return False
        self.latencies.append(latency_ms)
        if len(self.latencies) < self.min_samples: return False
        mean = statistics.mean(self.latencies)
        stdev = statistics.stdev(self.latencies, mean) if len(self.latencies)>1 else 0.0
        if latency_ms > self.slo and latency_ms > mean + self.k * stdev:
            now = time.time()
            if now - self.last_alert < self.cooldown: return False
            self.last_alert = now
            msg = (f"*Anomaly*: {latency_ms:.0f}ms | rolling mean {mean:.0f}ms ± {self.k}σ={self.k*stdev:.0f}ms")
            self._alert(msg)
            return True
        return False

    def _alert(self, msg):
        if self.dry_run or not self.url:
            logger.info(f"DRY-RUN: {msg}")
            return
        try:
            requests.post(self.url, json={"text": msg}, timeout=5)
        except Exception as e:
            logger.error(f"alert failed: {e}")