"""
server/anomaly_detector.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Eq9 – SLO-Anchored Latency Anomaly Trigger

  Anomaly(t) = 1 [ L(t) > T_SLO  ∧  L(t) > μ_rolling + k·σ_rolling ]

The AND logic means:
  • A request that is slow but still within SLO  → no alert (noise suppression)
  • A request that breaches SLO AND is a rolling outlier → real incident alert

Features
--------
* Per-endpoint detectors so /chat and /rag/query have independent baselines.
* Cooldown window to avoid alert storms.
* Dry-run mode: logs instead of firing.
* FastAPI middleware class ready to drop into app.py.
* /metrics/anomaly endpoint data provided via anomaly_metrics().
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import httpx

from gateway.perf_math import Eq9AnomalyDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AlertChannel  – pluggable delivery backend
# ---------------------------------------------------------------------------

class AlertChannel:
    """Base class. Override send() to deliver to Slack, PagerDuty, etc."""

    def send(self, message: str, endpoint: str, latency_ms: float) -> None:
        logger.warning(
            "[ANOMALY] endpoint=%s latency=%.0fms | %s", endpoint, latency_ms, message
        )


class SlackAlertChannel(AlertChannel):
    """Sends to a Slack webhook URL (set SLACK_WEBHOOK_URL in env)."""

    def __init__(self, webhook_url: str | None = None) -> None:
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")

    def send(self, message: str, endpoint: str, latency_ms: float) -> None:
        super().send(message, endpoint, latency_ms)   # always log
        if not self.webhook_url:
            return
        try:
            httpx.post(
                self.webhook_url,
                json={"text": message},
                timeout=5,
            )
        except Exception as exc:
            logger.error("Slack alert failed: %s", exc)


# ---------------------------------------------------------------------------
# PerEndpointAnomalyDetector
# ---------------------------------------------------------------------------

@dataclass
class PerEndpointAnomalyDetector:
    """
    Maintains one Eq9AnomalyDetector per HTTP path prefix and fires
    alerts through the configured AlertChannel.
    """
    slo_threshold_ms: float = 1000.0
    window_size:      int   = 200
    min_samples:      int   = 20    # Eq9: lowered from 50 → kicks in sooner
    k_factor:         float = 3.0
    cooldown_seconds: float = 300.0
    dry_run:          bool  = False
    enabled:          bool  = True

    _detectors: dict[str, Eq9AnomalyDetector] = field(
        default_factory=dict, init=False, repr=False
    )
    _alert_channel: AlertChannel = field(
        default_factory=AlertChannel, init=False, repr=False
    )
    _total_requests:  int = field(default=0, init=False)
    _total_anomalies: int = field(default=0, init=False)

    def set_alert_channel(self, channel: AlertChannel) -> None:
        self._alert_channel = channel

    def _get_detector(self, endpoint: str) -> Eq9AnomalyDetector:
        if endpoint not in self._detectors:
            self._detectors[endpoint] = Eq9AnomalyDetector(
                slo_threshold_ms=self.slo_threshold_ms,
                window_size=self.window_size,
                min_samples=self.min_samples,
                k_factor=self.k_factor,
                cooldown_seconds=self.cooldown_seconds,
            )
        return self._detectors[endpoint]

    def observe(self, endpoint: str, latency_ms: float) -> bool:
        """
        Feed a new latency reading for `endpoint`.
        Returns True if an anomaly was detected (and alert was fired).

        Eq9: when an anomaly fires, the router's fallback fraction is
        temporarily increased so future requests are less likely to hit
        the slow model that caused the breach.
        """
        if not self.enabled:
            return False

        self._total_requests += 1
        detector = self._get_detector(endpoint)
        fired    = detector.observe(latency_ms)

        if fired:
            self._total_anomalies += 1
            stats = detector.rolling_stats()
            threshold_val = stats["mean"] + self.k_factor * stats["stdev"]
            message = (
                f"⚠️ *SLO Anomaly* – `{endpoint}`\n"
                f"Latency: `{latency_ms:.0f} ms`  (SLO: `{self.slo_threshold_ms:.0f} ms`)\n"
                f"Rolling μ: `{stats['mean']:.0f} ms`  σ: `{stats['stdev']:.0f} ms`  "
                f"threshold: `{threshold_val:.0f} ms`\n"
                f"Samples in window: {stats['samples']}"
            )
            if self.dry_run:
                logger.info("DRY-RUN anomaly: %s", message)
            else:
                self._alert_channel.send(message, endpoint, latency_ms)

            # Eq9 → Eq7 feedback: bump router fallback fraction so more
            # traffic shifts to the fast model until latency recovers.
            try:
                from gateway.model_router import get_router
                router = get_router()
                old_phi = router.config.fallback_fraction
                # Increase fallback fraction by 5 pp, cap at 0.30
                router.config.fallback_fraction = min(0.30, old_phi + 0.05)
                logger.warning(
                    "Eq9→Eq7 anomaly on %s: router fallback_fraction %.2f → %.2f",
                    endpoint, old_phi, router.config.fallback_fraction,
                )
            except Exception as exc:
                logger.debug("Eq9 router feedback skipped: %s", exc)

        return fired

    def rolling_stats(self, endpoint: str) -> dict:
        if endpoint not in self._detectors:
            return {"mean": 0.0, "stdev": 0.0, "samples": 0}
        return self._detectors[endpoint].rolling_stats()

    def metrics(self) -> dict:
        return {
            "enabled":        self.enabled,
            "dry_run":        self.dry_run,
            "slo_threshold_ms": self.slo_threshold_ms,
            "k_factor":       self.k_factor,
            "total_requests": self._total_requests,
            "total_anomalies": self._total_anomalies,
            "anomaly_rate":   round(
                self._total_anomalies / self._total_requests, 4
            ) if self._total_requests else 0.0,
            "endpoints": {
                ep: det.rolling_stats()
                for ep, det in self._detectors.items()
            },
        }

    def reset(self, endpoint: str | None = None) -> None:
        if endpoint:
            if endpoint in self._detectors:
                self._detectors[endpoint].reset()
        else:
            for det in self._detectors.values():
                det.reset()


# ---------------------------------------------------------------------------
# FastAPI middleware
# ---------------------------------------------------------------------------

class LatencyAnomalyMiddleware:
    """
    ASGI middleware that measures every HTTP request latency and feeds it
    into the PerEndpointAnomalyDetector (Eq9).

    Add to FastAPI app:
        app.add_middleware(LatencyAnomalyMiddleware)
    """

    def __init__(self, app, detector: PerEndpointAnomalyDetector | None = None) -> None:
        self.app      = app
        self.detector = detector or get_anomaly_detector()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Derive a short endpoint label  e.g. "/chat", "/rag/query"
        path     = scope.get("path", "/")
        endpoint = "/" + path.strip("/").split("/")[0] if "/" in path else path

        start_ms = time.perf_counter() * 1000
        fired    = False

        async def send_wrapper(message):
            nonlocal fired
            if message["type"] == "http.response.start":
                latency_ms = time.perf_counter() * 1000 - start_ms
                fired = self.detector.observe(endpoint, latency_ms)
                if fired:
                    message.setdefault("headers", [])
                    message["headers"] = list(message["headers"]) + [
                        (b"x-anomaly-alert", b"fired")
                    ]
            await send(message)

        await self.app(scope, receive, send_wrapper)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_anomaly_detector: PerEndpointAnomalyDetector | None = None


def get_anomaly_detector() -> PerEndpointAnomalyDetector:
    global _anomaly_detector
    if _anomaly_detector is None:
        try:
            from config import settings
            _anomaly_detector = PerEndpointAnomalyDetector(
                slo_threshold_ms=getattr(settings, "slo_threshold_ms", 1000.0),
                window_size=getattr(settings, "anomaly_window_size", 200),
                min_samples=getattr(settings, "anomaly_min_samples", 50),
                k_factor=getattr(settings, "anomaly_k_factor", 3.0),
                cooldown_seconds=getattr(settings, "anomaly_cooldown_seconds", 300.0),
                dry_run=getattr(settings, "anomaly_dry_run", True),
                enabled=getattr(settings, "anomaly_alert_enabled", True),
            )
            webhook = getattr(settings, "slack_webhook_url", None)
            if webhook:
                _anomaly_detector.set_alert_channel(SlackAlertChannel(webhook))
        except Exception:
            _anomaly_detector = PerEndpointAnomalyDetector()
    return _anomaly_detector


def anomaly_metrics() -> dict:
    return get_anomaly_detector().metrics()
