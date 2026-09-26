"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def record(self, *, blocked: bool, layer: str | None) -> None:
        """Update counters after one request has finished."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
        if layer == "rate_limiter":
            self.rate_limit_hits += 1

    def record_judge(self, *, safe: bool) -> None:
        self.judge_checks += 1
        if not safe:
            self.judge_fails += 1

    def check_metrics(self) -> list[Alert]:
        """Compute rates and append an Alert for each threshold exceeded.

        A metric that is already alerting is not re-added, so calling this
        after every request does not flood the list.
        """
        snap = self.snapshot()
        checks = [
            ("block_rate", snap["block_rate"], self.block_rate_threshold,
             "Block rate above threshold — possible attack campaign or over-blocking."),
            ("rate_limit_hits", self.rate_limit_hits, self.rate_limit_hit_threshold,
             "Many rate-limit hits — possible flooding / cost attack."),
            ("judge_fail_rate", snap["judge_fail_rate"], self.judge_fail_rate_threshold,
             "LLM judge rejects many replies — model output quality/safety degraded."),
        ]
        active = {a.metric for a in self.alerts}
        new_alerts = []
        for metric, value, threshold, message in checks:
            if value > threshold and metric not in active:
                alert = Alert(metric=metric, value=value, threshold=threshold, message=message)
                self.alerts.append(alert)
                new_alerts.append(alert)
        return new_alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.snapshot()
        data["thresholds"] = {
            "block_rate": self.block_rate_threshold,
            "rate_limit_hits": self.rate_limit_hit_threshold,
            "judge_fail_rate": self.judge_fail_rate_threshold,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
