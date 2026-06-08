from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

import networkx as nx

from app.analysis.root_cause import CausalGraph


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_TO_FLOAT: dict[Severity, float] = {
    Severity.LOW: 0.5,
    Severity.MEDIUM: 1.0,
    Severity.HIGH: 2.0,
    Severity.CRITICAL: 3.0,
}


class Channel(str, Enum):
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class AnomalyInput:
    metric: str
    timestamps: list[datetime]
    scores: list[float]
    causal_graph: CausalGraph | None = None


@dataclass
class AlertEvent:
    id: str
    root_metric: str
    related_metrics: list[str]
    start_time: datetime
    end_time: datetime
    severity: Severity
    suppressed: bool
    suppression_reason: str | None
    channel: Channel
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatorConfig:
    merge_window_minutes: int = 5
    cooldown_minutes: int = 30
    flicker_threshold: int = 3
    webhook_url: str | None = None
    default_channel: Channel = Channel.CONSOLE


class AlertAggregator:
    def __init__(self, config: AggregatorConfig | None = None) -> None:
        self.config = config or AggregatorConfig()
        self._cooldown_tracker: dict[str, datetime] = {}
        self._causal_graph: nx.DiGraph | None = None

    def set_causal_graph(self, graph: nx.DiGraph) -> None:
        self._causal_graph = graph

    def process(self, anomalies: list[AnomalyInput]) -> list[AlertEvent]:
        if not anomalies:
            return []

        merged = self._merge_related(anomalies)
        events = self._build_events(merged)
        events = self._assess_noise(events)
        events = self._apply_dependency_suppression(events)
        events = self._apply_cooldown_suppression(events)

        for event in events:
            if not event.suppressed:
                self._send(event)

        return events

    def _merge_related(self, anomalies: list[AnomalyInput]) -> list[list[AnomalyInput]]:
        window = timedelta(minutes=self.config.merge_window_minutes)
        sorted_anomalies = sorted(anomalies, key=lambda a: min(a.timestamps))
        groups: list[list[AnomalyInput]] = []

        for anomaly in sorted_anomalies:
            anomaly_start = min(anomaly.timestamps)
            merged = False
            for group in groups:
                group_start = min(min(a.timestamps) for a in group)
                if abs(anomaly_start - group_start) <= window and self._are_causally_related(group, anomaly):
                    group.append(anomaly)
                    merged = True
                    break
            if not merged:
                groups.append([anomaly])

        return groups

    def _are_causally_related(self, group: list[AnomalyInput], anomaly: AnomalyInput) -> bool:
        group_metrics = {a.metric for a in group}
        target = anomaly.metric

        if target in group_metrics:
            return True

        causal_graphs: list[CausalGraph] = []
        if anomaly.causal_graph is not None:
            causal_graphs.append(anomaly.causal_graph)
        for member in group:
            if member.causal_graph is not None:
                causal_graphs.append(member.causal_graph)

        for cg in causal_graphs:
            for cause, effect, _ in cg.edges:
                if cause in group_metrics and effect == target:
                    return True
                if cause == target and effect in group_metrics:
                    return True

        if self._causal_graph is not None:
            for member_metric in group_metrics:
                if self._causal_graph.has_edge(member_metric, target):
                    return True
                if self._causal_graph.has_edge(target, member_metric):
                    return True

        return False

    def _build_events(self, groups: list[list[AnomalyInput]]) -> list[AlertEvent]:
        events: list[AlertEvent] = []
        for group in groups:
            all_timestamps: list[datetime] = []
            all_metrics: list[str] = []
            root_metric = group[0].metric
            causal_graph: CausalGraph | None = None

            for anomaly in group:
                all_metrics.append(anomaly.metric)
                all_timestamps.extend(anomaly.timestamps)
                if anomaly.causal_graph is not None and anomaly.causal_graph.root_cause is not None:
                    root_metric = anomaly.causal_graph.root_cause
                    causal_graph = anomaly.causal_graph

            start_time = min(all_timestamps)
            end_time = max(all_timestamps)
            unique_related = list(dict.fromkeys(m for m in all_metrics if m != root_metric))

            score_map: dict[str, float] = {}
            max_score = 0.0
            for anomaly in group:
                if anomaly.scores:
                    peak = max(anomaly.scores)
                    score_map[anomaly.metric] = peak
                    if peak > max_score:
                        max_score = peak

            severity = self._score_to_severity(max_score)

            details: dict[str, Any] = {
                "score_map": score_map,
                "anomaly_count": sum(len(a.timestamps) for a in group),
            }
            if causal_graph is not None:
                details["causal_edges"] = [(e[0], e[1]) for e in causal_graph.edges]

            events.append(AlertEvent(
                id=str(uuid4()),
                root_metric=root_metric,
                related_metrics=unique_related,
                start_time=start_time,
                end_time=end_time,
                severity=severity,
                suppressed=False,
                suppression_reason=None,
                channel=self.config.default_channel,
                details=details,
            ))

        return events

    def _score_to_severity(self, score: float) -> Severity:
        if score >= 3.0:
            return Severity.CRITICAL
        if score >= 2.0:
            return Severity.HIGH
        if score >= 1.0:
            return Severity.MEDIUM
        return Severity.LOW

    def _assess_noise(self, events: list[AlertEvent]) -> list[AlertEvent]:
        for event in events:
            point_count = event.details.get("anomaly_count", 0)
            if point_count <= self.config.flicker_threshold:
                event.severity = Severity.LOW
                event.details["noise_assessment"] = "flicker"
            else:
                if event.severity in (Severity.LOW, Severity.MEDIUM):
                    event.severity = Severity.HIGH
                event.details["noise_assessment"] = "sustained"
        return events

    def _apply_dependency_suppression(self, events: list[AlertEvent]) -> list[AlertEvent]:
        if self._causal_graph is None:
            return events

        anomalous_roots: set[str] = set()
        for event in events:
            if not event.suppressed:
                anomalous_roots.add(event.root_metric)

        downstream_cache: dict[str, set[str]] = {}
        for metric in anomalous_roots:
            downstream_cache[metric] = self._get_all_downstream(metric)

        for event in events:
            if event.suppressed:
                continue
            for root, downstream in downstream_cache.items():
                if event.root_metric in downstream:
                    event.suppressed = True
                    event.suppression_reason = f"dependency_suppressed_by_{root}"
                    event.details["suppressed_by"] = root
                    break

        return events

    def _get_all_downstream(self, metric: str) -> set[str]:
        if self._causal_graph is None:
            return set()
        downstream: set[str] = set()
        visited: set[str] = set()
        stack = [metric]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for successor in self._causal_graph.successors(current):
                downstream.add(successor)
                stack.append(successor)
        return downstream

    def _apply_cooldown_suppression(self, events: list[AlertEvent]) -> list[AlertEvent]:
        cooldown = timedelta(minutes=self.config.cooldown_minutes)
        now = datetime.now()
        for event in events:
            if event.suppressed:
                continue
            key = event.root_metric
            if key in self._cooldown_tracker:
                last_alert_time = self._cooldown_tracker[key]
                if event.start_time - last_alert_time < cooldown:
                    event.suppressed = True
                    event.suppression_reason = f"cooldown_suppressed_{key}"
                    event.details["cooldown_since"] = last_alert_time.isoformat()
                    continue
            self._cooldown_tracker[key] = event.start_time

        expired = [k for k, v in self._cooldown_tracker.items() if now - v > cooldown]
        for k in expired:
            del self._cooldown_tracker[k]

        return events

    def _send(self, event: AlertEvent) -> None:
        if event.channel == Channel.WEBHOOK:
            self._send_webhook(event)
        elif event.channel == Channel.CONSOLE:
            self._send_console(event)

    def _send_webhook(self, event: AlertEvent) -> None:
        import json
        import urllib.request

        if not self.config.webhook_url:
            return

        payload = json.dumps({
            "id": event.id,
            "root_metric": event.root_metric,
            "related_metrics": event.related_metrics,
            "start_time": event.start_time.isoformat(),
            "end_time": event.end_time.isoformat(),
            "severity": event.severity.value,
            "details": event.details,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.config.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
        except Exception:
            pass

    def _send_console(self, event: AlertEvent) -> None:
        print(
            f"[ALERT] {event.severity.value.upper()} | "
            f"root={event.root_metric} | "
            f"related={event.related_metrics} | "
            f"window={event.start_time.isoformat()} -> {event.end_time.isoformat()} | "
            f"details={event.details}"
        )
