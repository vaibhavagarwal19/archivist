"""Per-request LLM telemetry.

A single chat request fans out into several model calls (intent, retrieval
decision, planning, the final answer). Without collecting usage across them
there is no way to answer "what did that query cost and where did the time
go", which makes tuning guesswork.

Metrics are held in a ContextVar rather than on the service, because the agent
is a process-wide singleton and concurrent requests must not share a counter.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# Groq pricing, USD per million tokens. Used for an indicative cost only.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "llama-3.1-8b-instant": (0.05, 0.08),
}
DEFAULT_PRICING = (0.15, 0.75)


@dataclass
class CallRecord:
    label: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0


@dataclass
class RequestMetrics:
    model: str = ""
    calls: list[CallRecord] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def llm_seconds(self) -> float:
        return sum(c.duration_s for c in self.calls)

    def estimated_cost_usd(self) -> float:
        in_rate, out_rate = MODEL_PRICING.get(self.model, DEFAULT_PRICING)
        return (self.input_tokens * in_rate + self.output_tokens * out_rate) / 1_000_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "llm_calls": len(self.calls),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_seconds": round(self.llm_seconds, 3),
            "estimated_cost_usd": round(self.estimated_cost_usd(), 6),
            "breakdown": [
                {
                    "step": c.label,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "seconds": round(c.duration_s, 3),
                }
                for c in self.calls
            ],
        }


_current: ContextVar[RequestMetrics | None] = ContextVar("request_metrics", default=None)


@contextmanager
def collect(model: str = ""):
    """Collect telemetry for everything inside this block."""
    metrics = RequestMetrics(model=model)
    token = _current.set(metrics)
    try:
        yield metrics
    finally:
        _current.reset(token)


def current() -> RequestMetrics | None:
    return _current.get()


@contextmanager
def record(label: str):
    """Time one model call and add its token usage to the active request.

    Yields a setter the caller hands the model response to; usage metadata is
    read from it. Outside a collect() block this is a no-op, so services stay
    usable in tests and scripts.
    """
    started = time.perf_counter()
    holder: dict[str, Any] = {}

    def capture(response: Any) -> Any:
        holder["response"] = response
        return response

    try:
        yield capture
    finally:
        metrics = _current.get()
        if metrics is not None:
            usage = getattr(holder.get("response"), "usage_metadata", None) or {}
            metrics.calls.append(
                CallRecord(
                    label=label,
                    input_tokens=int(usage.get("input_tokens", 0) or 0),
                    output_tokens=int(usage.get("output_tokens", 0) or 0),
                    duration_s=time.perf_counter() - started,
                )
            )
