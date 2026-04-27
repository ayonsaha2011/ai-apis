from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "openai_backend_requests_total",
    "Total API requests handled by endpoint and model.",
    ["endpoint", "model", "status"],
)
REQUEST_LATENCY = Histogram(
    "openai_backend_request_seconds",
    "API request latency by endpoint and model.",
    ["endpoint", "model"],
)
MODEL_LOADS = Counter(
    "openai_backend_model_loads_total",
    "Model load attempts by model and status.",
    ["model", "status"],
)
MODEL_STATE = Gauge(
    "openai_backend_model_state",
    "Model state gauge: configured=0, loading=1, ready=2, failed=3, disabled=4.",
    ["model"],
)
ACTIVE_INFERENCES = Gauge(
    "openai_backend_active_inferences",
    "Currently active inferences by model.",
    ["model"],
)

STATE_VALUES = {"configured": 0, "loading": 1, "ready": 2, "failed": 3, "disabled": 4}
