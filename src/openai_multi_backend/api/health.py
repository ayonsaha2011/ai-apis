from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from openai_multi_backend.api.schemas import ReadinessResponse
from openai_multi_backend.models.registry import ModelRegistry, get_registry

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", response_model=ReadinessResponse)
async def readyz(registry: ModelRegistry = Depends(get_registry)) -> ReadinessResponse:
    states = registry.health()
    status: Literal["ready", "degraded"] = (
        "degraded" if any(item.get("state") == "failed" for item in states.values()) else "ready"
    )
    return ReadinessResponse(status=status, models=states)


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
