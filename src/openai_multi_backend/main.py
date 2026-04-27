from __future__ import annotations

import time
import uuid
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles

from openai_multi_backend.api.health import router as health_router
from openai_multi_backend.api.openai_routes import router as openai_router
from openai_multi_backend.config import get_settings
from openai_multi_backend.errors import (
    OpenAIHTTPException,
    generic_exception_handler,
    openai_http_exception_handler,
    validation_exception_handler,
)
from openai_multi_backend.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    settings.prepare_directories()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="OpenAI Multi-Backend API",
        version="0.1.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
    )
    app.add_exception_handler(OpenAIHTTPException, openai_http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, generic_exception_handler)
    app.middleware("http")(request_id_middleware)
    app.include_router(health_router)
    app.include_router(openai_router)
    app.mount("/generated", StaticFiles(directory=settings.output_dir), name="generated")
    return app


async def request_id_middleware(
    request: Request, call_next: Callable[[Request], object]
) -> Response:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    start = time.monotonic()
    response = await call_next(request)  # type: ignore[misc]
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = f"{(time.monotonic() - start) * 1000:.2f}"
    return response


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "openai_multi_backend.main:create_app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        factory=True,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
