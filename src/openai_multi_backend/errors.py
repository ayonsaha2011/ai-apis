from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class OpenAIHTTPException(HTTPException):
    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code


def openai_error_payload(
    message: str,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "param": param, "code": code}}


async def openai_http_exception_handler(
    _request: Request, exc: OpenAIHTTPException
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=openai_error_payload(exc.message, exc.error_type, exc.param, exc.code),
    )


async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    loc = first.get("loc", [])
    param = ".".join(str(part) for part in loc if part not in {"body", "query", "path"}) or None
    message = first.get("msg", "Invalid request")
    return JSONResponse(
        status_code=422,
        content=openai_error_payload(str(message), param=param, code="validation_error"),
    )


async def generic_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=openai_error_payload(
            str(exc), error_type="server_error", code="internal_server_error"
        ),
    )
