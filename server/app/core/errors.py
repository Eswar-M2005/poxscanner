"""Structured JSON error helpers matching docs/contracts.md."""

from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class APIError(Exception):
    """Application error serialized as ``{"error": {"code", "message"}}``."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_body(code, message))


async def api_error_handler(_request: Request, exc: APIError) -> JSONResponse:
    return error_response(exc.status_code, exc.code, exc.message)


async def validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", ()) if item != "body")
        msg = str(err.get("msg", "Invalid value"))
        parts.append(f"{loc}: {msg}" if loc else msg)
    message = "; ".join(parts) if parts else "Request validation failed."
    return error_response(422, "validation_error", message)
