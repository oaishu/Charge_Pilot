"""Consistent API error envelope (same contract as the main backend).

Every error response has the shape:
    {"error": {"code": "...", "message": "...", "details": {}}}
"""
import re
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

_UPPER_CODE = re.compile(r"^[A-Z][A-Z0-9_]+$")


class AppError(Exception):
    """Domain-level error carrying a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


def error_payload(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def install_error_handlers(app) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
        if _UPPER_CODE.match(detail):
            code = detail
        elif exc.status_code == status.HTTP_401_UNAUTHORIZED:
            code = "UNAUTHORIZED"
        elif exc.status_code == status.HTTP_403_FORBIDDEN:
            code = "FORBIDDEN"
        elif exc.status_code == status.HTTP_404_NOT_FOUND:
            code = "NOT_FOUND"
        else:
            code = "HTTP_ERROR"
        headers = getattr(exc, "headers", None)
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(code, detail),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        # sanitize pydantic v2 error dicts (they can hold non-JSON-serializable ctx)
        safe_errors = []
        for err in exc.errors():
            cleaned = {k: v for k, v in err.items() if k not in ("ctx", "input", "url")}
            if err.get("ctx") is not None:
                cleaned["ctx"] = {
                    k: (str(v) if isinstance(v, Exception) else v)
                    for k, v in err["ctx"].items()
                }
            safe_errors.append(cleaned)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_payload(
                "VALIDATION_ERROR",
                "Request validation failed",
                {"errors": safe_errors},
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        import logging

        logging.getLogger(__name__).exception("Unhandled API error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_payload("INTERNAL_ERROR", "An unexpected error occurred"),
        )
