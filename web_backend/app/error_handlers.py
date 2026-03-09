from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

_STATUS_TO_CODE = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    500: "INTERNAL_ERROR",
}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        payload = _normalize_http_exception(exc)
        return JSONResponse(status_code=exc.status_code, content={"error": payload})

    @app.exception_handler(StarletteHTTPException)
    async def _starlette_http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        payload = _normalize_starlette_http_exception(exc)
        return JSONResponse(status_code=exc.status_code, content={"error": payload})

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "request validation failed",
                    "details": {"errors": exc.errors()},
                }
            },
        )

    @app.exception_handler(Exception)
    async def _unexpected_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "unexpected backend error",
                    "details": {"exception": str(exc)},
                }
            },
        )


def _normalize_http_exception(exc: HTTPException) -> dict[str, Any]:
    detail = exc.detail
    if isinstance(detail, dict):
        if "error" in detail and isinstance(detail["error"], dict):
            return _normalize_error_dict(detail["error"], exc.status_code)
        return _normalize_error_dict(detail, exc.status_code)
    if isinstance(detail, str):
        return {
            "code": _STATUS_TO_CODE.get(exc.status_code, "HTTP_ERROR"),
            "message": detail,
            "details": {},
        }
    return {
        "code": _STATUS_TO_CODE.get(exc.status_code, "HTTP_ERROR"),
        "message": "http error",
        "details": {"detail": detail},
    }


def _normalize_error_dict(raw: dict[str, Any], status_code: int) -> dict[str, Any]:
    out = {
        "code": str(raw.get("code") or _STATUS_TO_CODE.get(status_code, "HTTP_ERROR")),
        "message": str(raw.get("message") or "http error"),
    }
    if "hint" in raw and raw.get("hint") is not None:
        out["hint"] = raw.get("hint")
    if "details" in raw and raw.get("details") is not None:
        out["details"] = raw.get("details")
    elif raw:
        passthrough = {k: v for k, v in raw.items() if k not in {"code", "message", "hint"}}
        if passthrough:
            out["details"] = passthrough
    return out


def _normalize_starlette_http_exception(exc: StarletteHTTPException) -> dict[str, Any]:
    detail = exc.detail
    if isinstance(detail, dict):
        if "error" in detail and isinstance(detail["error"], dict):
            return _normalize_error_dict(detail["error"], exc.status_code)
        return _normalize_error_dict(detail, exc.status_code)
    if isinstance(detail, str):
        return {
            "code": _STATUS_TO_CODE.get(exc.status_code, "HTTP_ERROR"),
            "message": detail,
            "details": {},
        }
    return {
        "code": _STATUS_TO_CODE.get(exc.status_code, "HTTP_ERROR"),
        "message": "http error",
        "details": {"detail": detail},
    }
