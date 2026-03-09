from __future__ import annotations

from fastapi import HTTPException

from app.services.command_runner import SecurityPolicyError


def http_from_security_error(exc: SecurityPolicyError) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details or {},
            }
        },
    )
