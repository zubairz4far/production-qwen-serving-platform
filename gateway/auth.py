from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, status

from gateway.config import Settings


def require_api_key(request: Request, settings: Settings) -> None:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    if not any(secrets.compare_digest(token, expected) for expected in settings.public_api_key_set):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
