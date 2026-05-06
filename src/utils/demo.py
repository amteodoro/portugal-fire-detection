"""Demo-mode helpers.

`DEMO_MODE=true` (the public-deployment default) freezes the app to read
from `data/snapshot/` instead of any external API or database. These
helpers let routers and services check the flag and short-circuit.
"""
from fastapi import HTTPException

from src.utils.config import settings


def is_demo() -> bool:
    """Return True when the app is running in public-demo mode."""
    return settings.DEMO_MODE


def demo_blocked(detail: str = "Disabled in demo mode") -> None:
    """Raise 403 if running in demo mode. Use as a FastAPI dependency or
    inline guard at the top of a handler."""
    if is_demo():
        raise HTTPException(status_code=403, detail=detail)
