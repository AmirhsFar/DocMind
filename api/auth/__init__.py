"""Auth package — JWT-based signup/login for DocMind.

Exposes `router` so main.py can include it with a single import.
"""

from api.auth.router import router

__all__ = ["router"]
