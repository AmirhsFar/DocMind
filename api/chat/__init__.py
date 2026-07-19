"""Chat package — RAG chat sessions, messages, and the retrieval pipeline.

Exposes `router` so main.py can include it with a single import.
"""

from api.chat.router import router

__all__ = ["router"]
