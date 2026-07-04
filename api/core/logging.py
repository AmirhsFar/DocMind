import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once at startup.

    This is deliberately plain for Phase 0. In Phase 6 (hardening) we'll swap
    this for structlog with JSON output, request IDs, etc.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    # Quiet down noisy third-party loggers.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
