import logging
import logging.handlers
import os
import time
from functools import wraps
from pathlib import Path

from src.config import DATA_DIR

LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"
_configured = False


def _configure_root() -> None:
    """Rotating-file handlers on the root logger."""
    global _configured
    if _configured:
        return

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    have_stream = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.handlers.RotatingFileHandler)
        for h in root.handlers
    )
    if not have_stream:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

    have_file = any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        and Path(getattr(h, "baseFilename", "")).name == "app.log"
        for h in root.handlers
    )
    if not have_file:
        file_handler = logging.handlers.RotatingFileHandler(
            LOGS_DIR / "app.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    if level != "DEBUG":
        for noisy in ("httpx", "httpcore", "urllib3", "langsmith"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with the shared configuration applied."""
    _configure_root()
    return logging.getLogger(name)


# ═══════════════════════════════════════════════════════════════════════════════
# @log_call decorator
# ═══════════════════════════════════════════════════════════════════════════════

def _preview(obj, limit: int = 80) -> str:
    """Short repr for log output — truncates long values so log lines stay readable."""
    try:
        s = repr(obj)
    except Exception:
        s = f"<{type(obj).__name__}>"
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def log_call(fn=None, *, level: str = "INFO", preview: int = 80):
    """Decorator: log function entry (with arg preview), exit, and failure."""
    def decorate(func):
        logger = get_logger(func.__module__)
        log_at = getattr(logger, level.lower(), logger.info)

        @wraps(func)
        def wrapper(*args, **kwargs):
            qual = func.__qualname__
            args_s = ", ".join(_preview(a, preview) for a in args)
            kw_s = ", ".join(f"{k}={_preview(v, preview)}" for k, v in kwargs.items())
            sig = ", ".join(part for part in (args_s, kw_s) if part)
            log_at("call %s(%s)", qual, sig)

            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                dt = time.perf_counter() - t0
                logger.error("fail %s after %.2fs: %s", qual, dt, exc)
                raise
            dt = time.perf_counter() - t0
            log_at("done %s in %.2fs", qual, dt)
            return result

        return wrapper

    if fn is not None and callable(fn):
        return decorate(fn)
    return decorate
