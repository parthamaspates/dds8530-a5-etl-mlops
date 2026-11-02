"""Shared logging: one format for console and logs/pipeline.log, plus a timer
that records how long each pipeline step takes."""
import logging
import time
from contextlib import contextmanager

from config import LOG_DIR

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for handler in (logging.StreamHandler(), logging.FileHandler(LOG_DIR / "pipeline.log")):
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


@contextmanager
def timed_step(logger: logging.Logger, step: str, timings: dict):
    """Log the start, end and duration of a step; log and re-raise any error."""
    logger.info("START %s", step)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("FAILED %s", step)
        raise
    finally:
        timings[step] = round(time.perf_counter() - t0, 3)
    logger.info("END %s (%.2f s)", step, timings[step])
