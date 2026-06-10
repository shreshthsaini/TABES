"""Structured logging and per-step trace writing.

Two channels:
  * human-readable logs via the stdlib ``logging`` module (console + optional file)
  * machine-readable JSONL traces via :class:`TraceWriter` — one record per
    denoising step (entropies, candidate sets, TIS scores, selections, timings)
    so decoding trajectories can be replayed and analyzed offline.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", logfile: str | os.PathLike | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(level=getattr(logging, level.upper()), format=_FORMAT,
                        handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):  # torch tensors / numpy arrays
        return obj.tolist()
    if hasattr(obj, "item") and not isinstance(obj, (int, float, str, bool)):
        return obj.item()
    return obj


class TraceWriter:
    """Append-only JSONL writer for decoding / training traces."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)

    def write(self, record: dict[str, Any]) -> None:
        record = {"ts": time.time(), **record}
        self._fh.write(json.dumps(_to_jsonable(record)) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Timer:
    """Context manager measuring wall-clock seconds into ``.elapsed``."""

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0
