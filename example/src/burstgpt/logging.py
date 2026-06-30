"""Structured logging in JSON Lines format.

The original ``Logger`` repeatedly did ``f.write("\\n"); json.dump(...)``,
producing a file that is *not* valid JSON (multiple concatenated objects),
and multiple coroutines wrote to the same file with no lock, interleaving
output. Writing one JSON object per line (JSONL) is append-friendly,
streamable, and an ``asyncio.Lock`` serialises concurrent writers.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict


class JsonlLogger:
    """Append-only JSON Lines writer, safe for concurrent coroutines."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._lock = asyncio.Lock()
        # Truncate any previous run so we don't append to stale data.
        open(self.log_path, "w").close()

    async def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record)
        async with self._lock:
            # File I/O is fast relative to network calls; doing it under the
            # lock keeps lines from interleaving without a thread pool.
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def write_sync(self, record: Dict[str, Any]) -> None:
        """Synchronous write for metadata emitted outside the event loop."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
