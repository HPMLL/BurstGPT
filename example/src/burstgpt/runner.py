"""Profiling runner: schedules and issues queries against a backend.

Replaces the old ``ServerOnline``/``ServerBase`` God object. Construction
no longer does CSV parsing, dataset loading, and logger setup all at once;
those concerns now live in :mod:`burstgpt.workload`,
:mod:`burstgpt.prompts`, and :mod:`burstgpt.logging` respectively.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from datetime import datetime

import aiohttp
import numpy as np

from burstgpt.backends import build_backend
from burstgpt.config import RunConfig
from burstgpt.logging import JsonlLogger
from burstgpt.prompts import PromptMatcher, load_prompt_pool
from burstgpt.workload import build_workload


class ProfileRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.matcher = PromptMatcher(load_prompt_pool(config.data_path))
        self.logger = JsonlLogger(config.log_path)
        self.detail_logger = JsonlLogger(config.detail_log_path)
        self.backend = build_backend(config, self.detail_logger)
        self.workload = build_workload(
            config, count=config.surplus_prompts_num, rng=self.rng
        )

    async def _issue(self) -> None:
        cfg = self.config
        # Cap at surplus_prompts_num so trace replay and synthetic modes both
        # honour the requested query budget.
        queries = itertools.islice(iter(self.workload), cfg.surplus_prompts_num)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=4 * 60 * 60)
        ) as session:
            tasks = []
            for event_id, query in enumerate(queries):
                prompt = self.matcher.match(query.prompt_len, query.output_len)
                tasks.append(
                    asyncio.create_task(
                        self._fire(session, query, prompt, event_id)
                    )
                )
            await asyncio.gather(*tasks)

    async def _fire(self, session, query, prompt, event_id) -> None:
        # Sleep until this query's scheduled arrival, then issue it.
        await asyncio.sleep(query.arrival_time)
        print(f"[INFO] start {event_id} after {query.arrival_time:.3f}s")
        await self.backend.call(
            session,
            prompt.text,
            prompt_len=query.prompt_len,
            output_len=query.output_len,
            event_id=event_id,
        )

    def run(self) -> None:
        start = time.perf_counter()
        asyncio.run(self._issue())
        elapsed = time.perf_counter() - start
        self._save_summary(elapsed)

    def _save_summary(self, elapsed: float) -> None:
        print("[INFO] saving run summary")
        self.logger.write_sync(
            {
                "model_path": self.config.model_path,
                "backend": self.config.backend,
                "config": vars(self.config),
                "elapsed_s": round(elapsed, 3),
                "log_time": datetime.now().isoformat(),
            }
        )
