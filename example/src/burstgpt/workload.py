"""Workload generators.

Each generator yields a stream of :class:`Query` objects describing *when*
a request should be sent and *what* shape (prompt / output token lengths)
it should have. The actual prompt text is chosen later by the prompt
matcher, keeping "how much load" cleanly separate from "what text".

The original code packed trace replay and synthetic gamma/zipf generation
into a single ``_Query.get_query()`` method branching on ``trace is None``.
That method referenced ``self.gamma_step`` / ``self.shape_list`` /
``self.scale_list`` which were never initialised (synthetic mode crashed),
and called ``len(self.trace)`` even when ``trace`` was ``None``. Splitting
into two subclasses removes the branch and those bugs.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from burstgpt.config import RunConfig


@dataclass
class Query:
    """A single request to be issued at ``arrival_time`` (seconds from start)."""

    arrival_time: float
    prompt_len: int
    output_len: int


class WorkloadGenerator(abc.ABC):
    """Produces a stream of :class:`Query` objects."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Query]:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of queries this generator will yield (may be unbounded)."""
        raise NotImplementedError


class TraceWorkload(WorkloadGenerator):
    """Replays a BurstGPT CSV trace, optionally scaling the request rate."""

    def __init__(self, config: RunConfig):
        self._config = config
        self._frame = self._load(config)

    @staticmethod
    def _load(config: RunConfig) -> pd.DataFrame:
        # Stream the CSV in chunks so we only read ``prompt_num`` rows of a
        # multi-GB file. The original code did the same with a hard-coded
        # chunksize=20; keep the streaming idea but read in larger chunks.
        chunks = []
        loaded = 0
        for chunk in pd.read_csv(config.burstgpt_path, chunksize=10_000):
            chunks.append(chunk)
            loaded += len(chunk)
            if loaded >= config.prompt_num:
                break
        if not chunks:
            raise ValueError(f"no rows read from {config.burstgpt_path}")

        frame = pd.concat(chunks, ignore_index=True)
        frame = frame.head(config.prompt_num).copy()
        # Scale the request rate: scale=100 means 100x faster (timestamps
        # compressed towards 0).
        frame["Timestamp"] = frame["Timestamp"] / config.scale
        return frame

    def __len__(self) -> int:
        return len(self._frame)

    def __iter__(self) -> Iterator[Query]:
        cfg = self._config
        prev_ts = 0.0
        for ts, p_len, o_len in zip(
            self._frame["Timestamp"],
            self._frame["Request tokens"],
            self._frame["Response tokens"],
        ):
            arrival = float(ts) - prev_ts
            prev_ts = float(ts)
            yield Query(
                arrival_time=max(arrival, 0.0),
                prompt_len=min(int(p_len), cfg.max_prompt_len - 1),
                output_len=min(int(o_len), cfg.max_gen_len - 1),
            )


class SyntheticWorkload(WorkloadGenerator):
    """Synthetic workload: gamma inter-arrival times, zipf token lengths.

    Inter-arrival times are drawn from a Gamma distribution and prompt /
    output lengths from a Zipf distribution -- the modelled-scaling path
    described in the BurstGPT paper. Unlike the original, all parameters
    are initialised up front so this mode actually runs.
    """

    def __init__(self, config: RunConfig, count: int, rng: np.random.Generator):
        self._config = config
        self._count = count
        self._rng = rng

    def __len__(self) -> int:
        return self._count

    def _sample_len(self, limit: int) -> int:
        val = int(self._rng.zipf(a=self._config.zipf_param))
        while val >= limit:
            val = int(self._rng.zipf(a=self._config.zipf_param))
        return val

    def __iter__(self) -> Iterator[Query]:
        cfg = self._config
        for _ in range(self._count):
            delta = float(self._rng.gamma(cfg.gamma_shape, cfg.gamma_scale))
            yield Query(
                arrival_time=delta,
                prompt_len=self._sample_len(cfg.max_prompt_len),
                output_len=self._sample_len(cfg.max_gen_len),
            )


def build_workload(config: RunConfig, count: int, rng: np.random.Generator) -> WorkloadGenerator:
    """Factory selecting the generator implied by ``config``."""
    if config.use_burstgpt:
        return TraceWorkload(config)
    return SyntheticWorkload(config, count=count, rng=rng)
