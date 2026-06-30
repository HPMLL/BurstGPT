"""Prompt pool and shape matcher.

Given a desired ``(prompt_len, output_len)`` from the workload, we need to
pick a real prompt of approximately that shape from the prompt pool
(e.g. ShareGPT). The original implementation precomputed a dense
``max_prompt_len x max_gen_len`` (1024x1024) int32 lookup table and filled
gaps with ~80 lines of hand-rolled nearest-neighbour interpolation -- slow
to build and very hard to follow.

This module does the same nearest-neighbour selection with a sorted index:
prompts are grouped into buckets by prompt length, and within the closest
bucket we pick the entry whose output length is closest. O(log n) per query
and no giant matrix.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Prompt:
    text: str
    prompt_len: int
    output_len: int


def load_prompt_pool(path: str) -> List[Prompt]:
    """Load prompts from the preprocessed JSON file.

    Expected format (produced by ``scripts/preprocess_sharegpt.py``)::

        {"0": {"prompt": ..., "len_prompt": int, "len_output": int, "output": ...}, ...}
    """
    with open(path, "r") as f:
        raw: Dict[str, dict] = json.load(f)

    prompts: List[Prompt] = []
    for entry in raw.values():
        # Skip incomplete entries (some preprocessing runs only fill prompts).
        if "len_prompt" not in entry or "len_output" not in entry:
            continue
        prompts.append(
            Prompt(
                text=entry["prompt"],
                prompt_len=int(entry["len_prompt"]),
                output_len=int(entry["len_output"]),
            )
        )
    if not prompts:
        raise ValueError(f"no usable prompts loaded from {path}")
    return prompts


class PromptMatcher:
    """Nearest-neighbour prompt selection by ``(prompt_len, output_len)``."""

    def __init__(self, prompts: List[Prompt]):
        if not prompts:
            raise ValueError("prompt pool is empty")
        # Sort by prompt length so we can binary-search the closest bucket,
        # then group entries that share a prompt length.
        self._prompts = sorted(prompts, key=lambda p: p.prompt_len)
        self._prompt_lens = [p.prompt_len for p in self._prompts]
        self._by_len: Dict[int, List[Prompt]] = {}
        for p in self._prompts:
            self._by_len.setdefault(p.prompt_len, []).append(p)
        # Sort each bucket by output length for nearest-output lookup.
        for bucket in self._by_len.values():
            bucket.sort(key=lambda p: p.output_len)
        self._sorted_lens = sorted(self._by_len)

    def match(self, prompt_len: int, output_len: int) -> Prompt:
        """Return the pool prompt closest to the requested shape."""
        closest_len = self._closest(self._sorted_lens, prompt_len)
        bucket = self._by_len[closest_len]
        out_lens = [p.output_len for p in bucket]
        idx = self._closest_index(out_lens, output_len)
        return bucket[idx]

    @staticmethod
    def _closest(values: List[int], target: int) -> int:
        return values[PromptMatcher._closest_index(values, target)]

    @staticmethod
    def _closest_index(values: List[int], target: int) -> int:
        pos = bisect.bisect_left(values, target)
        if pos == 0:
            return 0
        if pos == len(values):
            return len(values) - 1
        before, after = values[pos - 1], values[pos]
        return pos if (after - target) < (target - before) else pos - 1
