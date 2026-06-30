"""Inference backends: HTTP clients for OpenAI-compatible serving frameworks."""

from __future__ import annotations

import abc
import json
import time
from typing import Any, Dict

import aiohttp

from burstgpt.config import RunConfig
from burstgpt.logging import JsonlLogger

# Requests can be long-running for big generations; allow a generous timeout.
_TIMEOUT = aiohttp.ClientTimeout(total=4 * 60 * 60)


class InferenceBackend(abc.ABC):
    """Issues a single inference request and records its latency."""

    def __init__(self, config: RunConfig, logger: JsonlLogger):
        self.config = config
        self.logger = logger

    @abc.abstractmethod
    async def call(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        prompt_len: int,
        output_len: int,
        event_id: int,
    ) -> None:
        raise NotImplementedError

    @property
    def _base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/generate"


class VllmBackend(InferenceBackend):
    async def call(self, session, prompt, prompt_len, output_len, event_id) -> None:
        cfg = self.config
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "stream": cfg.stream,
            "ignore_eos": cfg.ignore_eos,
            "max_tokens": int(output_len),
            "temperature": cfg.temperature,
        }
        first_chunk_time = 0.0
        start = time.perf_counter()
        async with session.post(self._base_url, json=payload) as resp:
            if resp.status != 200:
                print(f"[ERROR] {resp.status} {resp.reason}: {await resp.text()}")
                return

            if cfg.stream:
                output, first_chunk_time = await _read_null_delimited_stream(resp, start)
            else:
                output = await resp.json()
        total_time = time.perf_counter() - start

        await self.logger.write(
            {
                "event_id": event_id,
                "out_len": len(output["text"][0]),
                "out_len_expected": int(output_len),
                "in_len": int(prompt_len),
                "first_chunk_time": first_chunk_time,
                "total_chunk_time": total_time,
                "record_time": time.perf_counter(),
            }
        )


class LightllmBackend(InferenceBackend):
    """LightLLM backend. Streaming only (asserted in RunConfig validation)."""

    async def call(self, session, prompt, prompt_len, output_len, event_id) -> None:
        cfg = self.config
        payload = {
            "inputs": prompt,
            "parameters": {
                "do_sample": cfg.do_sample,
                "ignore_eos": cfg.ignore_eos,
                "max_new_tokens": int(output_len),
                "temperature": cfg.temperature,
            },
        }
        start = time.perf_counter()
        async with session.post(self._base_url, json=payload) as resp:
            if resp.status != 200:
                print(f"[ERROR] {resp.status} {resp.reason}: {await resp.text()}")
                return
            output, first_chunk_time = await _read_chunked_stream(resp, start)
        total_time = time.perf_counter() - start

        await self.logger.write(
            {
                "event_id": event_id,
                "in_len": int(prompt_len),
                "out_len_expected": int(output_len),
                "first_chunk_time": first_chunk_time,
                "total_chunk_time": total_time,
                "record_time": time.perf_counter(),
            }
        )


async def _read_null_delimited_stream(resp, start):
    """vLLM streams null-separated JSON objects; keep the last complete one."""
    buffer = b""
    json_str = b""
    first_chunk_time = 0.0
    first = True
    async for chunk in resp.content.iter_any():
        if first:
            first_chunk_time = time.perf_counter() - start
            first = False
        buffer += chunk
        while b"\0" in buffer:
            json_str, buffer = buffer.split(b"\0", 1)
    return json.loads(json_str.decode("utf-8")), first_chunk_time


async def _read_chunked_stream(resp, start):
    chunks = []
    first_chunk_time = 0.0
    first = True
    async for chunk, _ in resp.content.iter_chunks():
        if first:
            first_chunk_time = time.perf_counter() - start
            first = False
        chunks.append(chunk)
    output = json.loads(b"".join(chunks).decode("utf-8"))
    return output, first_chunk_time


def build_backend(config: RunConfig, logger: JsonlLogger) -> InferenceBackend:
    if config.backend == "vllm":
        return VllmBackend(config, logger)
    if config.backend == "lightllm":
        return LightllmBackend(config, logger)
    raise ValueError(f"unknown backend {config.backend!r}")
