"""BurstGPT request generator / profiler for LLM serving frameworks.

A thin, testable benchmark harness that replays BurstGPT traces (or a
synthetic gamma/zipf workload) against an OpenAI-compatible serving
backend such as vLLM or LightLLM.
"""

from burstgpt.config import RunConfig

__all__ = ["RunConfig"]
__version__ = "0.2.0"
