"""Typed run configuration.

Replaces the old ``Config`` object that simply held a handful of loosely
typed dicts (``server_config['stream']`` etc.). Using a dataclass gives us
defaults, validation, and -- importantly -- stops fields from being read
out of the wrong sub-dict (the original code stored ``scale`` /
``burstgpt_path`` in ``prompt_config`` but read them from ``server_config``,
so they were always ``None``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunConfig:
    # --- prompt pool / tokenizer ------------------------------------------
    data_path: str = "preprocess_data/shareGPT.json"
    model_path: Optional[str] = None
    seed: int = 0

    # --- workload ---------------------------------------------------------
    use_burstgpt: bool = False
    burstgpt_path: str = "../data/BurstGPT_1.csv"
    conv_or_api: str = "conv"  # "conv" | "api"
    scale: float = 1.0  # RPS scaling for trace replay; >1 means faster
    prompt_num: int = 500  # number of trace rows to load
    surplus_prompts_num: int = 16384  # total queries to issue

    # synthetic (gamma arrival + zipf length) knobs, used when not replaying
    qps: float = 1.0
    zipf_param: float = 1.1
    gamma_shape: float = 0.5
    gamma_scale: float = 2.0
    max_prompt_len: int = 1024
    max_gen_len: int = 1024

    # --- backend / server -------------------------------------------------
    backend: str = "vllm"  # "vllm" | "lightllm"
    host: str = "localhost"
    port: int = 17717
    stream: bool = False
    ignore_eos: bool = False
    do_sample: bool = False
    temperature: float = 0.0
    max_tokens: int = 128

    # --- logging ----------------------------------------------------------
    log_path: str = "./server_log.jsonl"
    detail_log_path: str = "./detail_server_log.jsonl"

    def __post_init__(self) -> None:
        if self.conv_or_api not in ("conv", "api"):
            raise ValueError(
                f"conv_or_api must be 'conv' or 'api', got {self.conv_or_api!r}"
            )
        if self.backend not in ("vllm", "lightllm"):
            raise ValueError(
                f"backend must be 'vllm' or 'lightllm', got {self.backend!r}"
            )
        if self.scale <= 0:
            raise ValueError(f"scale must be > 0, got {self.scale}")
        if self.use_burstgpt and not self.burstgpt_path:
            raise ValueError("burstgpt_path is required when use_burstgpt is set")
        if self.backend == "lightllm" and not self.stream:
            raise ValueError("the lightllm backend only supports stream=True")

    @property
    def log_type(self) -> str:
        """BurstGPT 'Log Type' column value for the selected mode."""
        return "Conversation log" if self.conv_or_api == "conv" else "API log"
