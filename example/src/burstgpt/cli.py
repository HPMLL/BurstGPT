"""Command-line entry point. Maps argparse flags onto :class:`RunConfig`.

The old ``profile_vllm_server.py`` manually copied each arg into one of two
dicts; here the flag names mirror :class:`RunConfig` fields and we build it
in one shot, so adding a config field no longer means editing wiring code.
"""

from __future__ import annotations

import argparse
import dataclasses

from burstgpt.config import RunConfig
from burstgpt.runner import ProfileRunner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BurstGPT request generator / profiler")

    # Prompt pool / tokenizer
    p.add_argument("--data_path", default=RunConfig.data_path,
                   help="Preprocessed prompt JSON file")
    p.add_argument("--model_path", default=RunConfig.model_path,
                   help="Tokenizer path (metadata only)")
    p.add_argument("--seed", type=int, default=RunConfig.seed)

    # Workload
    p.add_argument("--use_burstgpt", action="store_true", default=RunConfig.use_burstgpt,
                   help="Replay a BurstGPT trace instead of synthetic gamma/zipf")
    p.add_argument("--burstgpt_path", default=RunConfig.burstgpt_path)
    p.add_argument("--conv_or_api", choices=["conv", "api"], default=RunConfig.conv_or_api)
    p.add_argument("--scale", type=float, default=RunConfig.scale,
                   help="Trace RPS scaling; 100 means 100x faster")
    p.add_argument("--prompt_num", type=int, default=RunConfig.prompt_num,
                   help="Number of trace rows to load")
    p.add_argument("--surplus_prompts_num", type=int, default=RunConfig.surplus_prompts_num,
                   help="Total number of queries to issue")
    p.add_argument("--qps", type=float, default=RunConfig.qps)
    p.add_argument("--zipf_param", type=float, default=RunConfig.zipf_param)
    p.add_argument("--gamma_shape", type=float, default=RunConfig.gamma_shape)
    p.add_argument("--gamma_scale", type=float, default=RunConfig.gamma_scale)
    p.add_argument("--max_prompt_len", type=int, default=RunConfig.max_prompt_len)
    p.add_argument("--max_gen_len", type=int, default=RunConfig.max_gen_len)

    # Backend / server
    p.add_argument("--backend", choices=["vllm", "lightllm"], default=RunConfig.backend)
    p.add_argument("--host", default=RunConfig.host)
    p.add_argument("--port", type=int, default=RunConfig.port)
    p.add_argument("--stream", action="store_true", default=RunConfig.stream)
    p.add_argument("--ignore_eos", action="store_true", default=RunConfig.ignore_eos)
    p.add_argument("--do_sample", action="store_true", default=RunConfig.do_sample)
    p.add_argument("--temperature", type=float, default=RunConfig.temperature)
    p.add_argument("--max_tokens", type=int, default=RunConfig.max_tokens)

    # Logging
    p.add_argument("--log_path", default=RunConfig.log_path)
    p.add_argument("--detail_log_path", default=RunConfig.detail_log_path)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    field_names = {f.name for f in dataclasses.fields(RunConfig)}
    config = RunConfig(**{k: v for k, v in vars(args).items() if k in field_names})
    print(config)
    ProfileRunner(config).run()


if __name__ == "__main__":
    main()
