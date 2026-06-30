"""Preprocess a ShareGPT-style dump into the prompt-pool JSON used by the
benchmark.

Reads a ShareGPT conversations file, tokenises each (human prompt, gpt
response) pair, and writes ``{idx: {prompt, len_prompt, output, len_output}}``
-- exactly the format :func:`burstgpt.prompts.load_prompt_pool` expects.

Requires the optional ``preprocess`` extra::

    pip install -e '.[preprocess]'
"""

from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess ShareGPT into prompt-pool JSON")
    p.add_argument("--data_path", required=True,
                   help="ShareGPT conversations JSON (e.g. sg_90k_part1_html_cleaned.json)")
    p.add_argument("--model_path", required=True, help="Tokenizer path")
    p.add_argument("--out_path", default="preprocess_data/shareGPT.json")
    p.add_argument("--limit", type=int, default=300,
                   help="Max number of prompt/response pairs to keep")
    p.add_argument("--plot_path", default=None,
                   help="If set, save a token-length histogram PNG here")
    return p


def extract_pairs(conversations):
    """Yield (prompt, response) pairs, accumulating conversation prefixes."""
    for convo in conversations:
        prefix = ""
        pending_prompt = None
        for message in convo.get("conversations", []):
            if message["from"] == "human":
                prefix += message["value"]
                pending_prompt = prefix
            elif message["from"] == "gpt" and pending_prompt is not None:
                prefix += message["value"]
                yield pending_prompt, message["value"]
                pending_prompt = None


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    with open(args.data_path) as f:
        conversations = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    pool = {}
    prompt_lens, output_lens = [], []
    for idx, (prompt, output) in enumerate(extract_pairs(conversations)):
        if idx >= args.limit:
            break
        len_prompt = len(tokenizer(prompt)["input_ids"])
        len_output = len(tokenizer(output)["input_ids"])
        pool[idx] = {
            "prompt": prompt,
            "len_prompt": len_prompt,
            "output": output,
            "len_output": len_output,
        }
        prompt_lens.append(len_prompt)
        output_lens.append(len_output)

    with open(args.out_path, "w") as f:
        json.dump(pool, f)
    print(f"wrote {len(pool)} prompts to {args.out_path}")

    if args.plot_path:
        _plot(prompt_lens, output_lens, args.plot_path)


def _plot(prompt_lens, output_lens, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(prompt_lens, 200, range=(0, 4096), color="blue", alpha=0.7, label="prompt")
    ax.hist(output_lens, 200, range=(0, 4096), color="red", alpha=0.7, label="output")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Frequency")
    ax.set_title("Token length distribution of ShareGPT dataset")
    ax.legend()
    fig.savefig(path)
    print(f"saved histogram to {path}")


if __name__ == "__main__":
    main()
