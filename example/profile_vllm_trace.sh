#!/bin/bash
set -euo pipefail

# Replay the BurstGPT trace against a running vLLM server.
# Install first:  pip install -e .
burstgpt-bench \
    --backend=vllm \
    --host=localhost \
    --port=8000 \
    --temperature=0 \
    --data_path=preprocess_data/shareGPT.json \
    --stream \
    --use_burstgpt \
    --burstgpt_path=../data/BurstGPT_1.csv \
    --conv_or_api=conv \
    --prompt_num=50 \
    --surplus_prompts_num=50 \
    --scale=1.2344107085 \
    --log_path=./server_log.jsonl \
    --detail_log_path=./detail_server_log.jsonl
