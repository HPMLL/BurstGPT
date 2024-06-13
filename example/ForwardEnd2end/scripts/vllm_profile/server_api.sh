#!/bin/bash 

python profile_vllm_server.py --port=17717 --temperature=0 --data_path=/shareGPT_prompt.json --stream --surplus_prompts_num=2000 --model_path=/llama-2-13b-chat-hf --use_burstgpt --prompt_num=2000 --scale=1.2344107085 --burstgpt_path=../data/BurstGPT_without_fails_1.csv

