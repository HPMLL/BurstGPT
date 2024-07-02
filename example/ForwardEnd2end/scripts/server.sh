#!/bin/bash 

python profile_vllm_server.py --port=17717 --temperature=0 --data_path=/path/to/shareGPT_prompt.json --stream --surplus_prompts_num=20 --use_burstgpt --prompt_num=20 --scale=1.2344107085 --burstgpt_path=../../../data/BurstGPT_1.csv
