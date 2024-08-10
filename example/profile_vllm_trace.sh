#!/bin/bash 

python profile_vllm_server.py --port=8000 --temperature=0 --data_path=preprocess_data/shareGPT.json --stream --surplus_prompts_num=50 --use_burstgpt --prompt_num=50 --scale=1.2344107085 --burstgpt_path=../data/BurstGPT_1.csv
