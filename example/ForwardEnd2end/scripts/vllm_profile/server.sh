#!/bin/bash 

python profile_vllm_server.py --port=17722 --temperature=0 --data_path=/shareGPT_prompt.json --stream --surplus_prompts_num=16384
