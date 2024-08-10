
import json
import argparse
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.stats as stats
import os
import time

from transformers import LlamaTokenizer, AutoTokenizer

def add_parser_arg(parser):
    parser.add_argument('--data_path_part1', type=str, nargs="?",
                        help='The path the alpaca json file',
                        default="/data2/share/inference_benchmark_dataset/sg_90k_part1_html_cleaned.json")
    parser.add_argument('--data_path_part2', type=str, nargs="?",
                        help='The path the alpaca json file',
                        default="/data2/share/inference_benchmark_dataset/sg_90k_part2_html_cleaned.json")
    parser.add_argument('--model_path', type=str, nargs="?",
                        help='The path the tokenizer',
                        default="/data2/share/llama/Llama-2-7b-hf/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arg(parser)
    args = parser.parse_args()
    data_path_part1 = args.data_path_part1
    data_path_part2 = args.data_path_part2
    with open(data_path_part1) as json_data:
        sharedGPT_json = json.load(json_data)
        json_data.close()
    
    prompts = [] 
    outputs = []
    conversation_idx = -1
    in_count = 0
    gen_count = 0
    start_time = time.perf_counter()
    for data in sharedGPT_json:
        prefix_conversation = ""
        for message in data['conversations']:
            if message["from"] == "human":
                conversation_idx += 1
                in_count += 1
                prefix_conversation += message["value"]
                prompts.append(prefix_conversation)
            elif message["from"] == "gpt":
                gen_count += 1
                prefix_conversation += message["value"]
                outputs.append(message["value"])

    print(f"time during: {time.perf_counter()-start_time}")    
    # start_time = time.perf_counter()
    # with open(data_path_part2) as json_data:
    #     sharedGPT_json = json.load(json_data)
    #     json_data.close()

    # for data in sharedGPT_json:
    #     prefix_conversation = ""
    #     for message in data['conversations']:
    #         if message["from"] == "human":
    #             in_count += 1
    #             prefix_conversation += message["value"]
    #             inputs[conversation_idx]["prompt"] = prefix_conversation
    #             prompts.append(prefix_conversation)
    #         elif message["from"] == "gpt":
    #             gen_count += 1
    #             prefix_conversation += message["value"]
    #             inputs[conversation_idx]["output"] = message["value"]
    #             outputs.append(message["value"])


    # print(f"time during: {time.perf_counter()-start_time}")    
    start_time = time.perf_counter()

    print(in_count)
    print(gen_count)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True
    )

    fig, ax = plt.subplots()
    tokens = []

    inputs = dict()
    count = 0
    for prompt in prompts[:300]:
        if count % 10 == 0:
            print(count)
        tokens.append(tokenizer(prompt)["input_ids"])
        inputs[count] = dict()
        inputs[count]["prompt"] = prompt
        inputs[count]["len_prompt"] = len(tokens[-1])
        count += 1
    token_lens = [len(token) for token in tokens ]
    n, bins, patches = ax.hist(token_lens, 200, range=(0,4096), color="blue", alpha=0.7)
    token_lens_np = np.array(token_lens)

    print(f"time during: {time.perf_counter()-start_time}")    
    start_time = time.perf_counter()

    tokens = []
    count = 0
    for output in outputs[:300]:
        if count % 10 == 0:
            print(count)
        tokens.append(tokenizer(output)["input_ids"])
        inputs[count]["output"] = output
        inputs[count]["len_output"] = len(tokens[-1])
        count += 1
    token_lens = [len(token) for token in tokens ]
    n, bins, patches = ax.hist(token_lens, 200, range=(0,4096), color="red", alpha=0.7)
    token_lens_np = np.array(token_lens)
    print(f"time during: {time.perf_counter()-start_time}")    

    ax.set_xlabel("Tokens num")
    ax.set_ylabel("Density")
    ax.set_title("Distribution token num of shareGPT dataset")
    ax.vlines(x=[token_lens_np.mean()], ymin=0, ymax=0.030, color='red', linestyle='dashed')
    plt.savefig("ShareGPT_Distribution.png")


    with open("shareGPT.json", "w") as f:
        json.dump(inputs, f)
        f.close()
