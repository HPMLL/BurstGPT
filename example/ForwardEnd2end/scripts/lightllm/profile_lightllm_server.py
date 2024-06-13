import sys
import os
import argparse

sys.path.append("../../")
from profile_server import ServerOnline, Config

def add_parser_arg(parser):
    parser.add_argument('--data_path', type=str, 
                        help='The path the alpaca json file',
                        default="../../preprocess_data/alpaca_prompt.json")
    parser.add_argument('--model_path', type=str, 
                        help='The path the tokenizer',
                        default="/hpc2hdd/home/zli755/data/llama-2-7b-chat-hf/")
    parser.add_argument('--log_path', type=str, 
                        help='The path the log file to save',
                        default="./server_log.json")
    parser.add_argument("--detail_log_path", type=str, default="./detail_server_log.json",
                        help='The detail log path')
    parser.add_argument('--max_num_seqs', type=int, 
                        help='The maximum seqs batched',
                        default=256)
    parser.add_argument('--max_num_batched_tokens', type=int, 
                        help='The maximum tokens batched',
                        default=4096)

    parser.add_argument("--ignore_eos", action="store_true", default=False,
                        help="Ignore the eos")
    parser.add_argument("--max_tokens", type=int, 
                        help="The maximum tokens generated", default=128)
    parser.add_argument("--stream", action='store_true', default=False,
                        help="If the output is streaming or not, only for server mode")
    parser.add_argument("--qps", type=float, default=1.0,
                        help="Query Per Second, \
                        the parameter of Possion distribution to generate the query.")
    parser.add_argument("--port", type=int, default=17717,
                        help="The server port")

    #https://platform.openai.com/docs/api-reference/completions/create
    parser.add_argument("--temperature", type=int, default=0,
                        help="Ref to Openai api temperature")

    parser.add_argument("--dummy_data", action="store_true", default=False,
                        help="Use dummy_data to profile")
    parser.add_argument("--input_seq_len", type=int, default=128,
                        help="The input seq len with dummy_data")

    parser.add_argument("--num_batch_load", type=int, default=1,
                        help="The batch num")
    parser.add_argument("--batch_size_prompt", type=int, default=1,
                        help="The batch size of prompt per query")

    parser.add_argument('--num_prompts_load', type=int,
                        help='The maximum number of prompts to load',
                        default=sys.maxsize)
    parser.add_argument('--seed', type=int, default=0,
                        help="The random seed of prompt set used in shuffle")

    parser.add_argument("--backend", type=str, choices=["lightllm", "vllm"],
                        help="The inference backend, including lightllm, vllm")
    parser.add_argument("--do_sample", action="store_true", default=False,
                        help="whether do sample or not, for lightllm ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arg(parser)
    args = parser.parse_args()

    server_config = dict()
    prompt_config = dict()
    server_config['do_sample'] = args.do_sample
    server_config['stream'] = args.stream
    server_config['ignore_eos'] = args.ignore_eos
    server_config['max_tokens'] = args.max_tokens
    server_config['qps'] = args.qps
    server_config['port'] = args.port
    server_config['temperature'] = args.temperature
    prompt_config['dummy_data'] = args.dummy_data
    prompt_config['input_seq_len'] = args.input_seq_len
    prompt_config['seed'] = args.seed

    config = Config(server_config=server_config, prompt_config=prompt_config)
    server = ServerOnline(model_path=args.model_path,
                           data_path=args.data_path,
                           backend=args.backend,
                           log_path=args.log_path,
                           config=config,
                           detail_log_path=args.detail_log_path)
    server.start_profile()
    server.save_log()
