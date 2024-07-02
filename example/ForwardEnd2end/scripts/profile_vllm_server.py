import sys
import os
import argparse

sys.path.append("../")
from profile_server import ServerOnline, Config

def add_parser_arg(parser):
    parser.add_argument('--data_path', type=str, 
                        help='The path the alpaca json file',
                        default="../../preprocess_data/shareGPT.json")
    parser.add_argument('--model_path', type=str, 
                        help='The path the tokenizer',
                        default="/hpc2hdd/home/zli755/data/llama-2-7b-hf/")
    parser.add_argument('--log_path', type=str, 
                        help='The path the log file to save',
                        default="./server_log.json")
    parser.add_argument("--detail_log_path", type=str, default="./detail_server_log_1.json",
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
    parser.add_argument("--host", type=str, default='localhost',
                        help="The server host")
    parser.add_argument("--port", type=int, default=17717,
                        help="The server port")

    #https://platform.openai.com/docs/api-reference/completions/create
    parser.add_argument("--temperature", type=int, default=0,
                        help="Ref to Openai api temperature")

    parser.add_argument('--seed', type=int, default=0,
                        help="The random seed of prompt set used in shuffle")

    parser.add_argument("--surplus_prompts_num", type=int, default=16384,
                        help="The total query num(surplus) to send")
    
    parser.add_argument("--use_burstgpt", action="store_true", default=False,
                        help="Using BurstGPT trace instead of using gamma and zipf")

    parser.add_argument("--burstgpt_path", type=str, 
                        help="BurstGPT trace path",
                        default="/hpc2hdd/home/ychen906/repo/BurstGPT/data/BurstGPT_without_fails.csv")
    
    parser.add_argument("--prompt_num", type=int,  default=500,
                        help="Prompt number, 500 by default")
    
    parser.add_argument("--conv_or_api", type=str,  default='conv',
                        help="Using BurstGPT Conv or API trace, use conv by default")
    
    parser.add_argument("--scale", type=float,  default=100,
                        help="Scale trace, 100 means 100 times faster, use 100 by default")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arg(parser)
    args = parser.parse_args()

    server_config = dict()
    prompt_config = dict()
    server_config['stream'] = args.stream
    server_config['ignore_eos'] = args.ignore_eos
    server_config['qps'] = args.qps
    server_config['host'] = args.host
    server_config['port'] = args.port
    server_config['temperature'] = args.temperature
    server_config['max_tokens'] = args.max_tokens
    prompt_config['seed'] = args.seed
    prompt_config['surplus_prompts_num'] = args.surplus_prompts_num
    prompt_config['use_burstgpt'] = args.use_burstgpt
    prompt_config['burstgpt_path'] = args.burstgpt_path
    prompt_config['conv_or_api'] = args.conv_or_api
    prompt_config['scale'] = args.scale
    prompt_config['prompt_num'] = args.prompt_num
    
    print(prompt_config)

    config = Config(server_config=server_config, prompt_config=prompt_config)
    server = ServerOnline(model_path=args.model_path,
                           data_path=args.data_path,
                           backend="vllm",
                           log_path=args.log_path,
                           config=config,
                           detail_log_path=args.detail_log_path)
    server.start_profile()
    server.save_log()
