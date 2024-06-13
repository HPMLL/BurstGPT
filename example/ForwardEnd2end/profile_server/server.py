import os
import sys
import time
import json
import numpy as np
from profile_server import PromptSet
from profile_server import Logger
import asyncio
from datetime import datetime
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from torch.profiler import profile, record_function, ProfilerActivity
import contextlib
import random
import bisect
import pandas as pd



sys.path.append("../")
from framework_inference_call import vllm_inference_call_server, lightllm_inference_call_server

# reference mlperf inference
class ServerBase(object):
    def __init__(self, model_path, data_path, backend,
                 device, log_path, config):
        self.config = config
        self.model_path = model_path
        self.data_path = data_path
        self.device = device
        self.backend = backend
        self.log_path = log_path
        # currently, batch is only used in offline mode
        if self.config.prompt_config['use_burstgpt']:
            # self.trace = pd.read_csv(self.config.prompt_config['burstgpt_path'])
            self.conv_or_api = self.config.prompt_config['conv_or_api']
            chunk_iter = pd.read_csv(self.config.prompt_config['burstgpt_path'], chunksize=100)

            chunks = []
            conv_or_api = 'Conversation log' if self.config.prompt_config['conv_or_api'] == 'conv' else 'API log'
                
            print(conv_or_api)
            total_prompt_num = 0
            # start_num = 500
            # max_pn = self.config.prompt_config['prompt_num'] + total_prompt_num
            # print("loading trace.", end="")
            for chunk in chunk_iter:
                # if total_prompt_num 
                # print(".", end="")
                # filtered_chunk = chunk #[chunk['Log Type'] == conv_or_api]
                chunks.append(chunk)
                total_prompt_num += len(chunk)
                if total_prompt_num >= self.config.prompt_config['prompt_num']: break
            self.trace = pd.concat(chunks)
            self.trace.reset_index(drop=True, inplace=True)

            print("scale", self.config.prompt_config['scale'])
            self.trace['end_timestamp'] = self.trace['end_timestamp'] / self.config.prompt_config['scale']
            # self.trace['Timestamp'] = self.trace['Timestamp'] / self.config.prompt_config['scale']
        else:
            self.trace = None
            self.conv_or_api = None
        
        print(self.trace)

        self.dataset = PromptSet(self.data_path, config.prompt_config)

        self.logger = Logger(self.log_path)
        self.num_out_tokens = 0
        self.num_in_tokens = 0

    def issue_queries(self):
        pass
    
    def inference_call(self):
        pass

    def start_profile(self):
        pass

    def save_log(self):
        pass


class ServerOnline(ServerBase):
    class _Query(object):
        def __init__(self, inputs, qps=1, trace=None, conv_or_api=None):
            self.inputs = inputs 
            self.qps = qps
            self.query_time = 0
            self.max_prompt_len = 1024 #  4096 # 1024
            self.max_gen_len = 1024 # 4096 # 1024
            self.prefill_idx = self.get_prefill_idx()
            self.zipf_param = 1.1
            self.gamma_shape = 0.25
            self.gamma_scale = 2

            self.query_id = 0

            self.gamma_shape_dict = dict()
            
            # if trace is not None:
            self.trace = trace
            self.conv_or_api = conv_or_api

            df = pd.read_csv('/data/work_days_avg_gamma_per20min_conv_filtered_8_to_plus_1.csv')
            # df = pd.read_csv('/data/work_days_avg_gamma_per20min_api_filtered_12_to_plus_5.csv') # api
            # shape_array = df['shape'].to_numpy()
            # scale_array = df['scale'].to_numpy()
            self.shape_list = df['alpha'].tolist()
            self.scale_list = df['beta'].tolist()
            self.gamma_step = -1
            self.step_start_time = 0
            
            ### LOAD BURST_GPT PROMPT LEN
            chunk_bgpt_iter = pd.read_csv("../data/BurstGPT_without_fails_1.csv", chunksize=100)
            chunks_bgpt = []
            total_prompt_num = 0
            for chunk_bgpt in chunk_bgpt_iter:
                filtered_chunk_bgpt = chunk_bgpt[chunk_bgpt['Log Type'] == 'Conversation log']
                chunks_bgpt.append(filtered_chunk_bgpt)
                total_prompt_num += len(filtered_chunk_bgpt)
                if total_prompt_num >= 41800: break
            self.trace_bgpt = pd.concat(chunks_bgpt)            
            self.trace_bgpt.reset_index(drop=True, inplace=True)


        def get_prefill_idx(self):
            prefill_idx = np.ones((self.max_prompt_len, self.max_gen_len), dtype=np.int32) * -1
            record = np.zeros(self.max_prompt_len)
            print("prefill record")
            for idx, data in zip(range(len(self.inputs)), self.inputs):
                len_prompt = data[1]
                len_output = data[2]
                if len_prompt >= self.max_prompt_len or len_output >= self.max_gen_len:
                    continue

                prefill_idx[len_prompt][len_output] = idx
                record[len_prompt] = 1
            print("prefill generate idx in one prompt len")
            for idx_ii in range(self.max_prompt_len):
                if record[idx_ii] == 0:
                    continue

                previous_idx = -1
                for idx_jj in range(self.max_gen_len):
                    if prefill_idx[idx_ii][idx_jj] != -1:
                        for _idx_jj in range(idx_jj):
                            prefill_idx[idx_ii][_idx_jj] = prefill_idx[idx_ii][idx_jj]
                        previous_idx = idx_jj
                        break

                for idx_jj in range(previous_idx, self.max_gen_len):
                    if prefill_idx[idx_ii][idx_jj] == -1:
                        flag = 0
                        for _idx_jj in range(idx_jj, min(self.max_gen_len, idx_jj*2-previous_idx)):
                            if prefill_idx[idx_ii][_idx_jj] != -1:
                                flag = 1
                                prefill_idx[idx_ii][idx_jj] = prefill_idx[idx_ii][_idx_jj]
                                break
                        if flag == 0:
                            prefill_idx[idx_ii][idx_jj] = prefill_idx[idx_ii][previous_idx]
                    else:
                        previous_idx = idx_jj
                
            previous_idx = -1
            print("prefill prompt len")
            for _idx_ii in range(self.max_prompt_len):
                if record[_idx_ii] != 0:
                    for __idx_ii in range(_idx_ii):
                        for idx_jj in range(self.max_gen_len):
                            prefill_idx[__idx_ii][idx_jj] = prefill_idx[_idx_ii][idx_jj]
                    previous_idx = _idx_ii
                    break
            
            print("prefill prompt len via prompt")
            for idx_ii in range(previous_idx, self.max_prompt_len):
                if record[idx_ii] == 0:
                    flag = 0
                    for _idx_ii in range(idx_ii, min(self.max_prompt_len, idx_ii*2-previous_idx)):
                        if record[_idx_ii] == 1:
                            flag = 1
                            for _idx_jj in range(self.max_gen_len):
                                prefill_idx[idx_ii][_idx_jj] = prefill_idx[_idx_ii][_idx_jj]
                            break
                    if flag == 0:
                        for _idx_jj in range(self.max_gen_len):
                            prefill_idx[idx_ii][_idx_jj] = prefill_idx[previous_idx][_idx_jj]
                else:
                    previous_idx = idx_ii

            return prefill_idx

        def get_query(self):
            # print("get_query", self.query_id)
            
            if self.trace is None or self.query_id >= len(self.trace):
                # print("self.trace is None.")

                if self.query_id >= 1000 and self.gamma_step < len(self.shape_list) - 1:
                    # print("self.step_start_time", self.step_start_time)
                    # print("time.monotonic()", time.monotonic())
                    print("time.monotonic() - self.step_start_time", time.monotonic() - self.step_start_time)
                    if self.gamma_step == -1 or self.query_time - self.step_start_time > 60:
                        self.gamma_step += 1
                        self.step_start_time = self.query_time
                    # elif self.query_id % 60 == 0:
                    #     self.gamma_step += 1

                    print(" self.gamma_step: ", self.gamma_step)
                    self.gamma_shape = self.shape_list[self.gamma_step]
                    self.gamma_scale = 1 / self.scale_list[self.gamma_step] / 30.6094502625 # SCALE

                # gamma_shape control
                # if self.query_id >= 1000 and self.query_id < 1200:
                #     self.gamma_shape = 1/ ((64-2)*(self.query_id-1000)/200+2)**2 # 1 / sqrt(k) from 2 to 64
                # elif self.query_id >= 1200 and self.query_id < 1400:
                #     self.gamma_shape = 1/ ((64-2)*(1400-self.query_id)/200+2)**2
                # elif self.query_id >= 2000 and self.query_id < 2200:
                #     self.gamma_shape = 1/ ((32-2)*(self.query_id-2000)/200+2)**2 # 1 / sqrt(k) from 2 to 32
                # elif self.query_id >= 2200 and self.query_id < 2400:
                #     self.gamma_shape = 1/ ((32-2)*(2400-self.query_id)/200+2)**2
                # elif self.query_id >= 3000 and self.query_id < 3200:
                #     self.gamma_shape = 1/ ((16-2)*(self.query_id-3000)/200+2)**2 # 1 / sqrt(k) from 2 to 16
                # elif self.query_id >= 3200 and self.query_id < 3400:
                #     self.gamma_shape = 1/ ((16-2)*(3400-self.query_id)/200+2)**2
                # elif self.query_id >= 4000 and self.query_id < 4200:
                #     self.gamma_shape = 1/ ((8-2)*(self.query_id-4000)/200+2)**2 # 1 / sqrt(k) from 2 to 8
                # elif self.query_id >= 4200 and self.query_id < 4400:
                #     self.gamma_shape = 1/ ((8-2)*(4400-self.query_id)/200+2)**2
                # elif self.query_id >= 5000 and self.query_id < 5200:
                #     self.gamma_shape = 1/ ((4-2)*(self.query_id-5000)/200+2)**2 # 1 / sqrt(k) from 2 to 4
                # elif self.query_id >= 5200 and self.query_id < 5400:
                #     self.gamma_shape = 1/ ((4-2)*(5400-self.query_id)/200+2)**2
                else:
                    # self.gamma_shape = 0.25
                    self.gamma_shape = 0.5
                    self.gamma_scale = 2.0
                    



                delta_time = np.random.gamma(self.gamma_shape, self.gamma_scale)
                sampled_prompt_len = np.random.zipf(a=self.zipf_param)
                sampled_output_len = np.random.zipf(a=self.zipf_param)
                while sampled_prompt_len >= self.max_prompt_len:
                    sampled_prompt_len = np.random.zipf(a=self.zipf_param)
                while sampled_output_len >= self.max_gen_len:
                    sampled_output_len = np.random.zipf(a=self.zipf_param)
                self.query_time = delta_time + self.query_time
            
            else: # using trace
                
                delta_time = self.trace.at[self.query_id, 'end_timestamp'] - self.query_time
                self.query_time = self.trace.at[self.query_id, 'end_timestamp']
                sampled_prompt_len = self.trace_bgpt.at[self.query_id, 'Request tokens']
                if sampled_prompt_len >= self.max_prompt_len:
                    sampled_prompt_len = self.max_prompt_len - 1
                sampled_output_len = self.trace_bgpt.at[self.query_id, 'Response tokens']
                if sampled_output_len >= self.max_gen_len:
                    sampled_output_len = self.max_gen_len - 1
                # sampled_prompt_len = np.random.zipf(a=self.zipf_param)
                # sampled_output_len = np.random.zipf(a=self.zipf_param)
                # while sampled_prompt_len >= self.max_prompt_len:
                #     sampled_prompt_len = np.random.zipf(a=self.zipf_param)
                # while sampled_output_len >= self.max_gen_len:
                #     sampled_output_len = np.random.zipf(a=self.zipf_param)

            # if self.query_id >= 0 and self.query_id <= 54400:
            if self.query_id >= 0 and self.query_id <= len(self.trace)-1:
                if self.query_id % 10 == 0:
                    self.gamma_shape_dict[self.query_id] = dict()
                    self.gamma_shape_dict[self.query_id]["query_id"] = self.query_id
                    self.gamma_shape_dict[self.query_id]["query time"] = time.monotonic() + self.query_time
                    self.gamma_shape_dict[self.query_id]["zipf_param"] = self.zipf_param
                    self.gamma_shape_dict[self.query_id]["gamma_shape"] = self.gamma_shape
                    self.gamma_shape_dict[self.query_id]["gamma_scale"] = self.gamma_scale
                # if self.query_id == 54000:
                if self.query_id == len(self.trace)-1:
                    current_time = datetime.now().strftime("%H:%M")
                    if self.trace is None:
                        current_time_str = "gamma_zipf_" + current_time.replace(":", "_") + ".json"
                    else:
                        current_time_str = "trace_" + self.conv_or_api + '_' + current_time.replace(":", "_") + ".json"
                    with open(current_time_str, "a") as f:
                        json.dump(self.gamma_shape_dict, f)

            prompt_len = self.inputs[self.prefill_idx[sampled_prompt_len][sampled_output_len]][1]
            output_len = self.inputs[self.prefill_idx[sampled_prompt_len][sampled_output_len]][2]
            
            self.query_id += 1

            return [self.inputs[self.prefill_idx[sampled_prompt_len][sampled_output_len]][0], prompt_len, self.max_gen_len, sampled_prompt_len, self.max_gen_len, delta_time, self.query_time]

    def __init__(self, model_path, data_path, backend="vllm",
                 device="gpu", log_path="./server_log_trace_gamma.json", 
                 config=None, detail_log_path="./detail_server_log_trace_gamma_13b_conv.json"):
        ServerBase.__init__(self, model_path, data_path, backend, device, 
                            log_path, config)
        self.qps = config.server_config.get('qps')
        self.detail_log_path = detail_log_path
        self.detail_logger = Logger(self.detail_log_path)
        # import sys
        # sys.exit()

        self.inputs = []
        for idx, data in self.dataset.data.items():
            self.inputs.append((self.dataset.data[idx]["prompt"], self.dataset.data[idx]["len_prompt"], self.dataset.data[idx]["len_output"], self.dataset.data[idx]['output']))

        self.inputs.sort(key=lambda x: (x[1], x[2]))

        self.queries = self._Query(qps=self.qps, inputs=self.inputs, trace=self.trace, conv_or_api=self.conv_or_api)

    async def issue_queries(self):

        # 16384 is a request num exceed the maximum req num we need
        surplus_prompts_num = self.config.prompt_config['surplus_prompts_num']

        event_id = self.logger.tick_start("Query", time.perf_counter())

        _task_list = []
        for _ in range(surplus_prompts_num):
            detail_event_id = self.detail_logger.tick_start(__name__, time.perf_counter())
            _prompt, in_num, out_num, sampled_in_num, sampled_out_num, _, _sleep_time = self.queries.get_query()
            if self.backend=="vllm":
                # async
                _task_list.append(
                    asyncio.create_task(
                        vllm_inference_call_server(_prompt, in_num, out_num, sampled_in_num, sampled_out_num, _sleep_time, self.config, self.detail_logger, detail_event_id)
                    ))
            elif self.backend=="lightllm":
                _task_list.append(
                    asyncio.create_task(
                        lightllm_inference_call_server(_prompt, _sleep_time, self.config, self.detail_logger, detail_event_id)        
                    ))
        
        await asyncio.gather(*_task_list)
        self.logger.tick_end(event_id, time.perf_counter())
        

    def start_profile(self):
        asyncio.run(self.issue_queries())
    
    def save_log(self):
        print("[INFO]Saving log data")
        #self.logger.log_kv("Throughput", self.num_out_tokens
        #                    / float(self.logger.event_log[event_id]["during"][:-2]))
        self.logger.log_kv("model path", self.model_path)
        self.logger.log_kv("server_config", str(self.config.server_config))
        self.logger.log_kv("prompt_config", str(self.config.prompt_config))
        self.logger.log_kv("log time", str(datetime.now()))
        self.logger.save()