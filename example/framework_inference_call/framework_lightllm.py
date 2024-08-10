import argparse
import time

from typing import Tuple
import aiohttp
import asyncio
import time
import json

# query code is borrowed from https://github.com/L1aoXingyu/llm-infer-bench
async def lightllm_inference_call_server(prompt, sleep_time, config, logger, event_id):
    # with lightllm, streaming always true
    await asyncio.sleep(sleep_time)
    assert config.server_config['stream'], "lightllm only support streaming with True"
    
    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)  # 4 hours
    async with aiohttp.ClientSession(timeout=timeout) as session:
        generation_input = {
            "inputs": prompt,
            "parameters": {
            "do_sample": config.server_config['do_sample'],
            "ignore_eos": config.server_config['ignore_eos'],
            "max_new_tokens": config.server_config['max_tokens'],
            "temperature": config.server_config['temperature'],
            }
        }
        first_chunk_time = 0
        start_time = time.perf_counter()
        async with session.post(
            f"http://localhost:{config.server_config['port']}/generate", json=generation_input
        ) as resp:
            if resp.status != 200:
                print(f"Error: {resp.status} {resp.reason}")
                print(await resp.text())
                return None, None, None

            if config.server_config['stream']:
                chunks = []
                first_chunk_received = False
                async for chunk,_ in resp.content.iter_chunks():
                    # If this is the first chunk, record the time taken
                    if not first_chunk_received:
                        first_chunk_time = time.perf_counter() - start_time
                        first_chunk_received = True
                    chunks.append(chunk)

                output = b"".join(chunks).decode("utf-8")
                output = json.loads(output)

            else:
                raise NotImplementedError
            
            end_time = time.perf_counter()
            total_chunk_time = end_time - start_time

            # should counter the output token length after gather all the outputs
    await session.close()
    return output["generated_text"][0], query.prompt, first_chunk_time, total_chunk_time
