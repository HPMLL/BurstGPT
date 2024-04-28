# A GPT-3.5 & GPT-4 Workload Trace to Optimize LLM Serving Systems 

This repository contains public releases of a real-world trace dataset of LLM serving workloads for the benefit of the research and academic community.

This LLM serving is powered by Microsoft Azure.

There are currently two files in `/data`:

- `BurstGPT.csv` contains all of our trace in 2 month with some failure that `Response tokens` are 0s. Totally 1429.7k lines.

- `BurstGPT_without_fails.csv` contains all of our trace in 2 month without failure. Totally 1404.3k lines.

## Usage
1. You may scale the RPS in the trace according to your evaluation setups.
2. You may also model the patterns in the trace as indicated in our paper and scale the parameters in the models.
3. If you have some specific needs, we are eager to assist you in exploring and leveraging the trace to its fullest potential. Please let us know of any issues or questions by sending email to [mailing list](mailto:ychen906@connect.hkust-gz.edu.cn).

## Future Plans
1. We will continue to update the time range of the trace.
2. We will open-source the full benchmark suite for LLM inference soon.

## Paper

<a href="https://arxiv.org/pdf/2401.17644.pdf" target="_blank"><img style="display: inline-block; margin:0; cursor:pointer;" alt="arXiv" src="https://img.shields.io/badge/arXiv-8A2BE2?style=flat-square&logo=arxiv&logoColor=FFFFFF"></a>

If the trace is utilized in your research, please ensure to reference our paper:

```bibtex
@misc{wang2024efficient,
      title={Towards Efficient and Reliable LLM Serving: A Real-World Workload Study}, 
      author={Yuxin Wang and Yuhan Chen and Zeyu Li and Zhenheng Tang and Rui Guo and Xin Wang and Qiang Wang and Amelie Chi Zhou and Xiaowen Chu},
      year={2024},
      eprint={2401.17644},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```

## Main characteristics

- Duration: 61 consecutive days in 2 consecutive months.
- Dataset size: 1.4m lines, ~50MB.

## Schema

- `Timestamp`: request submission time, seconds from 0:00:00 on the first day.
- `Model`: called models, including `ChatGPT` and `GPT-4`.
- `Request tokens`: Request tokens length.
- `Response tokens`: Response tokens length.
- `Total tokens`: Request tokens length plus response tokens length.
- `Log Type`: the way users call the model, in conversation mode or using API, including `Conversation log` and `API log`.

## Data Overview

<!-- <div align="center">
  <img src="img/Fig1-4.png" alt="" width="800"/><br>

  *Figure 1, 2, 3, 4: Periodicity in BurstGPT.*<br>
</div> -->
<div align="center">
  <img src="img/Fig1-2.png" alt="" width="900"/><br>

  *Figure 1: Weekly Periodicity in BurstGPT.*<br>
</div>

<div align="center">
  <img src="img/Fig3-4.png" alt="" width="900"/><br>

  *Figure 2: Daily Periodicity in BurstGPT.*<br>
</div>

<div align="center">
  <img src="img/Fig10-13.png" alt="" width="900"/><br>

  *Figure 3: Average Daily Request and Response Throughput in BurstGPT.*<br>
</div>

<!-- <div align="center">
  <img src="img/Fig7.png" alt="" width="800"/><br>

  *Figure 9: Distribution of Request and Response Tokens in BurstGPT and Llama-2-13b-chat.*<br>
</div> -->

<div align="center">
  <img src="img/req-res.png" alt="" width="600"/><br>

  *Figure 4: Statistics of Request and Response Tokens in BurstGPT.*<br>
</div>
