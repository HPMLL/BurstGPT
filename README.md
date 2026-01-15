# A ChatGPT(GPT-3.5) & GPT-4 Workload Trace to Optimize LLM Serving Systems

> [!NOTE]
> ✨ Traces with new columns `SessionID` and `Elapsed time` are available now!

This repository contains public releases of a real-world trace dataset of LLM serving workloads for the benefit of the research and academic community.

This LLM serving is powered by Microsoft Azure.

## Main characteristics

### `BurstGPT_1` & `BurstGPT_2`
- Duration: 121 consecutive days in 4 consecutive months.
- Dataset size: ~5.29M lines, ~188MB.

### `BurstGPT_3`

- Duration: 110 consecutive days in 4 consecutive months.
- Dataset size: ~5.34M lines, ~220MB.

## Data

There are currently 6 files in [Release v1.2](https://github.com/HPMLL/BurstGPT/releases/tag/v1.2):

- `BurstGPT_1.csv` contains all of our trace in the first 2 months with some failure that `Response tokens` are `0`s. Totally 1429.7k lines.

- `BurstGPT_without_fails_1.csv` contains all of our trace in the first 2 months without failure. Totally 1404.3k lines.

- `BurstGPT_2.csv` contains all of our trace in the second 2 months with some failure that `Response tokens` are `0`s. Totally 3858.4k lines.

- `BurstGPT_without_fails_2.csv` contains all of our trace in the second 2 months without failure. Totally 3784.2k lines.

- `BurstGPT_3.csv` contains all of our trace in another 110 days with some failure that `Response tokens` are `0`s. Totally 5344.0k lines.

- `BurstGPT_without_fails_3.csv` contains all of our trace in another 110 days without failure. Totally 4956.1k lines.

`BurstGPT_1.csv` is also in `/data` for you to use.

## Schema

- `Timestamp`: request submission time, seconds from `0:00:00` on the first day.
- `Session ID`: conversation ID, only conversation mode have this, traces that share the same value of `Session ID` are in the same conversation session.
- `Elapsed time`: time between the request submission time and system response time, in seconds.
- `Model`: called models, including `ChatGPT`(GPT-3.5) and `GPT-4`.
- `Request tokens`: Request tokens length.
- `Response tokens`: Response tokens length.
- `Total tokens`: Request tokens length plus response tokens length.
- `Log Type`: the way users call the model, in conversation mode or using API, including `Conversation log` and `API log`.


## Usage

1. You may scale the average Requests Per Second (RPS) in the trace according to your evaluation setups.
2. You may also model the patterns in the trace as indicated in our paper <a href="https://dl.acm.org/doi/10.1145/3711896.3737413"><img style="display: inline-block; margin:0; cursor:pointer;" src="https://img.shields.io/badge/ACMDL-3495EA?style=flat-square&"></a> and scale the parameters in the models.
3. Check our simple request generator demo in `example/`. If you have some specific needs, we are eager to assist you in exploring and leveraging the trace to its fullest potential. Please let us know of any issues or questions by sending email to [mailing list](mailto:y.chen@connect.hkust-gz.edu.cn).

## Paper

<a href="https://arxiv.org/pdf/2401.17644.pdf" target="_blank"><img style="display: inline-block; margin:0; cursor:pointer;" alt="arXiv" src="https://img.shields.io/badge/arXiv-8A2BE2?style=flat-square&logo=arxiv&logoColor=FFFFFF"></a>
<a href="https://dl.acm.org/doi/10.1145/3711896.3737413"><img style="display: inline-block; margin:0; cursor:pointer;" src="https://img.shields.io/badge/ACMDL-3495EA?style=flat-square&"></a>

If the trace is utilized in your research, please ensure to reference our paper:

```bibtex
@inproceedings{BurstGPT,
  author    = {Yuxin Wang and Yuhan Chen and Zeyu Li and Xueze Kang and Yuchu Fang and Yeju Zhou and Yang Zheng and Zhenheng Tang and Xin He and Rui Guo and Xin Wang and Qiang Wang and Amelie Chi Zhou and Xiaowen Chu},
  title     = {{BurstGPT}: A Real-World Workload Dataset to Optimize LLM Serving Systems},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD ’25)},
  year      = {2025},
  address   = {Toronto, ON, Canada},
  publisher = {ACM},
  doi       = {https://doi.org/10.1145/3711896.3737413},
  url       = {https://doi.org/10.1145/3711896.3737413},
}
```

## Data Overview (First 2 Months)

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
