import numpy as np
import pytest

from burstgpt.config import RunConfig
from burstgpt.prompts import Prompt, PromptMatcher
from burstgpt.workload import SyntheticWorkload, TraceWorkload


def test_config_defaults_validate():
    cfg = RunConfig()
    assert cfg.log_type == "Conversation log"
    assert RunConfig(conv_or_api="api").log_type == "API log"


@pytest.mark.parametrize("bad", [
    dict(conv_or_api="nope"),
    dict(backend="triton"),
    dict(scale=0),
    dict(backend="lightllm", stream=False),
])
def test_config_rejects_invalid(bad):
    with pytest.raises(ValueError):
        RunConfig(**bad)


def test_matcher_nearest_neighbour():
    pool = [
        Prompt("a", prompt_len=10, output_len=5),
        Prompt("b", prompt_len=10, output_len=50),
        Prompt("c", prompt_len=100, output_len=20),
    ]
    m = PromptMatcher(pool)
    # exact-ish prompt len, output closest to 5
    assert m.match(10, 6).text == "a"
    # output closer to 50
    assert m.match(10, 45).text == "b"
    # prompt len closest to 100
    assert m.match(90, 0).text == "c"
    # below range still resolves
    assert m.match(0, 0).text == "a"


def test_synthetic_workload_runs_and_is_bounded():
    cfg = RunConfig(max_prompt_len=64, max_gen_len=64)
    rng = np.random.default_rng(0)
    wl = SyntheticWorkload(cfg, count=20, rng=rng)
    queries = list(wl)
    assert len(queries) == 20 == len(wl)
    for q in queries:
        assert 0 < q.prompt_len < cfg.max_prompt_len
        assert 0 < q.output_len < cfg.max_gen_len
        assert q.arrival_time >= 0


def test_synthetic_workload_is_deterministic_per_seed():
    cfg = RunConfig()
    a = list(SyntheticWorkload(cfg, 10, np.random.default_rng(7)))
    b = list(SyntheticWorkload(cfg, 10, np.random.default_rng(7)))
    assert [q.arrival_time for q in a] == [q.arrival_time for q in b]


def test_trace_workload_loads_and_scales(tmp_path):
    csv = tmp_path / "trace.csv"
    csv.write_text(
        "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
        "10,ChatGPT,100,20,120,Conversation log\n"
        "30,ChatGPT,5000,9000,14000,Conversation log\n"
    )
    cfg = RunConfig(use_burstgpt=True, burstgpt_path=str(csv), prompt_num=2, scale=2.0,
                    max_prompt_len=1024, max_gen_len=1024)
    wl = TraceWorkload(cfg)
    queries = list(wl)
    assert len(queries) == 2
    # First arrival = 10/scale - 0 = 5
    assert queries[0].arrival_time == pytest.approx(5.0)
    # Second arrival = 30/2 - 10/2 = 10
    assert queries[1].arrival_time == pytest.approx(10.0)
    # Lengths clamped to max-1
    assert queries[1].prompt_len == 1023
    assert queries[1].output_len == 1023
