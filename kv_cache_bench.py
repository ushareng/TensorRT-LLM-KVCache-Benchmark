import os
import time
import torch
from tensorrt_llm.runtime import ModelRunner

BASE_ENGINE_DIR = os.environ.get("BASE_ENGINE_DIR", "/root/engines/base")
KV_ENGINE_DIR = os.environ.get("KV_ENGINE_DIR", "/root/engines/kv_opt")

BATCH = int(os.environ.get("BATCH", "8"))
INPUT_LEN = int(os.environ.get("INPUT_LEN", "512"))
OUTPUT_LEN = int(os.environ.get("OUTPUT_LEN", "512"))
WARMUP = int(os.environ.get("WARMUP", "10"))
RUNS = int(os.environ.get("RUNS", "30"))

PAD_ID = int(os.environ.get("PAD_ID", "0"))
EOS_ID = int(os.environ.get("EOS_ID", "2"))

def make_batch_input_ids(batch: int, seqlen: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.full((batch, seqlen), 1, dtype=torch.int32, device=device)

def run_generate(runner: ModelRunner, batch_input_ids: torch.Tensor):
    device = batch_input_ids.device
    batch_input_lengths = torch.full(
        (batch_input_ids.shape[0],),
        batch_input_ids.shape[1],
        dtype=torch.int32,
        device=device
    )

    # TRT-LLM 0.12 signature
    try:
        return runner.generate(
            batch_input_ids=batch_input_ids,
            batch_input_lengths=batch_input_lengths,
            max_new_tokens=OUTPUT_LEN,
            end_id=EOS_ID,
            pad_id=PAD_ID,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
        )
    except TypeError:
        # Some builds don't require lengths
        return runner.generate(
            batch_input_ids=batch_input_ids,
            max_new_tokens=OUTPUT_LEN,
            end_id=EOS_ID,
            pad_id=PAD_ID,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
        )

def benchmark(engine_dir: str) -> dict:
    runner = ModelRunner.from_dir(engine_dir)
    batch_input_ids = make_batch_input_ids(BATCH, INPUT_LEN)

    for _ in range(WARMUP):
        _ = run_generate(runner, batch_input_ids)

    t0 = time.perf_counter()
    for _ in range(RUNS):
        _ = run_generate(runner, batch_input_ids)
    t1 = time.perf_counter()

    sec_per_req = (t1 - t0) / RUNS
    tokens_per_req = BATCH * (INPUT_LEN + OUTPUT_LEN)
    tok_per_s_est = tokens_per_req / sec_per_req

    return {
        "engine_dir": engine_dir,
        "batch": BATCH,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "runs": RUNS,
        "sec_per_req": sec_per_req,
        "tok_per_s_est": tok_per_s_est,
    }

def pretty(d: dict, title: str):
    print(f"\n=== {title} ===")
    for k, v in d.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    base = benchmark(BASE_ENGINE_DIR)
    kv = benchmark(KV_ENGINE_DIR)

    pretty(base, "Baseline (paged_kv_cache=disable)")
    pretty(kv, "KV Optimized (paged_kv_cache=enable)")

    speedup = base["sec_per_req"] / kv["sec_per_req"]
    print(f"\nLatency speedup: {speedup:.2f}x (higher is better)")

