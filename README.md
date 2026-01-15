# TensorRT-LLM KV Cache Benchmark

This repository provides a reproducible benchmarking workflow to compare **TensorRT-LLM inference performance** with and without **paged KV cache optimization**.

---

## Overview

The benchmark measures:
- Average latency per request
- Estimated token throughput (tokens/sec)

It uses **synthetic fixed-length token inputs** to isolate GPU inference performance and remove tokenizer or dataset variability.

---

## What This Benchmark Does

- Builds two TensorRT-LLM engines:
  - Baseline (`paged_kv_cache=disable`)
  - KV-optimized (`paged_kv_cache=enable`)
- Runs warmup iterations
- Runs timed inference iterations
- Compares latency and throughput

No real dataset is used.

---

## Input Data

There is **no dataset** involved.

Input tokens are generated synthetically:

```python
torch.full((batch, seq_len), 1, dtype=torch.int32)
```

This ensures:
- Deterministic inputs
- No tokenizer overhead
- Focus on runtime execution only

---

## Prerequisites

### Hardware
- NVIDIA GPU (Ampere+ recommended)

### Software
- Python 3.10
- CUDA Toolkit
- TensorRT-LLM v0.12.0
- PyTorch (CUDA-enabled)

---

## Model Preparation

### Download Model (Example: TinyLlama)

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir="/root/hf_models/tinyllama"
)
```

---

### Convert to TensorRT-LLM Checkpoint

```bash
python TensorRT-LLM/examples/llama/convert_checkpoint.py   --model_dir /root/hf_models/tinyllama   --output_dir /root/trt_ckpts/tinyllama_trt   --dtype float16   --tp_size 1
```

---

## Build Engines

### Baseline Engine

```bash
trtllm-build   --checkpoint_dir /root/trt_ckpts/tinyllama_trt   --output_dir /root/engines/base   --max_batch_size 8   --max_seq_len 2048   --paged_kv_cache disable
```

### KV Optimized Engine

```bash
trtllm-build   --checkpoint_dir /root/trt_ckpts/tinyllama_trt   --output_dir /root/engines/kv_opt   --max_batch_size 8   --max_seq_len 2048   --paged_kv_cache enable
```

---

## Running the Benchmark

### Environment Variables

```bash
export BASE_ENGINE_DIR=/root/engines/base
export KV_ENGINE_DIR=/root/engines/kv_opt

export BATCH=8
export INPUT_LEN=512
export OUTPUT_LEN=512
export WARMUP=10
export RUNS=30
```

### Run

```bash
python kv_cache_bench.py
```

---

## Example Output

```text
=== Baseline (paged_kv_cache=disable) ===
sec_per_req: 5.38
tok_per_s_est: 1522

=== KV Optimized (paged_kv_cache=enable) ===
sec_per_req: 5.82
tok_per_s_est: 1407

Latency speedup: 0.92x
```

---

## Notes

- KV cache benefits depend on sequence length and batch size
- For short sequences, overhead may outweigh gains
- This is a microbenchmark, not end-to-end application latency

---

## License

MIT License
