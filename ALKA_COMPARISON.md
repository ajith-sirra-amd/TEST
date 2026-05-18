# Benchmark Configuration Comparison: NVIDIA B200 vs AMD MI355X

This document summarizes available benchmark runs across two hardware platforms: **NVIDIA B200** (using TRTLLM, VLLM, or SGLANG engines) and **AMD MI355X** (using ATOM or VLLM).  
All configurations have **Output Sequence Length (OSL) = 1024** – omitted from tables for brevity.

## ✅ Table 1: Comparable Configurations (Both NVIDIA & AMD Data Exist)

Use these for direct hardware comparison.  
*Note: engines differ – NVIDIA may use TRTLLM/VLLM/SGLANG, AMD uses ATOM or VLLM.*

| Model | TP | EP | CONC | ISL | NVIDIA Files | AMD Files |
|-------|----|----|------|-----|--------------|-----------|
| **DeepSeek-V4-Pro** | 8 | DISABLED | 16 | 1024 | TRTLLM, VLLM | ATOM |
| **GLM5** | 4 | DISABLED | 256 | 1024 | SGLANG | ATOM |
| | 4 | DISABLED | 256 | 8192 | SGLANG | ATOM |
| | 4 | DISABLED | 4 | 1024 | SGLANG | ATOM |
| | 4 | DISABLED | 4 | 8192 | SGLANG | ATOM |
| **KIMI-K2.5** | 4 | DISABLED | 64 | 1024 | VLLM | ATOM |
| | 8 | DISABLED | 4 | 1024 | VLLM | ATOM |
| | 8 | DISABLED | 4 | 8192 | VLLM | ATOM |
| **MINIMAX-M2.5** | 2 | TRUE | 512 | 8192 | VLLM | ATOM |
| | 8 | DISABLED | 4 | 1024 | VLLM | ATOM |
| **QWEN3.5** | 2 | DISABLED | 128 | 1024 | SGLANG | ATOM |
| | 2 | DISABLED | 128 | 8192 | SGLANG | ATOM |
| | 4 | DISABLED | 4 | 1024 | SGLANG | ATOM |
| | 4 | DISABLED | 4 | 8192 | SGLANG | ATOM |

## ❌ Table 2: Missing / Incomplete Configurations (Only One Vendor)

These cannot be used for direct hardware comparison.  
**AMD‑only** runs – useful for single‑vendor analysis or future test expansion. (No NVIDIA‑only configs found in current dataset.)

| Model | TP | EP | CONC | ISL | Present Only On | Engine (if known) |
|-------|----|----|------|-----|-----------------|-------------------|
| **DeepSeek-R1-0528** | 4 | DISABLED | 256 | 1024 | AMD | ATOM |
| | 4 | DISABLED | 256 | 8192 | AMD | ATOM |
| | 8 | DISABLED | 4 | 1024 | AMD | ATOM |
| | 8 | DISABLED | 4 | 8192 | AMD | ATOM |
| **GPT-OSS-120B** | 1 | DISABLED | 128 | 1024 | AMD | VLLM |
| | 1 | DISABLED | 128 | 8192 | AMD | VLLM |
| | 1 | DISABLED | 256 | 1024 | AMD | VLLM |
| | 1 | DISABLED | 256 | 8192 | AMD | VLLM |
| | 4 | DISABLED | 4 | 1024 | AMD | VLLM |
| | 4 | DISABLED | 4 | 8192 | AMD | VLLM |
| **KIMI-K2.5** | 4 | DISABLED | 64 | 8192 | AMD | ATOM |
| | 4 | DISABLED | 128 | 1024 | AMD | ATOM |
| | 4 | DISABLED | 128 | 8192 | AMD | ATOM |
| **MINIMAX-M2.5** | 1 | DISABLED | 128 | 8192 | AMD | ATOM |
| | 1 | DISABLED | 512 | 1024 | AMD | ATOM |
| | 2 | DISABLED | 1024 | 1024 | AMD | ATOM |
| | 8 | DISABLED | 4 | 8192 | AMD | ATOM |
| **QWEN3.5** | 2 | DISABLED | 256 | 1024 | AMD | ATOM |
| | 2 | DISABLED | 256 | 8192 | AMD | ATOM |

## 📌 Recommendations for Fair Comparison

For each comparable configuration (Table 1), extract these metrics from the JSON files:
- Output Throughput (tokens/sec)
- End‑to‑End Latency (avg, p50, p90, p99)
- Time to First Token (TTFT)
- Inter‑Token Latency (ITL)
- GPU utilization (if available)
- Power / Energy (if available)

Be aware that software stacks differ across vendors and may influence results independently of hardware.
