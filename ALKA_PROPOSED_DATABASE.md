# Proposed ALKA Folder Structure

**Hierarchy**: Model → Hardware → Software → Config → Timestamp

---

## Structure Overview

```
ALKA_DATABASE/
├── data/
│   ├── <model-name>/                    # Normalized model name
│   │   ├── <hardware>/                  # GPU type (mi355x, mi300x, b200, h200, h100)
│   │   │   ├── <backend>/               # Software stack (vllm, sglang, trtllm, atom)
│   │   │   │   ├── <config>/            # Config variant (tp4, tp8, tp4_conc256, etc.)
│   │   │   │   │   ├── YYYYMMDD_HHMMSS/ # Timestamped run
│   │   │   │   │   │   ├── amd_summary.csv (or nv_summary.csv)
│   │   │   │   │   │   ├── amd_kernels.csv (or nv_kernels.csv)
│   │   │   │   │   │   └── metadata.json
│   │   │   │   │   └── latest/          # Symlink → most recent YYYYMMDD_HHMMSS/
│   │   │
│   └── datasets/                        # Generic auxiliary data (kernel shapes, references, etc.)
│       ├── gemm_shapes/
│       ├── attention_patterns/
│       └── ...
│
├── outputs/
│   └── <model-name>/
│       ├── <hw>/                        # Single-vendor analysis (e.g., mi355x)
│       │   ├── YYYYMMDD_HHMMSS/
│       │   │   ├── amd_categorized.csv
│       │   │   ├── ai_analysis.json
│       │   │   ├── report.html
│       │   │   └── run_info.txt
│       │   └── latest/                  # Symlink → most recent YYYYMMDD_HHMMSS/
│       │
│       └── <hw_vs_hw>/                  # Cross-vendor comparison (e.g., mi355x_vs_b200)
│           ├── YYYYMMDD_HHMMSS/
│           │   ├── amd_categorized.csv
│           │   ├── nv_categorized.csv
│           │   ├── merged_results.csv
│           │   ├── ai_analysis.json
│           │   ├── report.html
│           │   └── run_info.txt
│           └── latest/                  # Symlink → most recent YYYYMMDD_HHMMSS/
```

---

## Concrete Example

```
data/
├── llama-3.1-8b/
│   ├── mi355x/
│   │   ├── atom/
│   │   │   └── tp1/
│   │   │       ├── 20260410_093324/
│   │   │       │   ├── amd_summary.csv
│   │   │       │   ├── amd_kernels.csv
│   │   │       │   └── metadata.json
│   │   │       └── latest@ → 20260410_093324/
│   │   ├── vllm/
│   │   │   └── tp1/
│   │   │       ├── 20260414_084321/
│   │   │       │   ├── amd_summary.csv
│   │   │       │   ├── amd_kernels.csv
│   │   │       │   └── metadata.json
│   │   │       └── latest@ → 20260414_084321/
│   │   └── sglang/
│   │       └── tp1/
│   │           ├── 20260410_100717/
│   │           │   ├── amd_summary.csv
│   │           │   ├── amd_kernels.csv
│   │           │   └── metadata.json
│   │           └── latest@ → 20260410_100717/
│   │
│   ├── mi300x/
│   │   └── vllm/
│   │       └── tp1/
│   │           ├── 20260414_090813/
│   │           │   ├── amd_summary.csv
│   │           │   ├── amd_kernels.csv
│   │           │   └── metadata.json
│   │           └── latest@ → 20260414_090813/
│   │
│   ├── h200/
│   │   └── vllm/
│   │       └── tp1/
│   │           ├── 20260410_093415/
│   │           │   ├── nv_summary.csv
│   │           │   ├── nv_kernels.csv
│   │           │   └── metadata.json
│   │           └── latest@ → 20260410_093415/
│   │
│   └── b200/
│       └── trtllm/
│           └── tp1/
│               ├── 20260410_093725/
│               │   ├── nv_summary.csv
│               │   ├── nv_kernels.csv
│               │   └── metadata.json
│               └── latest@ → 20260410_093725/
│
├── gpt-oss-120b/
│   ├── mi355x/
│   │   └── atom/
│   │       └── tp1/
│   │           ├── 20260410_084612/
│   │           │   ├── amd_summary.csv
│   │           │   ├── amd_kernels.csv
│   │           │   └── metadata.json
│   │           └── latest@ → 20260410_084612/
│   │
│   ├── mi300x/
│   │   └── vllm/
│   │       └── tp8/
│   │           ├── 20260408_065413/
│   │           │   ├── amd_summary.csv
│   │           │   ├── amd_kernels.csv
│   │           │   └── metadata.json
│   │           ├── 20260409_035002/
│   │           │   ├── amd_summary.csv
│   │           │   ├── amd_kernels.csv
│   │           │   └── metadata.json
│   │           └── latest@ → 20260409_035002/
│   │
│   ├── h200/
│   │   ├── vllm/
│   │   │   └── tp8/
│   │   │       ├── 20260414_175219/
│   │   │       │   ├── nv_summary.csv
│   │   │       │   ├── nv_kernels.csv
│   │   │       │   └── metadata.json
│   │   │       └── latest@ → 20260414_175219/
│   │   └── trtllm/
│   │       └── tp8/
│   │           ├── 20260414_183814/
│   │           │   ├── nv_summary.csv
│   │           │   ├── nv_kernels.csv
│   │           │   └── metadata.json
│   │           └── latest@ → 20260414_183814/
│   │
│   └── b200/
│       └── vllm/
│           └── tp8/
│               ├── 20260408_134301/
│               │   ├── nv_summary.csv
│               │   ├── nv_kernels.csv
│               │   └── metadata.json
│               └── latest@ → 20260408_134301/
│
└── datasets/
    ├── gemm_shapes/
    ├── attention_patterns/
    └── kernel_references/
```

```
outputs/
├── llama-3.1-8b/
│   ├── mi355x/
│   │   ├── 20260415_100000/
│   │   │   ├── amd_categorized.csv
│   │   │   ├── ai_analysis.json
│   │   │   ├── report.html
│   │   │   └── run_info.txt
│   │   └── latest@ → 20260415_100000/
│   │
│   └── mi300x_vs_h200/
│       ├── 20260415_110000/
│       │   ├── amd_categorized.csv
│       │   ├── nv_categorized.csv
│       │   ├── merged_results.csv
│       │   ├── ai_analysis.json
│       │   ├── report.html
│       │   └── run_info.txt
│       └── latest@ → 20260415_110000/
│
├── gpt-oss-120b/
│   ├── mi300x/
│   │   ├── 20260409_103008/
│   │   │   ├── amd_categorized.csv
│   │   │   ├── ai_analysis.json
│   │   │   ├── report.html
│   │   │   └── run_info.txt
│   │   └── latest@ → 20260409_103008/
│   │
│   └── mi300x_vs_b200/
│       ├── 20260415_120000/
│       │   ├── amd_categorized.csv
│       │   ├── nv_categorized.csv
│       │   ├── merged_results.csv
│       │   ├── ai_analysis.json
│       │   ├── report.html
│       │   └── run_info.txt
│       └── latest@ → 20260415_120000/
└── qwen-3.5-35b/
    └── mi300x_vs_h200/
        ├── 20260414_110400/
        │   ├── amd_categorized.csv
        │   ├── nv_categorized.csv
        │   ├── merged_results.csv
        │   ├── ai_analysis.json
        │   ├── report.html
        │   └── run_info.txt
        └── latest@ → 20260414_110400/
```

---

## Naming Conventions

### Model Names (normalized)
- `llama-3.1-8b` ← `meta-llama-Llama-3.1-8B-Instruct`
- `gpt-oss-120b` ← `openai-gpt-oss-120b`
- `kimi-k2.5-mxfp4` ← `amd-Kimi-K2.5-MXFP4`
- `kimi-k2.5-nvfp4` ← `nvidia-Kimi-K2.5-NVFP4`
- `minimax-m2.5` ← `MiniMaxAI-MiniMax-M2.5`
- `minimax-m2.5-nvfp4` ← `nvidia-MiniMax-M2.5-NVFP4`
- `deepseek-r1-0528-mxfp4` ← `DeepSeek-R1-0528-MXFP4-Preview`
- `qwen-3.5-35b` ← `Qwen-Qwen3.5-35B-A3B`
- `glm-4.7-fp8` ← `GLM-4.7-FP8`

### Hardware
- AMD: `mi355x`, `mi325x`, `mi300x`
- NVIDIA: `b200`, `h200`, `h100`

### Backend
- `vllm`, `sglang`, `trtllm`, `atom`

### Config
- Tensor Parallel: `tp1`, `tp2`, `tp4`, `tp8`
- With Concurrency: `tp4_conc16_isl1024_osl1024`, `tp8_conc256_isl1024_osl1024`, `tp4_conc4_isl1024_osl1024`

### Timestamp
- Format: `YYYYMMDD_HHMMSS`
- Example: `20260413_142844`

---

## Metadata File (metadata.json)

Each timestamped run should include a metadata file:

```json
{
  "timestamp": "20260413_142844",
  "model": {
    "hf_id": "amd-Kimi-K2.5-MXFP4",
    "normalized_name": "kimi-k2.5-mxfp4"
  },
  "hardware": {
    "type": "mi355x",
    "runner": "mi355x-p02-g57",
    "runner_label": "alka-mi355x-p02-g57"
  },
  "software": {
    "backend": "atom",
    "version": "1.2.3",
    "config": "tp4_conc256"
  },
  "benchmark": {
    "config_path": "3rdparty/benchNap/config/batch_run/config_kimi_k25.yaml",
    "tp": 4,
    "concurrency": 256,
    "input_seq_len": 128,
    "output_seq_len": 128
  },
  "git": {
    "commit": "53d0301",
    "branch": "ajith_main"
  },
  "workflow": {
    "run_id": "12345678",
    "run_url": "https://github.com/ROCm/ALKA/actions/runs/12345678"
  }
}
```

---

## Benefits

### Easy Browsing
```bash
# Find all vLLM runs for llama-3.1-8b on MI355X
ls data/llama-3.1-8b/mi355x/vllm/

# Access latest vLLM run for llama-3.1-8b on MI355X
cat data/llama-3.1-8b/mi355x/vllm/tp1/latest/amd_summary.csv

# Compare all backends for gpt-oss-120b on MI300X
ls data/gpt-oss-120b/mi300x/

# Find all cross-vendor comparisons
find outputs -name "*_vs_*"

# View latest analysis report
open outputs/gpt-oss-120b/mi300x/latest/report.html
```

### Easy Cross-Vendor Matching
```bash
# Compare same config across AMD/NVIDIA
data/kimi-k2.5-mxfp4/mi355x/vllm/tp4_conc256/
data/kimi-k2.5-nvfp4/b200/vllm/tp4_conc256/
```

### Version Tracking
Multiple timestamped runs under same config enable tracking experiments over time.

### Symlink "latest"
- Every config directory contains a `latest/` symlink pointing to the most recent timestamped folder
- Enables quick access to the latest results without knowing the exact timestamp
- Automatically updated by CI workflows and migration scripts when new runs are added
- Examples:
  - `data/llama-3.1-8b/mi355x/vllm/tp1/latest/` → `20260414_084321/`
  - `outputs/gpt-oss-120b/mi300x/latest/` → `20260409_103008/`

---

## Migration Notes

1. **Current `data/profiles/`** → Restructure to new hierarchy
2. **Current `data/profiles_batch/`** → Merge into same hierarchy
3. **Current `data/0203/`, `data/0210/`** → Archive or discard (legacy format)
4. **Current `data/gemm_shapes/`** → Move to `data/datasets/gemm_shapes/`
5. **Current `outputs/`** → Restructure to new hierarchy
6. **CI workflows** → Update to save using new paths
7. **ALKA tools** → Update `alka_parser.py`, `alka_report.py` to read/write new structure
