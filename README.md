# AIDC Week 3 - Day 2 (W3D2)

Welcome to the **W3D2** branch of the **aidc-w3** repository! This directory contains the scripts, configuration files, and baseline results developed during the Saudi Digital Academy AI Data Center Bootcamp.

## 📂 Repository Structure

* **`baselines.json`**: Contains the baseline evaluation metrics and results generated from the day's lab work.
* **`batch_check.json`**: Configuration or log file used for batch processing and validation checks.
* **`profile.json`**: User or environment profile configurations for the pipeline.
* **`colab_scaffold.py`**: Scaffold script utilized for structuring and setting up the Google Colab environment.
* **`verify_cell.py`**: Validation script used to test and verify code execution and cell outputs.

## 📊 Result Snapshot & Output
<img width="1108" height="192" alt="Screenshot 2026-08-31 133644" src="https://github.com/user-attachments/assets/462face5-563a-49b0-bd7b-4859dc5f97f3" />


## 🚀 Overview & Usage

This workspace contains automated scripts and generated outputs designed to handle model baselines and infrastructure verification. 

1. **Environment Setup**: Ensure your environment matches the required configuration using the provided scripts.
2. **Verification**: Run `verify_cell.py` to validate dependencies and check environment health.
3. **Results**: The `baselines.json` file serves as the primary output record for this session's benchmarks.


## 🧪 Extra Lab: The Paged KV Allocator (PagedAttention Simulation)

This extra lab explores memory management strategies for KV-cache to improve LLM serving concurrency without requiring a GPU, comparing a naive contiguous slab allocator against a block-pool allocator (similar to PagedAttention).

### 📈 Lab Summary & Metrics
* **Slab Allocator (Naive):** Reserved memory for the max length (4096 tokens), supporting only **18 resident sequences** under a 2 GB budget.
* **Block-Pool Allocator (PagedAttention):** Allocated memory dynamically block-by-block (16 tokens per block), supporting all **60 resident sequences** with zero rejections.
* **Concurrency Advantage:** Achieved a **3.33x** improvement in concurrency.

### 🔍 Verification Output & Result

> ** (Verification Result):**
<img width="1392" height="82" alt="Screenshot 2026-08-31 145809" src="https://github.com/user-attachments/assets/6c88878e-f137-4dfb-8cc0-9a9ef9b779d4" />



```text
Extra Lab: reference: slab 18 resident, block-pool 60 resident, advantage 3.33x
GREEN CHECK: PASS
```

## 🐛 Extra Lab: The Prompt That Wasn't As Long As You Asked (Bug Lab)

Investigated and fixed a silent token-shortfall bug in `prompt_of_len()`, where requesting large token lengths (like 4096) silently returned shorter lengths (around 3008 tokens) due to list slicing past the text ceiling without throwing an error.

### 🛠️ Fix & Validation
* **The Root Cause:** Python list slicing past the end of tokens returns whatever exists without raising exceptions.
* **The Solution:** Implemented a dynamic text growth loop inside `prompt_of_len()` combined with an explicit `assert len(ids) == n_tokens` check to ensure fail-fast safety.

### 🔍 Verification Output & Result

> ** (Verification Result):**

<img width="1212" height="314" alt="Screenshot 2026-08-31 150247" src="https://github.com/user-attachments/assets/a457116e-ba7b-4624-9fcd-263a0c2ddd98" />
```text
for n in [128, 512, 2048, 4096]:
    p = prompt_of_len(n)
    actual = len(tok(p)["input_ids"])
    assert actual == n, f"prompt_of_len({n}) produced {actual} tokens"
GREEN CHECK: PASS
