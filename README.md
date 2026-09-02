# W3D4: Model Quantization (AWQ), vLLM Serving & Function-Calling Smoke Test

This repository contains the implementation, configuration logs, and verification artifacts for **Day 4** of the AI Data Center Bootcamp. The core objective of this lab is to quantize and serve an instruction-tuned model locally using AWQ, optimize KV-cache allocation under a constrained GPU memory budget, and lock the configuration to pass a strict function-calling smoke test.

---

## 🚀 Project Overview

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct-AWQ`
- **Inference Engine:** vLLM (with fused AWQ kernels)
- **Environment:** Custom Python 3.10 Virtual Environment (built for compatibility)
- **Tool-Call Parser:** `hermes` (with automatic tool-choice enabled)
- **Memory Optimization:** Configured with a GPU memory utilization flag of `0.85`, yielding optimal KV-cache block allocation (~11.7 GB VRAM).

---

## 🛠️ Key Components

1. **Environment Setup:** Custom isolated Python 3.10 environment configured via script to handle library dependencies (`vllm==0.6.5`, `autoawq==0.2.7`, `transformers`, etc.).
2. **AWQ Server Launch:** Serving the quantized weights locally via vLLM's OpenAI-compatible API endpoint on port `8050` / `8000`.
3. **Function-Calling Smoke Test (`smoke_test.py`):** Automated evaluation suite validating 10 distinct prompts, including multi-tool execution and distractor compliance.
4. **Model Locking (`model-lock.md`):** A strict configuration record locking the exact startup flags, quantization type, and score threshold.

---

## 📊 Results & Artifacts

- **Smoke Test Score:** `10/10` (Passed)
- **Distractor Cleanliness:** `True` (Maintained call-free behavior on distractor prompts as required for consumer safety).
- **VRAM Footprint:** ~11,723 MiB (Maximizing available KV-cache space within the 0.85 utilization limit).

---

## 📂 Repository Structure

```text
├── model-lock.md         # Strict configuration lock record and flags
├── smoke_result.json     # Detailed JSON output of the 10/10 function-calling smoke test
├── server.log            # vLLM startup and graph capturing logs
└── README.md             # Project documentation

```

## 🧪 Verification

The setup includes an automated verification script (`verify_cell.py`) that checks both the smoke test results and the strict formatting of the model lock file. 

To run the verification locally in your notebook, use:

```bash
python verify_cell.py

```


smoke score: 10/10, distractor clean: True
model-lock.md: all fields filled
GREEN CHECK: PASS



Here is the successful execution proof:
<img width="892" height="140" alt="Screenshot 2026-09-02 135540" src="https://github.com/user-attachments/assets/f6a0da04-8970-4314-b5b1-d6cebb2f29cb" />



