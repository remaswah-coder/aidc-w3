# W3d3: vLLM vs. Static Batching A/B Testing & Benchmarking

This repository contains the implementation, benchmarking scripts, and evaluation reports for comparing static batching against continuous batching using **vLLM** and **Qwen/Qwen2.5-1.5B-Instruct**. 

Developed as part of the AI Engineering coursework / SDA AI Data Center Bootcamp.

---

## 📂 Project Structure

```text
├── ab_report.json          # Generated A/B testing performance metrics and speedup ratios
├── benchmark_sweep.py      # Async concurrency sweep client (httpx-based)
└── README.md               # Project documentation


## 📊 Key Performance Metrics

The benchmarking sweep evaluates throughput (`tokens/sec`) across multiple concurrency levels to compare traditional static batching against continuous batching using vLLM.

| Concurrency Level | Baseline (Static Batching) | vLLM (Continuous Batching) | Speedup |
| :---: | :---: | :---: | :---: |
| **8** | ~96.6 tokens/s | ~359.3 tokens/s | **3.72x** 🚀 |

### ✅ Verification Status

![Lab Verification - Green Check Pass]
<img scr="1714" height="110" alt="Screenshot 2026-09-01 164444" src="https://github.com/user-attachments/assets/79d42d74-e074-451b-9158-dcef59b77436" 



---

## 🛠️ Tech Stack & Tools

* **Inference Engine:** [vLLM](https://github.com/vllm-project/vllm) (OpenAI-compatible server)
* **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
* **Concurrency Testing:** Python (`asyncio`, `httpx`)
* **Version Control:** Git & GitHub

---

## ⚙️ How to Run

1. **Launch the vLLM Server:**
   ```bash
   python3 -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-1.5B-Instruct \
       --port 8000 \
       --gpu-memory-utilization 0.80

2. **Execute the Concurrency Sweep & Generate Report:**
   Run the benchmark script in your environment to test concurrencies `[1, 4, 8]` and automatically write the metrics to `ab_report.json`.

3. **Verify Results:**
   Ensure the output matches the validation criteria and passes the test:
   ```text
   GREEN CHECK: PASS



