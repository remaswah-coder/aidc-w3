# AIDC W3D1 — Profile Inference on a Real GPU

This lab profiles Qwen/Qwen2.5-1.5B-Instruct on a Google Colab Tesla T4. It compares FP16 and INT8 across three context lengths, then tests batch 1 against batch 8 to show why GPU utilisation is not the same as useful throughput.

## Objective

* Measure resident VRAM at context lengths 512, 2048, and 4096.
* Compare FP16 and INT8 memory usage and generation speed.
* Measure mean GPU utilisation during single-request decoding.
* Compare batch-1 and batch-8 throughput.
* Produce profile.json and batch_check.json as evidence.

## Predictions

Predictions recorded before opening Colab:

| Measurement | Prediction |
| :--- | :--- |
| FP16 weight memory | 3 GB |
| INT8 weight memory | 1.5 GB |
| FP16 resident VRAM at context 512 | 4 GB |
| FP16 resident VRAM at context 4096 | 8 GB |
| Single-request decode utilisation | 1% |

The weight-memory predictions used:
$$\text{memory} = \text{parameters} \times \text{bytes per parameter}$$

* **FP16:** $1.5\text{B} \times 2\text{ bytes} = 3\text{ GB}$
* **INT8:** $1.5\text{B} \times 1\text{ byte} = 1.5\text{ GB}$

The resident-VRAM prediction assumed that the longer context would add about 4 GB because the KV cache grows with context length.

## Environment

| Component | Value |
| :--- | :--- |
| Runtime | Google Colab |
| GPU | Tesla T4, 15360 MiB |
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Transformers | 4.46.* |
| Accelerate | 1.1.* |
| BitsAndBytes | 0.49.2 |

The lab deliberately uses Transformers directly and does not install vLLM. Colab's existing Torch remains in place, avoiding the Torch and NumPy compatibility changes that a vLLM installation would introduce.

## Method

For each dtype, the model was loaded once and profiled at three context lengths:

```python
rows = []
for dtype in ["fp16", "int8"]:
    model = load(dtype)
    for context in [512, 2048, 4096]:
        row = profile(model, dtype, context)
        print(row)
        rows.append(row)
    del model
    free_vram()
```

Each run:

Created a prompt at the requested context length.

Performed an eight-token warm-up.

Recorded resident PyTorch VRAM.

Sampled nvidia-smi utilisation every two seconds.

Generated 128 new tokens.

Calculated tokens per second.


<img width="770" height="256" alt="Screenshot 2026-08-30 143147" src="https://github.com/user-attachments/assets/30ad7258-301c-46a6-ac86-9376a7c82730" />



## Findings

* Resident VRAM increased with context length for both dtypes.
* FP16 used more VRAM than INT8 at every context length.
* Single-request utilisation rose with context, but utilisation alone did not describe throughput.
* Batching greatly improved total throughput (from 23.0 to 187.6 tokens/s) by giving the GPU more work to process in parallel.

## Artifacts

The experiment produced:
* `profile.json`
* `batch_check.json`


Verification
Final result:

Plaintext
```
rows: 6, dtypes: ['int8', 'fp16'], contexts: [512, 2048, 4096]
batch-1 tokens/s: 23.0, batch-8 tokens/s: 187.6
GREEN CHECK: PASS
```



## Key Takeaway
GPU utilisation answers whether the GPU had work during a sampling interval. It does not answer how much useful work was completed. Throughput, latency, memory use, batch size, and context length must be considered together when evaluating inference performance.
