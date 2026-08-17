# GPU benchmark protocol

This project separates deterministic CI evidence from hardware-dependent serving measurements.

## Required host

- Linux host with NVIDIA Container Toolkit and Docker Compose GPU support
- NVIDIA GPU with enough VRAM for `Qwen/Qwen3-1.7B` plus the LoRA adapter
- Hugging Face access to the base model and adapter when required

## Run

```bash
cp .env.example .env
# replace PUBLIC_API_KEYS and VLLM_API_KEY before exposure

docker compose up -d --build
curl -f http://localhost:8080/health/ready

python -m scripts.run_gpu_suite \
  --base-url http://localhost:8080 \
  --api-key '<PUBLIC_API_KEY>' \
  --model tool-calling \
  --concurrencies 1,2,4,8 \
  --requests-per-point 32
```

The suite writes four artifacts under `evals/results/gpu_suite/`:

- `runtime_metadata.json` — GPU name/UUID, driver, VRAM, Compose version, vLLM image ID, Git commit, loaded model aliases
- `tool_calling.json` — per-case served tool-call behavior and strict aggregate metrics
- `concurrency_sweep.json` — request throughput, output-token throughput, client TTFT, vLLM engine TTFT, queue time, ITL, latency, and SLO decisions by concurrency
- `suite_summary.json` — compact portfolio summary and best SLO-passing point

## Measurement rules

1. Do not publish a result without its runtime metadata artifact.
2. Do not mix results from different GPUs, driver versions, model revisions, or vLLM image IDs in the same performance row.
3. Keep client-observed TTFT and vLLM engine TTFT separate; they measure different boundaries.
4. Run tool correctness before choosing a high-throughput configuration. A faster configuration that changes correctness is not a valid winner.
5. Keep raw result JSON in `evals/results/` locally; only commit reviewed, named baselines.

## Hugging Face Jobs

Hugging Face Jobs can supply T4, L4, A10G, A100, H200 and other accelerators. Jobs are paid compute, so this repository does not automatically start one from CI. A GPU Job should only be launched deliberately after choosing a hardware flavor and accepting its cost.
