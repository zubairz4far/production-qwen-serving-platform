# Production Qwen Serving Platform

A production-shaped inference platform for serving **Qwen3** and the evaluated LoRA adapter `zubairz4far/qwen3-1.7b-tool-calling` through an OpenAI-compatible API.

The goal is not to wrap a model in one endpoint. The project separates the GPU inference runtime from an independently testable edge/control layer so reliability, security, backpressure, observability, and SLO evaluation can be validated without requiring a GPU in CI.

## V0.1 architecture

```text
client
  |
  | Bearer API key + x-request-id
  v
FastAPI serving gateway :8080
  |-- auth boundary
  |-- model allowlist
  |-- request-size limit
  |-- concurrency/backpressure gate
  |-- SSE streaming proxy
  |-- /health/live
  |-- /health/ready
  |-- /metrics
  |
  v
vLLM v0.26.0 :8000
  |-- Qwen/Qwen3-1.7B base
  |-- static LoRA: zubairz4far/qwen3-1.7b-tool-calling
  |-- OpenAI-compatible chat API
  |-- Hermes tool-call parser
  |-- Qwen3 reasoning parser
  |-- native /health + /metrics
  |
  +--> NVIDIA GPU

Prometheus scrapes gateway + vLLM metrics.
```

## Why the LoRA is loaded statically

vLLM supports LoRA adapters at server startup with `--enable-lora --lora-modules name=path`. Runtime LoRA loading exists, but vLLM's documentation warns that dynamic loading has security risks and should only be used in an isolated trusted environment. This project therefore uses static startup loading for the production baseline.

## Quick start on an NVIDIA GPU host

```bash
cp .env.example .env
# set HF_TOKEN if the model or environment needs it
# replace both API keys

docker compose up --build
```

Then:

```bash
python scripts/smoke_test.py \
  --base-url http://localhost:8080 \
  --api-key '<PUBLIC_API_KEY>' \
  --model tool-calling
```

The public gateway is on `:8080`. The vLLM service is intentionally not published to the host in the default Compose file.

## OpenAI-compatible request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Authorization: Bearer <PUBLIC_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"tool-calling",
    "messages":[{"role":"user","content":"What is the status of order 123?"}],
    "temperature":0
  }'
```

Streaming is proxied as SSE by setting `"stream": true`.

## Tool-calling serving configuration

The vLLM service enables Qwen3 tool-call parsing with:

```text
--enable-auto-tool-choice
--tool-call-parser hermes
--reasoning-parser qwen3
```

The LoRA alias is `tool-calling`, so clients select the fine-tuned adapter through the normal OpenAI `model` field.

## Backpressure instead of unbounded overload

The gateway holds a bounded generation semaphore. When all inference slots are occupied, new requests get only a short queue window; after that the gateway returns `503` with `Retry-After: 1`. This prevents an overloaded GPU from accumulating an unbounded request queue.

`MAX_CONCURRENCY` is intentionally an operational tuning parameter. It must be benchmarked on the actual GPU instead of being presented as a universal optimum.

## Readiness semantics

`/health/live` only confirms that the gateway process is alive.

`/health/ready` is stricter. It checks the vLLM health endpoint and confirms that the configured LoRA model alias appears in `/v1/models`. A gateway whose process is up but whose model failed to load is therefore **not ready**.

## Metrics

The gateway exposes low-cardinality Prometheus metrics for:

- request counts by fixed route and status class
- request duration
- in-flight generation slots
- queue saturation rejections
- upstream transport failures

vLLM's native `/metrics` endpoint is scraped separately, preserving engine-level metrics such as token and scheduler behavior.

## Load benchmark

Run a streaming benchmark against a real GPU deployment:

```bash
python scripts/benchmark.py \
  --base-url http://localhost:8080 \
  --api-key '<PUBLIC_API_KEY>' \
  --model tool-calling \
  --requests 100 \
  --concurrency 8 \
  --output evals/results/t4-c8.json
```

The benchmark records success rate, request throughput, total latency, and time-to-first-byte/TTFT proxy timing.

Evaluate the result against the current target policy:

```bash
python -m scripts.evaluate_slo --result evals/results/t4-c8.json
```

The thresholds in `evals/slo_policy.json` are **targets**, not claimed measurements. Real portfolio metrics should only be added after a GPU benchmark has actually run.

## CI

GPU-less CI deliberately validates what can be proven without pretending to benchmark inference hardware:

1. Ruff linting
2. gateway unit/API tests
3. auth-boundary regression
4. readiness model-presence regression
5. SSE proxy regression
6. SLO evaluator regression
7. gateway Docker build
8. Docker Compose configuration validation

A later GPU benchmark artifact will be kept separate from the deterministic control-plane CI evidence.

## Current V0.1 boundaries

- vLLM runtime is pinned to `v0.26.0`; GPU compatibility must still be validated on the target NVIDIA host.
- `MAX_LORA_RANK=64` is a safe configurable startup ceiling, not a claim about the adapter's exact rank; it should be reduced to the adapter's actual rank after inspecting its deployed config to avoid wasted memory.
- one gateway process uses an in-process semaphore; multi-replica global admission control requires a shared rate/admission layer.
- authentication is a static bearer-key baseline; production internet exposure should normally sit behind a managed identity/API gateway.
- current benchmark measures client-observed first streamed bytes rather than engine-native TTFT; both will be compared once real vLLM metrics are collected.
- no GPU throughput or latency number is claimed until the benchmark runs on a named GPU.

## Next milestones

1. run on an NVIDIA T4/A10/L4 and record exact GPU, CUDA, vLLM, model, adapter, context, and concurrency settings
2. add tool-call correctness smoke/eval against the served LoRA
3. sweep concurrency and context sizes and publish throughput/TTFT/p95 curves
4. correlate gateway results with vLLM Prometheus engine metrics
5. add graceful shutdown/draining and retry-safe client behavior
6. add optional multi-GPU tensor parallel profile
7. freeze the best measured configuration as v1.0.0
