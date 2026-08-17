# Production Qwen Serving Platform

A production-shaped inference platform for serving **Qwen3** and the evaluated LoRA adapter `zubairz4far/qwen3-1.7b-tool-calling` behind an OpenAI-compatible API.

The project is intentionally more than `vllm serve`: it separates a GPU inference runtime from an independently testable edge/control plane so authentication, model policy, backpressure, streaming, readiness, observability, correctness evaluation, and SLO measurement can be verified independently.

## Current milestone — v0.2.0

V0.2 adds the measurement layer needed before claiming production serving performance:

- served tool-call correctness evaluation through `/v1/chat/completions`
- 24-case hard suite covering required tool calls, missing arguments, no-tool negatives, hard negatives, and prompt injection
- strict tool-name + exact-argument scoring
- hallucinated-tool and clarification-restraint metrics
- client-observed TTFT plus vLLM per-request engine TTFT, queue time, mean ITL, and token throughput
- concurrency sweeps with SLO decisions and automatic best passing point selection
- reproducibility capture for GPU, driver, VRAM, Git commit, Docker Compose, vLLM image ID, and loaded model aliases
- one-command measured GPU suite that writes reviewable JSON artifacts

No GPU throughput or latency number is claimed until that suite has actually run on named hardware.

## Architecture

```text
client
  |
  | Bearer API key + x-request-id
  v
FastAPI serving gateway :8080
  |-- public auth boundary
  |-- model allowlist
  |-- bounded request body
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

## Quick start on an NVIDIA GPU host

```bash
cp .env.example .env
# replace PUBLIC_API_KEYS and VLLM_API_KEY
# set HF_TOKEN if required

docker compose up -d --build
curl -f http://localhost:8080/health/ready
```

The public gateway is on `:8080`. vLLM itself is not published to the host in the default Compose profile.

## Served tool-call correctness

The V0.2 suite is derived from the same behavior family used to evaluate the fine-tuned model: `get_order`, `cancel_order`, `get_weather`, missing required arguments, explicit no-tool prompts, hard negatives, and prompt-injection attempts.

```bash
python -m scripts.evaluate_tool_calling \
  --base-url http://localhost:8080 \
  --api-key '<PUBLIC_API_KEY>' \
  --model tool-calling \
  --policy evals/tool_calling_targets.json \
  --output evals/results/tool_calling.json
```

The report includes overall accuracy, tool-selection accuracy, strict exact-call accuracy, parse validity, hallucinated-tool rate on non-tool cases, clarification restraint, prompt-injection strict accuracy, category accuracy, and client latency.

The target policy is an acceptance target, **not a pre-claimed score**. A measured run may fail it; the artifact should still be preserved and analyzed rather than hidden.

## GPU performance benchmark

The streaming benchmark requests usage reporting and records both client and engine-side metrics:

```bash
python -m scripts.benchmark \
  --base-url http://localhost:8080 \
  --api-key '<PUBLIC_API_KEY>' \
  --model tool-calling \
  --requests 100 \
  --concurrency 8 \
  --output evals/results/c8.json
```

Reported measurements include request success rate, requests/second, output tokens/second, end-to-end latency, client-observed TTFT, vLLM engine TTFT, vLLM queue time, mean inter-token latency, per-request token throughput, and server-metric coverage.

## Concurrency sweep

```bash
python -m scripts.sweep \
  --base-url http://localhost:8080 \
  --api-key '<PUBLIC_API_KEY>' \
  --model tool-calling \
  --concurrencies 1,2,4,8 \
  --requests-per-point 32 \
  --output evals/results/concurrency_sweep.json
```

Each point is evaluated against `evals/slo_policy.json`. The report chooses the highest-throughput point that still passes configured SLO targets; it does not simply choose the fastest overloaded setting.

## Full measured GPU suite

```bash
python -m scripts.run_gpu_suite \
  --base-url http://localhost:8080 \
  --api-key '<PUBLIC_API_KEY>' \
  --model tool-calling \
  --concurrencies 1,2,4,8 \
  --requests-per-point 32
```

This writes `runtime_metadata.json`, `tool_calling.json`, `concurrency_sweep.json`, and `suite_summary.json`. See `docs/GPU_BENCHMARK.md` for the reproducibility protocol.

## Runtime metadata

`python -m scripts.capture_runtime` records the Git commit, Python/platform information, NVIDIA GPU name/UUID, driver, total VRAM, compute capability, Docker Compose version, exact vLLM image reference/image ID, gateway readiness, and model aliases exposed by `/v1/models`.

A performance result without its runtime metadata should not be published as a portfolio baseline.

## Backpressure instead of an unbounded queue

The gateway holds a bounded generation semaphore. When all inference slots remain occupied beyond the short queue window, it returns `503` with `Retry-After: 1`. `MAX_CONCURRENCY` is an operational tuning parameter and must be chosen from measured hardware behavior rather than copied from another GPU.

## Readiness

`/health/live` only confirms that the gateway process is alive. `/health/ready` checks the vLLM health endpoint and verifies that the configured LoRA alias is actually present in `/v1/models`.

## Security baseline

- caller bearer tokens are never forwarded to vLLM
- vLLM uses a separate private upstream API key
- model IDs are allowlisted at the gateway
- request bodies are bounded while streaming in
- vLLM is not host-published by default
- the LoRA is loaded statically rather than through runtime adapter mutation
- gateway metrics use low-cardinality labels

See `SECURITY.md` for deployment boundaries.

## Deterministic CI

GPU-less CI verifies only claims that do not require inference hardware:

1. clean Python install
2. Ruff linting
3. unit/API/regression tests
4. tool-evaluation asset validation
5. SLO evaluator regression
6. gateway Docker build
7. Docker Compose validation

Hardware-dependent correctness and performance artifacts stay separate from CI fixtures so a synthetic number can never be presented as a real GPU measurement.

## V0.2 boundaries

- no real GPU benchmark is committed yet
- the default admission semaphore is per gateway process; multi-replica global admission needs a shared layer
- static bearer keys are a baseline, not a replacement for managed identity at an internet edge
- server per-request metrics depend on vLLM emitting the final usage/metrics chunk; coverage is reported explicitly
- performance still depends on prompt/output length, model revision, GPU, driver, vLLM image, and concurrency
- Hugging Face Jobs can be used for a controlled GPU run, but Jobs are paid compute and are not auto-triggered by CI

## Road to v1.0.0

1. run the V0.2 suite on a named NVIDIA GPU and commit a reviewed baseline
2. inspect served tool-calling regressions versus the original model benchmark
3. tune concurrency, context length, GPU memory utilization, and LoRA rank ceiling using measured evidence
4. add graceful draining/shutdown behavior
5. add a multi-GPU tensor-parallel profile only if a real use case justifies it
6. freeze the best measured configuration and final evidence as v1.0.0
