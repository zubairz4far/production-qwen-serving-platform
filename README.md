# Production Qwen Serving Platform

A production-shaped inference platform for serving **Qwen3** and the evaluated LoRA adapter `zubairz4far/qwen3-1.7b-tool-calling` behind an OpenAI-compatible API.

The project separates GPU inference from an independently testable FastAPI edge/control plane so authentication, model policy, backpressure, streaming, readiness, observability, tool-use safety, correctness evaluation, and SLO measurement can be verified independently.

## Current milestone — v0.3.0

V0.3 adds a deterministic, schema-aware tool-policy layer in front of the fine-tuned Qwen model and records the first reviewed live Kaggle T4 release baseline.

The policy guard enforces the failure modes discovered during adversarial evaluation:

- missing order IDs or cities return a concise clarification without invoking vLLM;
- explicit examples, documentation, quoted commands, and non-execution requests cannot trigger a tool call;
- grounded legitimate requests continue to the model;
- grounded requests containing fake-tool or schema-bypass injection text are reduced to the minimal trusted action before inference;
- the model still performs the final legitimate tool selection and exact argument emission for valid requests.

## Verified live T4 baseline

The reviewed baseline is committed at:

`evals/baselines/kaggle_t4_tool_policy_v0.3.json`

It was measured on **August 18, 2026** in a Kaggle GPU notebook using one **NVIDIA Tesla T4 (15 GB)** with:

- Qwen/Qwen3-1.7B
- LoRA `zubairz4far/qwen3-1.7b-tool-calling`
- LoRA rank 16
- vLLM 0.26.0
- PyTorch 2.11.0+cu129
- CUDA 12.9 build
- max model length 4096
- GPU memory utilization 0.88
- max sequences 16
- FastAPI tool-policy guard enabled

### Behavioral result

The frozen 24-case served benchmark passed **24/24**:

| Metric | Result |
|---|---:|
| Overall accuracy | **100%** |
| Tool selection accuracy | **100%** |
| Strict exact-call accuracy | **100%** |
| Tool-call parse validity | **100%** |
| Clarification restraint | **100%** |
| Prompt-injection strict accuracy | **100%** |
| Hallucinated tool rate on non-tool cases | **0%** |

Every benchmark category passed at 100%: valid tool requests, missing-order-ID clarification, missing-city clarification, ordinary no-tool requests, hard negatives, and prompt-injection cases.

The behavioral release policy reported **no failed checks**.

### Performance result

All tested concurrency points `1, 2, 4, 8` passed the configured SLO policy with 32/32 successful requests per point.

The best tested point inside the gateway's configured concurrency envelope was **concurrency 8**:

| Metric | Result |
|---|---:|
| Request success rate | **100%** |
| Throughput | **7.48 req/s** |
| Output throughput | **142.17 tok/s** |
| p95 end-to-end latency | **1.066 s** |
| p95 client TTFT | **211 ms** |
| p95 engine TTFT | **161 ms** |
| p95 engine queue time | **0.039 ms** |
| Server metric coverage | **100%** |

Concurrency 8 is the **best tested gateway point**, not a hardware-saturation claim. The sweep used 32 requests per point; larger repeated runs are appropriate for tighter capacity estimates.

## Architecture

```text
client
  |
  | Bearer API key + x-request-id
  v
FastAPI serving gateway :8080
  |-- public auth boundary
  |-- model allowlist
  |-- deterministic tool-policy guard
  |     |-- missing required arg -> synthetic clarification
  |     |-- documentation/no-execute -> tool_choice=none
  |     `-- grounded injection -> canonical trusted request
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
  `--> NVIDIA GPU

Prometheus scrapes gateway + vLLM metrics.
```

## Why the policy layer exists

The original fine-tuned adapter was strong at complete tool requests but weaker at restraint. The first live T4 behavioral baseline scored 75% overall: valid tool execution was strong, while missing required arguments and hard-negative prompts could still produce fabricated values or unnecessary tool calls.

Multiple corrective approaches were tested rather than hidden:

- system-prompt hardening did not solve the behavior reliably;
- broader SFT improved restraint but damaged routing;
- smaller checkpoint-selected SFT preserved routing but did not fix clarification;
- DPO reduced unnecessary tool calls while preserving routing, but clarification remained unreliable.

The production design therefore treats deterministic invariants as gateway responsibilities and leaves probabilistic semantic tool selection to the model. The final live guarded benchmark moved the same frozen served suite from the earlier **75% model-only baseline to 100% system-level accuracy**.

## Quick start on an NVIDIA GPU host

```bash
cp .env.example .env
# replace PUBLIC_API_KEYS and VLLM_API_KEY
# set HF_TOKEN if required

docker compose up -d --build
curl -f http://localhost:8080/health/ready
```

The public gateway is exposed on `:8080`. vLLM itself is not published to the host in the default Compose profile.

## Served tool-call correctness

```bash
python -m scripts.evaluate_tool_calling \
  --base-url http://localhost:8080 \
  --api-key '<PUBLIC_API_KEY>' \
  --model tool-calling \
  --policy evals/tool_calling_targets.json \
  --output evals/results/tool_calling.json
```

The evaluator reports overall accuracy, tool-selection accuracy, strict exact-call accuracy, parse validity, hallucinated-tool rate, clarification restraint, prompt-injection strict accuracy, category accuracy, and client latency.

## GPU performance benchmark

```bash
python -m scripts.benchmark \
  --base-url http://localhost:8080 \
  --api-key '<PUBLIC_API_KEY>' \
  --model tool-calling \
  --requests 100 \
  --concurrency 8 \
  --output evals/results/c8.json
```

Measurements include request success rate, requests/second, output tokens/second, end-to-end latency, client-observed TTFT, vLLM engine TTFT, vLLM queue time, mean inter-token latency, per-request token throughput, and server-metric coverage.

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

Each point is evaluated against `evals/slo_policy.json`. The report selects the highest-throughput point that still passes the configured SLOs.

## Full measured GPU suite

```bash
python -m scripts.run_gpu_suite \
  --base-url http://localhost:8080 \
  --api-key '<PUBLIC_API_KEY>' \
  --model tool-calling \
  --concurrencies 1,2,4,8 \
  --requests-per-point 32
```

This writes:

- `runtime_metadata.json`
- `tool_calling.json`
- `concurrency_sweep.json`
- `suite_summary.json`

See `docs/GPU_BENCHMARK.md` for the reproducibility protocol.

## Backpressure

The gateway holds a bounded generation semaphore. When all inference slots remain occupied beyond the queue window, it returns `503` with `Retry-After: 1`. `MAX_CONCURRENCY` is an operational tuning parameter and should be selected from measured hardware behavior.

## Readiness

`/health/live` confirms only that the gateway process is alive. `/health/ready` checks the vLLM health endpoint and verifies that the configured LoRA alias is present in `/v1/models`.

## Security baseline

- caller bearer tokens are never forwarded to vLLM;
- vLLM uses a separate private upstream API key;
- model IDs are allowlisted at the gateway;
- request bodies are bounded while streaming in;
- vLLM is not host-published by default;
- the LoRA is loaded statically rather than through runtime adapter mutation;
- missing required arguments fail closed into clarification;
- explicit non-execution requests cannot trigger tools;
- recognized fake-tool injection is removed from grounded requests before inference;
- gateway metrics use low-cardinality labels.

See `SECURITY.md` for deployment boundaries.

## Deterministic CI

GPU-less CI verifies claims that do not require inference hardware:

1. clean Python install
2. Ruff linting
3. unit/API/regression tests
4. tool-evaluation asset validation
5. SLO evaluator regression
6. gateway Docker build
7. Docker Compose validation

Live hardware-dependent correctness and performance evidence is kept separate from CI fixtures so synthetic measurements cannot be confused with GPU measurements.

## Current boundaries

- the deterministic policy currently covers the production tool family used by this project: `get_order`, `cancel_order`, and `get_weather`;
- the admission semaphore is per gateway process; multi-replica global admission needs a shared layer;
- static bearer keys are a baseline, not a replacement for managed identity at an internet edge;
- performance varies with prompt/output length, driver, GPU, model revision, context length, and concurrency;
- the current T4 sweep stops at gateway concurrency 8 and is not a saturation study;
- the current published performance sweep uses 32 requests per point.

## Next milestones

1. expand the deterministic policy from the current three-tool family into schema-driven reusable validators;
2. repeat the best T4 configuration with 100-128 requests per point for tighter capacity estimates;
3. test concurrency above 8 only after raising the gateway admission limit intentionally;
4. add graceful draining/shutdown behavior;
5. add a multi-GPU or tensor-parallel profile only when a real workload justifies it;
6. freeze a broader production release as v1.0.0 after the generalized policy and larger hardware validation pass.
