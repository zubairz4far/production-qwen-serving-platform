# Security Policy

## Security model

The public FastAPI gateway is the only service exposed by the default Docker Compose configuration. The vLLM API remains on the internal Compose network and is protected with a separate upstream bearer key.

The gateway enforces:

- public bearer-key authentication for `/v1/*`
- separation between public caller keys and the private vLLM key
- a fixed model allowlist
- bounded request bodies
- bounded inference concurrency with queue timeout and `503 + Retry-After`
- generated or caller-supplied request IDs without forwarding caller credentials upstream
- low-cardinality Prometheus labels
- no dynamic LoRA loading in the production baseline

## Secrets

Never commit `.env`, Hugging Face tokens, public gateway API keys, or the vLLM upstream API key. Rotate any secret immediately if it is exposed.

## Network boundary

The default Compose file publishes only the gateway on port `8080`. Prometheus is bound to loopback. vLLM is intentionally not mapped to a host port.

For internet-facing deployments, put the gateway behind TLS and a managed identity/API gateway or equivalent perimeter control.

## Model serving boundary

The baseline loads the LoRA adapter statically at vLLM startup. Dynamic runtime LoRA loading is intentionally excluded from the trusted production path.

## Resource exhaustion

`MAX_REQUEST_BYTES`, `MAX_CONCURRENCY`, and `QUEUE_TIMEOUT_SECONDS` are hard resource controls. They must be tuned from real GPU measurements rather than increased without load testing.

## Reporting

Do not include live credentials, private prompts, customer data, or model-access tokens in a public vulnerability report. Provide a minimal reproduction using synthetic data.
