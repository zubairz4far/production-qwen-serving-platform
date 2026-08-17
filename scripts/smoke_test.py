from __future__ import annotations

import argparse

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="change-me-public")
    parser.add_argument("--model", default="tool-calling")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=120.0) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        models = client.get("/v1/models", headers={"authorization": f"Bearer {args.api_key}"})
        chat = client.post(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {args.api_key}"},
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": "Reply with exactly: serving-ok"}],
                "temperature": 0,
                "max_tokens": 16,
            },
        )

    live.raise_for_status()
    ready.raise_for_status()
    models.raise_for_status()
    chat.raise_for_status()
    payload = chat.json()
    if not payload.get("choices"):
        raise SystemExit("Chat response did not contain choices.")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
