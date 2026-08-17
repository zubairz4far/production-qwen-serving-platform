from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def run_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def gpu_metadata() -> list[dict[str, Any]]:
    raw = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not raw:
        return []
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "name": parts[0],
                "uuid": parts[1],
                "driver_version": parts[2],
                "memory_total_mib": parts[3],
                "compute_capability": parts[4],
            }
        )
    return rows


def docker_metadata() -> dict[str, Any]:
    container_id = run_command(["docker", "compose", "ps", "-q", "vllm"])
    image = None
    image_id = None
    if container_id:
        image = run_command(["docker", "inspect", "--format", "{{.Config.Image}}", container_id])
        image_id = run_command(["docker", "inspect", "--format", "{{.Image}}", container_id])
    return {
        "compose_version": run_command(["docker", "compose", "version", "--short"]),
        "vllm_container_id": container_id,
        "vllm_image": image,
        "vllm_image_id": image_id,
    }


def service_metadata(base_url: str, api_key: str) -> dict[str, Any]:
    headers = {"authorization": f"Bearer {api_key}"}
    try:
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=10.0) as client:
            ready = client.get("/health/ready")
            models = client.get("/v1/models", headers=headers)
        model_ids = []
        if models.status_code < 400:
            model_ids = [item.get("id") for item in models.json().get("data", [])]
        return {
            "ready_status": ready.status_code,
            "models_status": models.status_code,
            "model_ids": model_ids,
        }
    except (httpx.HTTPError, ValueError, TypeError):
        return {"ready_status": None, "models_status": None, "model_ids": []}


def collect(base_url: str, api_key: str) -> dict[str, Any]:
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpus": gpu_metadata(),
        "docker": docker_metadata(),
        "service": service_metadata(base_url, api_key),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture reproducibility metadata for a GPU run.")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="change-me-public")
    parser.add_argument("--output", default="evals/results/runtime_metadata.json")
    args = parser.parse_args()
    metadata = collect(args.base_url, args.api_key)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
