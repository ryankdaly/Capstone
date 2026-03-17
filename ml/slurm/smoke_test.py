#!/usr/bin/env python3
"""Smoke test — validates each layer of the HPEMA stack independently.

Run this on the GPU node after vLLM is up:
    python ml/slurm/smoke_test.py

Tests (in order):
  1. vLLM health check
  2. Raw chat completion (can the model talk?)
  3. Constrained decoding (does guided_json enforce our schema?)
  4. Backend health check
  5. Full pipeline call (if backend is running)
"""

from __future__ import annotations

import json
import sys
import time

import httpx

VLLM_BASE = "http://localhost:8001"
VLLM_API = f"{VLLM_BASE}/v1"
BACKEND_BASE = "http://localhost:8000"
MODEL = "microsoft/Phi-4-Reasoning-Plus"

# Schema for constrained decoding test — a minimal CodeCandidate
CODE_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "source_code": {"type": "string"},
        "dafny_spec": {"type": "string"},
        "reasoning_trace": {"type": "string"},
        "language": {"type": "string"},
    },
    "required": ["source_code", "language"],
}


def test_vllm_health() -> bool:
    """Test 1: Is vLLM responding?"""
    print("\n[1/5] vLLM health check...", end=" ")
    try:
        r = httpx.get(f"{VLLM_BASE}/health", timeout=5.0)
        if r.status_code == 200:
            print("OK")
            return True
        print(f"FAIL (status {r.status_code})")
        return False
    except httpx.ConnectError:
        print("FAIL (connection refused — is vLLM running?)")
        return False


def test_raw_completion() -> bool:
    """Test 2: Can the model generate a basic response?"""
    print("[2/5] Raw chat completion...", end=" ")
    try:
        r = httpx.post(
            f"{VLLM_API}/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                "max_tokens": 20,
                "temperature": 0.0,
            },
            timeout=60.0,
        )
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            print(f"OK — model said: {content!r}")
            return True
        print(f"FAIL (status {r.status_code}: {r.text[:200]})")
        return False
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def test_constrained_decoding() -> bool:
    """Test 3: Does guided_json enforce our schema?"""
    print("[3/5] Constrained decoding (guided_json)...", end=" ")
    try:
        r = httpx.post(
            f"{VLLM_API}/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You generate safety-critical code. Respond with JSON only.",
                    },
                    {
                        "role": "user",
                        "content": "Write a C function that adds two integers with overflow checking.",
                    },
                ],
                "max_tokens": 1024,
                "temperature": 0.2,
                "extra_body": {"guided_json": CODE_CANDIDATE_SCHEMA},
            },
            timeout=120.0,
        )
        if r.status_code != 200:
            print(f"FAIL (status {r.status_code}: {r.text[:200]})")
            return False

        content = r.json()["choices"][0]["message"]["content"]

        # Verify it's valid JSON matching the schema
        parsed = json.loads(content)
        if "source_code" not in parsed or "language" not in parsed:
            print(f"FAIL (missing required fields: {list(parsed.keys())})")
            return False

        print(f"OK — got valid JSON with {len(parsed['source_code'])} chars of code")
        print(f"      Language: {parsed['language']}")
        return True

    except json.JSONDecodeError as e:
        print(f"FAIL (invalid JSON: {e})")
        return False
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def test_backend_health() -> bool:
    """Test 4: Is the FastAPI backend responding?"""
    print("[4/5] Backend health check...", end=" ")
    try:
        r = httpx.get(f"{BACKEND_BASE}/health", timeout=5.0)
        if r.status_code == 200:
            print("OK")
            return True
        print(f"FAIL (status {r.status_code})")
        return False
    except httpx.ConnectError:
        print("SKIP (backend not running)")
        return False


def test_pipeline() -> bool:
    """Test 5: Can the full pipeline run? (requires backend)"""
    print("[5/5] Full pipeline (1 iteration)...", end=" ")
    try:
        r = httpx.get(f"{BACKEND_BASE}/health", timeout=5.0)
        if r.status_code != 200:
            print("SKIP (backend not running)")
            return False
    except httpx.ConnectError:
        print("SKIP (backend not running)")
        return False

    try:
        with httpx.stream(
            "POST",
            f"{BACKEND_BASE}/api/v1/pipeline/run",
            json={
                "requirement_text": "Write a function that returns the absolute value of an integer without using stdlib.",
                "safety_standard": "DO_178C",
                "target_language": "C",
                "max_iterations": 1,
            },
            timeout=300.0,
        ) as response:
            events = []
            event_type = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: ") and event_type:
                    events.append(event_type)
                    event_type = None

            if "pipeline_complete" in events:
                print(f"OK — received {len(events)} events: {events}")
                return True
            else:
                print(f"FAIL — no pipeline_complete event. Got: {events}")
                return False

    except Exception as e:
        print(f"FAIL ({e})")
        return False


def main():
    print("=" * 50)
    print("HPEMA Smoke Test")
    print("=" * 50)

    results = {}

    # Test 1-2: vLLM
    results["vllm_health"] = test_vllm_health()
    if not results["vllm_health"]:
        print("\nvLLM is not running. Remaining tests require it.")
        print("Start vLLM first, then re-run this script.")
        sys.exit(1)

    results["raw_completion"] = test_raw_completion()
    results["constrained_decoding"] = test_constrained_decoding()

    # Test 4-5: Backend
    results["backend_health"] = test_backend_health()
    results["pipeline"] = test_pipeline()

    # Summary
    print("\n" + "=" * 50)
    print("Results:")
    for name, passed in results.items():
        status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
        print(f"  {name}: {status}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{passed}/{total} passed")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
