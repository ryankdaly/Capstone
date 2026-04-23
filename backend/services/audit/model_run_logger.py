"""structured per-run logger for model thinking/code artifacts.

writes one JSON file per pipeline run:
    logs/model_runs/<run_id>.json

separate from:
- logs/agent.log          (raw debug dump)
- logs/audit/<run_id>.jsonl  (append-only audit trail)
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from backend.api.schemas.pipeline import PipelineRequest
from backend.config import settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRunLogger:
    """Mutable structured logger for one JSON artifact per run."""

    def __init__(self, log_dir: str | None = None) -> None:
        self._log_dir = Path(log_dir or settings.pipeline.model_run_log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, run_id: UUID) -> Path:
        return self._log_dir / f"{run_id}.json"

    def _read(self, run_id: UUID) -> dict[str, Any]:
        path = self._path(run_id)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, run_id: UUID, payload: dict[str, Any]) -> None:
        path = self._path(run_id)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def start_run(self, run_id: UUID, request: PipelineRequest) -> None:
        with self._lock:
            payload = {
                "run_id": str(run_id),
                "started_at": _utc_now(),
                "completed_at": None,
                "status": "running",
                "request": {
                    "requirement_text": request.requirement_text,
                    "safety_standard": request.safety_standard.value,
                    "target_language": request.target_language.value,
                    "max_iterations": request.max_iterations,
                    "stage": request.stage.value,
                    "run_tests": request.run_tests,
                    "retry_context": request.retry_context,
                },
                "iterations": [],
                "error": None,
            }
            self._write(run_id, payload)

    def complete_run(
        self,
        run_id: UUID,
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        with self._lock:
            payload = self._read(run_id)
            if not payload:
                return
            payload["completed_at"] = _utc_now()
            payload["status"] = status
            payload["error"] = error
            self._write(run_id, payload)

    def log_agent_start(
        self,
        run_id: UUID,
        *,
        iteration: int,
        agent: str,
        attempt: int | None = None,
        cycle: int | None = None,
        fix_round: int | None = None,
    ) -> None:
        with self._lock:
            payload = self._read(run_id)
            if not payload:
                return
            slot = self._ensure_slot(
                payload,
                iteration=iteration,
                agent=agent,
                attempt=attempt,
                cycle=cycle,
                fix_round=fix_round,
            )
            slot.setdefault("started_at", _utc_now())
            self._write(run_id, payload)

    def log_token(
        self,
        run_id: UUID,
        *,
        iteration: int,
        agent: str,
        token: str,
        attempt: int | None = None,
        cycle: int | None = None,
        fix_round: int | None = None,
    ) -> None:
        with self._lock:
            payload = self._read(run_id)
            if not payload:
                return
            slot = self._ensure_slot(
                payload,
                iteration=iteration,
                agent=agent,
                attempt=attempt,
                cycle=cycle,
                fix_round=fix_round,
            )
            slot["streamed_text"] = slot.get("streamed_text", "") + token
            self._write(run_id, payload)

    def log_output(
        self,
        run_id: UUID,
        *,
        iteration: int,
        agent: str,
        output: dict[str, Any],
        attempt: int | None = None,
        cycle: int | None = None,
        fix_round: int | None = None,
    ) -> None:
        with self._lock:
            payload = self._read(run_id)
            if not payload:
                return
            slot = self._ensure_slot(
                payload,
                iteration=iteration,
                agent=agent,
                attempt=attempt,
                cycle=cycle,
                fix_round=fix_round,
            )
            slot["completed_at"] = _utc_now()
            slot["output"] = output
            self._promote_useful_fields(agent, slot, output)
            self._write(run_id, payload)

    def log_error(
        self,
        run_id: UUID,
        *,
        iteration: int,
        agent: str,
        error: str,
        attempt: int | None = None,
        cycle: int | None = None,
        fix_round: int | None = None,
        skipped: bool = False,
    ) -> None:
        with self._lock:
            payload = self._read(run_id)
            if not payload:
                return
            slot = self._ensure_slot(
                payload,
                iteration=iteration,
                agent=agent,
                attempt=attempt,
                cycle=cycle,
                fix_round=fix_round,
            )
            slot["completed_at"] = _utc_now()
            slot["error"] = error
            slot["skipped"] = skipped
            self._write(run_id, payload)

    def log_verification_result(
        self,
        run_id: UUID,
        *,
        iteration: int,
        result: dict[str, Any],
        cycle: int | None = None,
    ) -> None:
        with self._lock:
            payload = self._read(run_id)
            if not payload:
                return
            slot = self._ensure_slot(
                payload,
                iteration=iteration,
                agent="dafny_verifier",
                cycle=cycle,
            )
            slot["started_at"] = slot.get("started_at", _utc_now())
            slot["completed_at"] = _utc_now()
            slot["output"] = result
            self._promote_useful_fields("dafny_verifier", slot, result)
            self._write(run_id, payload)

    def log_test_result(
        self,
        run_id: UUID,
        *,
        iteration: int,
        result: dict[str, Any],
        fix_round: int | None = None,
    ) -> None:
        with self._lock:
            payload = self._read(run_id)
            if not payload:
                return
            slot = self._ensure_slot(
                payload,
                iteration=iteration,
                agent="checker_tests",
                fix_round=fix_round,
            )
            slot["started_at"] = slot.get("started_at", _utc_now())
            slot["completed_at"] = _utc_now()
            slot["output"] = result
            slot["pytest_output"] = result.get("pytest_output", "")
            self._write(run_id, payload)

    def _ensure_iteration(self, payload: dict[str, Any], iteration: int) -> dict[str, Any]:
        iterations: list[dict[str, Any]] = payload.setdefault("iterations", [])
        while len(iterations) < iteration:
            iterations.append(
                {
                    "iteration": len(iterations) + 1,
                    "agents": {},
                }
            )
        return iterations[iteration - 1]

    def _slot_key(
        self,
        *,
        attempt: int | None,
        cycle: int | None,
        fix_round: int | None,
    ) -> str:
        return f"attempt={attempt};cycle={cycle};fix_round={fix_round}"

    def _ensure_slot(
        self,
        payload: dict[str, Any],
        *,
        iteration: int,
        agent: str,
        attempt: int | None = None,
        cycle: int | None = None,
        fix_round: int | None = None,
    ) -> dict[str, Any]:
        iteration_obj = self._ensure_iteration(payload, iteration)
        agents = iteration_obj.setdefault("agents", {})
        agent_obj = agents.setdefault(agent, {"runs": {}})
        key = self._slot_key(attempt=attempt, cycle=cycle, fix_round=fix_round)
        slot = agent_obj["runs"].setdefault(
            key,
            {
                "attempt": attempt,
                "cycle": cycle,
                "fix_round": fix_round,
                "started_at": None,
                "completed_at": None,
                "streamed_text": "",
                "thinking": "",
                "code": "",
                "dafny_code": "",
                "test_cases": [],
                "output": None,
                "error": None,
                "skipped": False,
            },
        )
        return slot

    def _promote_useful_fields(
        self,
        agent: str,
        slot: dict[str, Any],
        output: dict[str, Any],
    ) -> None:
        reasoning = output.get("reasoning_trace")
        if isinstance(reasoning, str) and reasoning.strip():
            slot["thinking"] = reasoning

        if agent == "actor":
            slot["code"] = output.get("source_code", "") or ""
            slot["dafny_code"] = output.get("dafny_spec", "") or ""

        elif agent == "checker":
            test_cases = output.get("test_cases")
            if isinstance(test_cases, list):
                slot["test_cases"] = test_cases

        elif agent == "dafny_architect":
            slot["dafny_code"] = output.get("dafny_source", "") or output.get("dafny_spec", "") or ""

        elif agent == "dafny_verifier":
            # no model thinking/code here; keep full output only
            pass

        elif agent == "policy":
            # no code, just thinking + policy output
            pass