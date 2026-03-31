"""Pipeline orchestrator — the state machine that drives the Actor-Checker loop.

~300 lines. Deterministic control flow, non-deterministic LLM outputs.
This is intentionally NOT LangGraph/CrewAI — it's a simple, auditable
state machine that gives total control over the feedback loop.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator
from uuid import UUID

from backend.api.schemas.agents import (
    CheckerVerdict,
    CodeCandidate,
    FeedbackMessage,
    PolicyVerdict,
    VerificationResult,
)
from backend.api.schemas.pipeline import (
    IterationRecord,
    PipelineRequest,
    PipelineStage,
    PipelineState,
    PipelineStatus,
    StreamEvent,
    StreamEventType,
    stage_enabled,
)
from backend.services.agents.actor import ActorAgent
from backend.services.agents.checker import CheckerAgent
from backend.services.agents.policy import PolicyAgent
from backend.services.audit.logger import AuditLogger
from backend.services.feedback import compose_feedback
from backend.services.llm.client import LLMClient
from backend.services.rag.retriever import StandardsRetriever
from backend.services.verification.dafny_runner import DafnyRunner

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Runs the full HPEMA generation pipeline.

    Lifecycle:
        1. Initialize pipeline state
        2. For each iteration (up to max_iterations):
           a. Actor generates code + Dafny spec
           b. Checker reviews code  }  run in parallel
              Dafny verifies spec   }
           c. Policy agent audits compliance
           d. If all pass → break
           e. Else → compose feedback → next iteration
        3. Return final state (COMPLETED or FAILED)
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        dafny_runner: DafnyRunner | None = None,
        retriever: StandardsRetriever | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._dafny = dafny_runner or DafnyRunner()
        self._retriever = retriever or StandardsRetriever()
        self._audit = audit_logger or AuditLogger()

        # Agents
        self._actor = ActorAgent(self._llm)
        self._checker = CheckerAgent(self._llm)
        self._policy = PolicyAgent(self._llm)

        # Exposed after run() completes — the CLI reads this.
        self.last_state: PipelineState | None = None

    async def run(
        self, request: PipelineRequest
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the pipeline, yielding SSE events as it progresses.

        Respects `request.stage` — the pipeline stops after the configured stage.
        Stages: ACTOR → CHECKER → POLICY. If stage=ACTOR, only code generation
        runs and the result is treated as successful (no checker/dafny/policy).
        """
        max_stage = request.stage
        state = PipelineState(
            request=request,
            status=PipelineStatus.RUNNING,
        )
        run_id = state.run_id

        self._audit.log(run_id, "pipeline_start", data={
            **request.model_dump(),
            "stage": max_stage.value,
        })

        feedback: FeedbackMessage | None = None
        best_score: float = -1.0
        best_iteration: int = 0

        # Actor-only has no feedback loop. Checker+ can loop on Dafny/Checker feedback.
        effective_max_iterations = (
            1 if max_stage == PipelineStage.ACTOR
            else request.max_iterations
        )

        try:
            for i in range(1, effective_max_iterations + 1):
                state.current_iteration = i

                iteration = IterationRecord(iteration=i)

                # --- ACTOR (always runs) ---
                yield self._event(run_id, StreamEventType.AGENT_START, "actor")
                self._audit.log(run_id, "agent_start", agent="actor", data={"iteration": i})

                code_candidate = await self._actor.run(
                    requirement=request.requirement_text,
                    language=request.target_language.value,
                    standard=request.safety_standard.value,
                    feedback=feedback,
                )
                iteration.code_candidate = code_candidate

                yield self._event(
                    run_id,
                    StreamEventType.AGENT_OUTPUT,
                    "actor",
                    {
                        "language": code_candidate.language,
                        "has_dafny": bool(code_candidate.dafny_spec),
                        "source_code": code_candidate.source_code,
                        "dafny_spec": code_candidate.dafny_spec,
                        "reasoning_trace": code_candidate.reasoning_trace,
                    },
                )
                self._audit.log(
                    run_id, "agent_output", agent="actor",
                    data=code_candidate.model_dump(),
                )

                # --- CHECKER + DAFNY (parallel) — requires stage >= CHECKER ---
                checker_report = None
                verification_result = None

                # Guard: if Actor produced no Dafny spec, note it in feedback
                has_dafny_spec = bool(code_candidate.dafny_spec and code_candidate.dafny_spec.strip())

                if stage_enabled(PipelineStage.CHECKER, max_stage):
                    yield self._event(run_id, StreamEventType.AGENT_START, "checker")
                    yield self._event(run_id, StreamEventType.AGENT_START, "dafny_verifier")

                    checker_task = self._checker.run(
                        source_code=code_candidate.source_code,
                        dafny_spec=code_candidate.dafny_spec,
                        language=request.target_language.value,
                        standard=request.safety_standard.value,
                    )
                    dafny_task = self._dafny.verify(code_candidate.dafny_spec)

                    checker_report, verification_result = await asyncio.gather(
                        checker_task, dafny_task
                    )

                    iteration.checker_report = checker_report
                    iteration.verification_result = verification_result

                    yield self._event(
                        run_id, StreamEventType.AGENT_OUTPUT, "checker",
                        {
                            "verdict": checker_report.verdict.value,
                            "issues": len(checker_report.issues),
                            "issues_detail": [iss.model_dump() for iss in checker_report.issues],
                            "test_cases": checker_report.test_cases,
                            "reasoning_trace": checker_report.reasoning_trace,
                        },
                    )
                    yield self._event(
                        run_id, StreamEventType.AGENT_OUTPUT, "dafny_verifier",
                        {
                            "verified": verification_result.verified,
                            "solver_output": verification_result.solver_output,
                            "failing_assertions": verification_result.failing_assertions,
                            "execution_time_seconds": verification_result.execution_time_seconds,
                        },
                    )

                    self._audit.log(run_id, "agent_output", agent="checker", data=checker_report.model_dump())
                    self._audit.log(run_id, "verification_result", data=verification_result.model_dump())

                # --- POLICY — requires stage >= POLICY ---
                policy_verdict = None

                if stage_enabled(PipelineStage.POLICY, max_stage):
                    yield self._event(run_id, StreamEventType.AGENT_START, "policy")

                    policy_context = self._retriever.retrieve(
                        query=f"{request.requirement_text} {request.safety_standard.value}",
                        standard=request.safety_standard.value,
                    )

                    policy_verdict = await self._policy.run(
                        source_code=code_candidate.source_code,
                        dafny_spec=code_candidate.dafny_spec,
                        standard=request.safety_standard.value,
                        checker_report=checker_report,
                        verification_result=verification_result,
                        policy_context=policy_context,
                    )
                    iteration.policy_verdict = policy_verdict

                    yield self._event(
                        run_id, StreamEventType.AGENT_OUTPUT, "policy",
                        {
                            "compliant": policy_verdict.compliant,
                            "risk_level": policy_verdict.risk_level.value,
                            "violations": [v.model_dump() for v in policy_verdict.violations],
                            "recommendations": policy_verdict.recommendations,
                            "reasoning_trace": policy_verdict.reasoning_trace,
                        },
                    )
                    self._audit.log(run_id, "agent_output", agent="policy", data=policy_verdict.model_dump())

                # --- CONVERGENCE CHECK ---
                if max_stage == PipelineStage.ACTOR:
                    # Actor-only — always pass (single inference, no verification)
                    all_pass = True
                elif max_stage == PipelineStage.CHECKER:
                    # Checker stage — pass requires Dafny verified + Checker pass
                    dafny_ok = verification_result is not None and verification_result.verified
                    checker_ok = checker_report is not None and checker_report.verdict == CheckerVerdict.PASS
                    all_pass = dafny_ok and checker_ok

                    feedback = compose_feedback(
                        iteration=i,
                        checker_report=checker_report,
                        verification_result=verification_result,
                        policy_verdict=None,
                    )

                    # Best-so-far tracking — detect regressions
                    score = (
                        (1.0 if dafny_ok else 0.0)
                        + (1.0 if checker_ok else 0.0)
                        + (0.5 if has_dafny_spec else 0.0)
                    )
                    if score > best_score:
                        best_score = score
                        best_iteration = i
                    elif i > 1 and score < best_score:
                        regression_note = (
                            f"REGRESSION: Iteration {i} scored worse than iteration {best_iteration}. "
                            f"Do NOT regress — preserve what worked before."
                        )
                        feedback = FeedbackMessage(
                            iteration=feedback.iteration,
                            checker_feedback=feedback.checker_feedback,
                            verification_feedback=feedback.verification_feedback,
                            policy_feedback=feedback.policy_feedback,
                            priority_summary=f"{regression_note} | {feedback.priority_summary}",
                        )

                    # If no Dafny spec was generated, add explicit feedback
                    if not has_dafny_spec:
                        feedback = FeedbackMessage(
                            iteration=feedback.iteration,
                            checker_feedback=feedback.checker_feedback,
                            verification_feedback=feedback.verification_feedback,
                            policy_feedback=feedback.policy_feedback,
                            priority_summary=f"MISSING DAFNY SPEC: You MUST generate a dafny_spec with requires/ensures clauses. | {feedback.priority_summary}",
                        )

                    iteration.feedback = feedback
                else:
                    # Full pipeline — all three must pass
                    dafny_ok = verification_result is not None and verification_result.verified
                    checker_ok = checker_report is not None and checker_report.verdict == CheckerVerdict.PASS
                    policy_ok = policy_verdict is not None and policy_verdict.compliant
                    all_pass = dafny_ok and checker_ok and policy_ok

                    feedback = compose_feedback(
                        iteration=i,
                        checker_report=checker_report,
                        verification_result=verification_result,
                        policy_verdict=policy_verdict,
                    )

                    # Best-so-far tracking
                    score = (
                        (1.0 if dafny_ok else 0.0)
                        + (1.0 if checker_ok else 0.0)
                        + (1.0 if policy_ok else 0.0)
                        + (0.5 if has_dafny_spec else 0.0)
                    )
                    if score > best_score:
                        best_score = score
                        best_iteration = i
                    elif i > 1 and score < best_score:
                        regression_note = (
                            f"REGRESSION: Iteration {i} scored worse than iteration {best_iteration}. "
                            f"Do NOT regress — preserve what worked before."
                        )
                        feedback = FeedbackMessage(
                            iteration=feedback.iteration,
                            checker_feedback=feedback.checker_feedback,
                            verification_feedback=feedback.verification_feedback,
                            policy_feedback=feedback.policy_feedback,
                            priority_summary=f"{regression_note} | {feedback.priority_summary}",
                        )

                    if not has_dafny_spec:
                        feedback = FeedbackMessage(
                            iteration=feedback.iteration,
                            checker_feedback=feedback.checker_feedback,
                            verification_feedback=feedback.verification_feedback,
                            policy_feedback=feedback.policy_feedback,
                            priority_summary=f"MISSING DAFNY SPEC: You MUST generate a dafny_spec with requires/ensures clauses. | {feedback.priority_summary}",
                        )

                    iteration.feedback = feedback

                iteration.completed_at = datetime.now(timezone.utc)
                state.iterations.append(iteration)

                yield self._event(
                    run_id, StreamEventType.ITERATION_COMPLETE, data={
                        "iteration": i, "all_pass": all_pass,
                        "summary": feedback.priority_summary if feedback else "",
                    },
                )

                if all_pass:
                    state.status = PipelineStatus.AWAITING_APPROVAL
                    state.final_code = code_candidate.source_code
                    state.final_proof = code_candidate.dafny_spec
                    break
            else:
                state.status = PipelineStatus.FAILED
                state.error = f"Failed to converge after {effective_max_iterations} iterations"

        except Exception as e:
            logger.exception("Pipeline error in run %s", run_id)
            state.status = PipelineStatus.FAILED
            state.error = str(e)
            yield self._event(
                run_id, StreamEventType.AGENT_ERROR, data={"error": str(e)},
            )

        state.completed_at = datetime.now(timezone.utc)
        self.last_state = state
        self._audit.log(run_id, "pipeline_complete", data={
            "status": state.status.value,
            "iterations": len(state.iterations),
            "stage": max_stage.value,
        })

        yield self._event(
            run_id, StreamEventType.PIPELINE_COMPLETE, data={
                "status": state.status.value,
                "iterations": len(state.iterations),
                "run_id": str(run_id),
                "stage": max_stage.value,
            },
        )

    # --- Helpers ---

    @staticmethod
    def _event(
        run_id: UUID,
        event_type: StreamEventType,
        agent: str | None = None,
        data: dict | None = None,
    ) -> StreamEvent:
        return StreamEvent(
            event_type=event_type,
            run_id=run_id,
            agent=agent,
            data=data or {},
        )
