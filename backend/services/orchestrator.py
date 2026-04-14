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
    TestRunResult,
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
from backend.services.agents.dafny_architect import DafnyArchitectAgent
from backend.services.agents.policy import PolicyAgent
from backend.services.audit.logger import AuditLogger
from backend.services.feedback import compose_feedback
from backend.services.llm.client import LLMClient
from backend.services.rag.retriever import StandardsRetriever
from backend.services.testing.runner import TestRunner
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
        test_runner: TestRunner | None = None,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._dafny = dafny_runner or DafnyRunner()
        self._retriever = retriever or StandardsRetriever()
        self._audit = audit_logger or AuditLogger()
        self._test_runner = test_runner or TestRunner()

        # Agents
        self._actor = ActorAgent(self._llm)
        self._dafny_architect = DafnyArchitectAgent(self._llm)
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

                code_candidate = None
                async for _item in self._actor.run_streaming(
                    requirement=request.requirement_text,
                    language=request.target_language.value,
                    standard=request.safety_standard.value,
                    feedback=feedback,
                ):
                    if isinstance(_item, str):
                        yield self._event(run_id, StreamEventType.AGENT_TOKEN, "actor", {"token": _item})
                    else:
                        code_candidate = _item
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

                # --- DAFNY ARCHITECT + CHECKER + DAFNY VERIFIER ---
                # Requires stage >= CHECKER. DafnyArchitect runs first (sequential),
                # then Checker and DafnyRunner run in parallel against its output.
                checker_report = None
                verification_result = None
                test_result: TestRunResult | None = None
                has_dafny_spec = False

                if stage_enabled(PipelineStage.CHECKER, max_stage):
                    # DafnyArchitect: translate Actor source code → Dafny spec
                    yield self._event(run_id, StreamEventType.AGENT_START, "dafny_architect")
                    self._audit.log(run_id, "agent_start", agent="dafny_architect", data={"iteration": i})

                    prior_verification = feedback.verification_feedback if feedback else None
                    dafny_result = None
                    async for _item in self._dafny_architect.run_streaming(
                        source_code=code_candidate.source_code,
                        requirement=request.requirement_text,
                        language=request.target_language.value,
                        verification_feedback=prior_verification,
                    ):
                        if isinstance(_item, str):
                            yield self._event(run_id, StreamEventType.AGENT_TOKEN, "dafny_architect", {"token": _item})
                        else:
                            dafny_result = _item
                    code_candidate.dafny_spec = dafny_result.dafny_source
                    has_dafny_spec = bool(code_candidate.dafny_spec.strip())

                    yield self._event(
                        run_id, StreamEventType.AGENT_OUTPUT, "dafny_architect",
                        {
                            "dafny_spec": dafny_result.dafny_source,
                            "reasoning_trace": dafny_result.reasoning_trace,
                        },
                    )
                    self._audit.log(run_id, "agent_output", agent="dafny_architect",
                                    data=dafny_result.model_dump())

                    yield self._event(run_id, StreamEventType.AGENT_START, "checker")
                    yield self._event(run_id, StreamEventType.AGENT_START, "dafny_verifier")

                    # Run Dafny verifier in background (subprocess, no LLM tokens),
                    # stream Checker in the foreground so tokens reach the display.
                    dafny_task = asyncio.create_task(
                        self._dafny.verify(code_candidate.dafny_spec)
                    )

                    checker_report = None
                    async for _item in self._checker.run_streaming(
                        source_code=code_candidate.source_code,
                        language=request.target_language.value,
                        standard=request.safety_standard.value,
                    ):
                        if isinstance(_item, str):
                            yield self._event(run_id, StreamEventType.AGENT_TOKEN, "checker", {"token": _item})
                        else:
                            checker_report = _item

                    verification_result = await dafny_task

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

                    # --- TEST EXECUTION (if enabled and language is Python) ---
                    is_python = request.target_language.value == "Python"

                    if (
                        request.run_tests
                        and is_python
                        and checker_report.test_cases
                    ):
                        yield self._event(run_id, StreamEventType.AGENT_START, "test_runner")
                        test_result = await self._test_runner.run(
                            source_code=code_candidate.source_code,
                            test_cases=checker_report.test_cases,
                        )
                        iteration.test_result = test_result

                        yield self._event(
                            run_id, StreamEventType.TEST_RUN, "test_runner",
                            {
                                "executed": test_result.executed,
                                "total": test_result.total,
                                "passed": test_result.passed,
                                "failed": test_result.failed,
                                "errors": test_result.errors,
                                "test_results": [t.model_dump() for t in test_result.test_results],
                                "pytest_output": test_result.pytest_output,
                            },
                        )
                        self._audit.log(run_id, "test_run", data=test_result.model_dump())
                    elif not request.run_tests and checker_report.test_cases:
                        # Tests exist but not executed — store them
                        test_result = TestRunResult(
                            executed=False,
                            total=len(checker_report.test_cases),
                            pytest_output="Test execution disabled (run_tests=false).",
                        )
                        iteration.test_result = test_result

                # --- POLICY — requires stage >= POLICY ---
                policy_verdict = None

                if stage_enabled(PipelineStage.POLICY, max_stage):
                    yield self._event(run_id, StreamEventType.AGENT_START, "policy")

                    policy_context = self._retriever.retrieve(
                        query=f"{request.requirement_text} {request.safety_standard.value}",
                        standard=request.safety_standard.value,
                    )

                    policy_verdict = None
                    async for _item in self._policy.run_streaming(
                        source_code=code_candidate.source_code,
                        dafny_spec=code_candidate.dafny_spec,
                        standard=request.safety_standard.value,
                        checker_report=checker_report,
                        verification_result=verification_result,
                        policy_context=policy_context,
                    ):
                        if isinstance(_item, str):
                            yield self._event(run_id, StreamEventType.AGENT_TOKEN, "policy", {"token": _item})
                        else:
                            policy_verdict = _item
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
                # Dafny is best-effort: tracked and fed back, but does NOT gate convergence.
                # Checker verdict + test results (and Policy at full stage) determine pass/fail.
                dafny_ok = verification_result is not None and verification_result.verified
                checker_ok = checker_report is not None and checker_report.verdict == CheckerVerdict.PASS
                tests_ok = (
                    test_result is None  # no tests = not a blocker
                    or not test_result.executed  # tests disabled = not a blocker
                    or test_result.failed == 0  # all tests passed
                )

                if max_stage == PipelineStage.ACTOR:
                    all_pass = True
                elif max_stage == PipelineStage.CHECKER:
                    # Checker verdict + test results gate convergence. Dafny is soft.
                    all_pass = checker_ok and tests_ok

                    feedback = compose_feedback(
                        iteration=i,
                        checker_report=checker_report,
                        verification_result=verification_result,
                        policy_verdict=None,
                        test_result=test_result,
                    )
                    iteration.feedback = feedback
                else:
                    # Full pipeline — Checker + Tests + Policy must pass. Dafny is soft.
                    policy_ok = policy_verdict is not None and policy_verdict.compliant
                    all_pass = checker_ok and tests_ok and policy_ok

                    feedback = compose_feedback(
                        iteration=i,
                        checker_report=checker_report,
                        verification_result=verification_result,
                        policy_verdict=policy_verdict,
                        test_result=test_result,
                    )
                    iteration.feedback = feedback

                # Best-so-far tracking — detect regressions across iterations
                score = (
                    (1.0 if checker_ok else 0.0)
                    + (1.0 if tests_ok else 0.0)
                    + (0.5 if dafny_ok else 0.0)
                    + (0.25 if has_dafny_spec else 0.0)
                )
                if max_stage == PipelineStage.POLICY:
                    policy_ok = policy_verdict is not None and policy_verdict.compliant
                    score += 1.0 if policy_ok else 0.0

                if score > best_score:
                    best_score = score
                    best_iteration = i
                elif i > 1 and score < best_score and feedback is not None:
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
