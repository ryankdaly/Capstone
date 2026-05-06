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
from backend.services.agents.base import write_run_header
from backend.services.agents.checker import CheckerAgent
from backend.services.agents.dafny_architect import DafnyArchitectAgent
from backend.services.agents.policy import PolicyAgent
from backend.config import resolve_data_path, settings
from backend.services.audit.logger import AuditLogger
from backend.services.feedback import compose_feedback
from backend.services.llm.client import LLMClient
from backend.services.rag.retriever import StandardsRetriever
from backend.services.testing.runner import TestRunner
from backend.services.verification.contract_extractor import DafnyContracts, extract_contracts
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
        self._retriever = retriever or StandardsRetriever(
            persist_dir=str(resolve_data_path(settings.policies.chromadb_dir, "chromadb")),
            auto_ingest_path=str(resolve_data_path(settings.policies.standards_dir, "standards")),
        )
        self._audit = audit_logger or AuditLogger()
        self._test_runner = test_runner or TestRunner()

        # Agents
        self._actor = ActorAgent(self._llm)
        self._dafny_architect = DafnyArchitectAgent(self._llm)
        self._checker = CheckerAgent(self._llm)
        self._policy = PolicyAgent(self._llm)

        # Exposed after run() completes — the CLI reads this.
        self.last_state: PipelineState | None = None

        # Dafny cancellation — set by cancel_dafny() to kill a zombie process.
        self._dafny_cancel: asyncio.Event = asyncio.Event()
        # True while Dafny verifier subprocess is running (read by display layer).
        self.dafny_running: bool = False

    def cancel_dafny(self) -> None:
        """Signal the in-flight Dafny subprocess to terminate immediately."""
        self._dafny_cancel.set()

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

        # --- Debug session header in agent.log ---
        try:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            header = (
                "\n"
                "################################################################################\n"
                f"# NEW PIPELINE RUN\n"
                f"# run_id  : {run_id}\n"
                f"# time    : {ts}\n"
                f"# stage   : {max_stage.value}\n"
                f"# standard: {request.safety_standard.value}\n"
                f"# language: {request.target_language.value}\n"
                f"# prompt  : {request.requirement_text[:200]}\n"
                "################################################################################\n"
            )
            write_run_header(header)
        except Exception:
            pass

        self._audit.log(run_id, "pipeline_start", data={
            **request.model_dump(),
            "stage": max_stage.value,
        })

        # Seed feedback from a prior failed run if the caller passed retry context.
        feedback: FeedbackMessage | None = (
            FeedbackMessage(
                iteration=0,
                priority_summary=(
                    f"[RETRY — prior run failed]\n{request.retry_context}"
                ),
            )
            if request.retry_context
            else None
        )
        best_score: float = -1.0
        best_iteration: int = 0
        # Stash verified Dafny spec to skip architect on next iteration.
        _cached_dafny_spec: str | None = None

        # Actor-only has no feedback loop. Checker+ can loop on Dafny/Checker feedback.
        effective_max_iterations = (
            1 if max_stage == PipelineStage.ACTOR
            else request.max_iterations
        )

        try:
            for i in range(1, effective_max_iterations + 1):
                state.current_iteration = i

                iteration = IterationRecord(iteration=i)

                # --- ACTOR (always runs) — up to 3 attempts ---
                _ACTOR_MAX_RETRIES = 3
                code_candidate = None
                _actor_skipped = False
                _actor_retry_hint: str | None = None

                for _attempt in range(1, _ACTOR_MAX_RETRIES + 1):
                    yield self._event(run_id, StreamEventType.AGENT_START, "actor")
                    self._audit.log(run_id, "agent_start", agent="actor",
                                    data={"iteration": i, "attempt": _attempt})
                    try:
                        async for _item in self._actor.run_streaming(
                            requirement=request.requirement_text,
                            language=request.target_language.value,
                            standard=request.safety_standard.value,
                            feedback=feedback,
                            retry_hint=_actor_retry_hint,
                        ):
                            if isinstance(_item, str):
                                yield self._event(run_id, StreamEventType.AGENT_TOKEN, "actor", {"token": _item})
                            else:
                                code_candidate = _item
                        break  # success
                    except Exception as _actor_exc:
                        _actor_retry_hint = str(_actor_exc)
                        logger.warning(
                            "Actor attempt %d/%d failed: %s",
                            _attempt, _ACTOR_MAX_RETRIES, _actor_exc,
                        )
                        if _attempt == _ACTOR_MAX_RETRIES:
                            _actor_skipped = True
                            yield self._event(
                                run_id, StreamEventType.AGENT_ERROR, "actor",
                                {
                                    "error": str(_actor_exc),
                                    "skipped": True,
                                    "message": "Actor failed after 3 attempts — skipping iteration.",
                                },
                            )
                            self._audit.log(run_id, "agent_skipped", agent="actor",
                                            data={"error": str(_actor_exc)})

                if _actor_skipped:
                    # Can't proceed without code — move to next pipeline iteration.
                    continue

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

                # ── PHASE 1: DAFNY (architect → verify, up to 3 full cycles) ──────────
                # Each cycle: DafnyArchitect generates a spec, DafnyVerifier checks it.
                # On verification failure the error is fed back to DafnyArchitect for
                # the next cycle.  On LLM failure we retry the architect without a
                # verification error attached.  Both phases complete before checker runs.
                checker_report = None
                verification_result = None
                test_result: TestRunResult | None = None
                has_dafny_spec = False

                if stage_enabled(PipelineStage.CHECKER, max_stage):
                    _DAFNY_MAX_CYCLES = 3
                    # Seed with any verification error from the previous outer iteration.
                    prior_dafny_verification = (
                        feedback.verification_feedback if feedback else None
                    )
                    dafny_result = None
                    dafny_arch_skipped = False
                    _dafny_retry_hint: str | None = None
                    dafny_contracts: DafnyContracts | None = None

                    # ── Fast path: reuse verified spec from prior iteration ──
                    _used_cached_spec = False
                    if _cached_dafny_spec is not None:
                        logger.info(
                            "Iteration %d: trying cached Dafny spec (skip architect).", i,
                        )
                        yield self._event(run_id, StreamEventType.AGENT_START, "dafny_verifier")
                        self._dafny_cancel.clear()
                        self.dafny_running = True
                        _cache_vr = await self._dafny.verify(
                            _cached_dafny_spec,
                            cancel_event=self._dafny_cancel,
                        )
                        self.dafny_running = False
                        yield self._event(
                            run_id, StreamEventType.AGENT_OUTPUT, "dafny_verifier",
                            {
                                "verified": _cache_vr.verified,
                                "solver_output": _cache_vr.solver_output,
                                "failing_assertions": _cache_vr.failing_assertions,
                                "execution_time_seconds": _cache_vr.execution_time_seconds,
                                "cycle": 0,
                                "cached": True,
                            },
                        )
                        if _cache_vr.verified:
                            _used_cached_spec = True
                            code_candidate.dafny_spec = _cached_dafny_spec
                            has_dafny_spec = True
                            verification_result = _cache_vr
                            dafny_contracts = extract_contracts(_cached_dafny_spec)
                            logger.info(
                                "Cached Dafny spec still verifies — skipping architect.",
                            )
                        else:
                            logger.info(
                                "Cached Dafny spec failed verification — regenerating.",
                            )
                            _cached_dafny_spec = None

                    _dafny_cycles = range(1, _DAFNY_MAX_CYCLES + 1) if not _used_cached_spec else range(0)
                    for _cycle in _dafny_cycles:
                        # ── Dafny Architect LLM ──────────────────────────────────────
                        yield self._event(run_id, StreamEventType.AGENT_START, "dafny_architect")
                        self._audit.log(run_id, "agent_start", agent="dafny_architect",
                                        data={"iteration": i, "cycle": _cycle})
                        try:
                            async for _item in self._dafny_architect.run_streaming(
                                source_code=code_candidate.source_code,
                                requirement=request.requirement_text,
                                language=request.target_language.value,
                                verification_feedback=prior_dafny_verification,
                                retry_hint=_dafny_retry_hint,
                            ):
                                if isinstance(_item, str):
                                    yield self._event(run_id, StreamEventType.AGENT_TOKEN,
                                                      "dafny_architect", {"token": _item})
                                else:
                                    dafny_result = _item
                        except Exception as _arch_exc:
                            _dafny_retry_hint = str(_arch_exc)
                            logger.warning(
                                "DafnyArchitect cycle %d/%d LLM failure: %s",
                                _cycle, _DAFNY_MAX_CYCLES, _arch_exc,
                            )
                            if _cycle == _DAFNY_MAX_CYCLES and dafny_result is None:
                                # Never got a valid spec — skip Dafny entirely.
                                dafny_arch_skipped = True
                                yield self._event(
                                    run_id, StreamEventType.AGENT_ERROR, "dafny_architect",
                                    {
                                        "error": str(_arch_exc),
                                        "skipped": True,
                                        "message": "Dafny spec generation failed after "
                                                   f"{_DAFNY_MAX_CYCLES} cycles — skipping.",
                                    },
                                )
                                self._audit.log(run_id, "agent_skipped", agent="dafny_architect",
                                                data={"error": str(_arch_exc)})
                            # LLM failure: carry the hint; don't touch prior_dafny_verification
                            continue

                        # Architect succeeded — publish output immediately.
                        code_candidate.dafny_spec = dafny_result.dafny_source
                        has_dafny_spec = bool(code_candidate.dafny_spec.strip())
                        yield self._event(
                            run_id, StreamEventType.AGENT_OUTPUT, "dafny_architect",
                            {
                                "dafny_spec": dafny_result.dafny_source,
                                "reasoning_trace": dafny_result.reasoning_trace,
                                "cycle": _cycle,
                            },
                        )
                        self._audit.log(run_id, "agent_output", agent="dafny_architect",
                                        data={**dafny_result.model_dump(), "cycle": _cycle})

                        # ── Dafny Verifier ────────────────────────────────────────────
                        yield self._event(run_id, StreamEventType.AGENT_START, "dafny_verifier")
                        self._dafny_cancel.clear()
                        self.dafny_running = True
                        _cycle_verification = await self._dafny.verify(
                            code_candidate.dafny_spec,
                            cancel_event=self._dafny_cancel,
                        )
                        self.dafny_running = False
                        verification_result = _cycle_verification

                        yield self._event(
                            run_id, StreamEventType.AGENT_OUTPUT, "dafny_verifier",
                            {
                                "verified": _cycle_verification.verified,
                                "solver_output": _cycle_verification.solver_output,
                                "failing_assertions": _cycle_verification.failing_assertions,
                                "execution_time_seconds": _cycle_verification.execution_time_seconds,
                                "cycle": _cycle,
                            },
                        )
                        self._audit.log(run_id, "verification_result",
                                        data={**_cycle_verification.model_dump(), "cycle": _cycle})

                        if _cycle_verification.verified or not has_dafny_spec:
                            if _cycle_verification.verified:
                                dafny_contracts = extract_contracts(code_candidate.dafny_spec)
                                _cached_dafny_spec = code_candidate.dafny_spec
                                logger.info(
                                    "Extracted %d requires + %d ensures from Dafny spec for '%s'",
                                    len(dafny_contracts.requires),
                                    len(dafny_contracts.ensures),
                                    dafny_contracts.method_name,
                                )
                            break  # Dafny phase complete ✓

                        # Verification failed — feed the errors back for next cycle.
                        logger.info(
                            "Dafny verification failed on cycle %d/%d; feeding errors "
                            "back to DafnyArchitect for next cycle.",
                            _cycle, _DAFNY_MAX_CYCLES,
                        )
                        prior_dafny_verification = _cycle_verification
                        # (loop continues to next cycle)

                    if dafny_arch_skipped:
                        code_candidate.dafny_spec = ""
                        has_dafny_spec = False

                    # When Dafny produced a spec but verification failed, pass the raw
                    # spec + solver output to the checker as a conceptual hint — not as
                    # verified contracts. The checker uses it to write sharper test cases
                    # without treating the unverified postconditions as ground truth.
                    dafny_unverified_spec: str | None = None
                    dafny_solver_output_hint: str | None = None
                    if has_dafny_spec and dafny_contracts is None:
                        dafny_unverified_spec = code_candidate.dafny_spec
                        if verification_result is not None:
                            dafny_solver_output_hint = verification_result.solver_output

                    # ── PHASE 2: CHECKER (generate → tests) ──────────────────────────
                    # Runs after Dafny phase is fully resolved.
                    # Outer loop: up to 2 extra retries when pytest cannot collect
                    # tests (syntax/indentation error in generated test file).
                    # Inner loop: up to 3 retries for LLM parse failures.
                    _CHECKER_MAX_RETRIES = 3
                    _TEST_FIX_MAX = 2  # max extra rounds to fix broken test syntax
                    checker_report = None
                    _checker_skipped = False
                    _checker_retry_hint: str | None = None
                    _test_fix_hint: str | None = None
                    is_python = request.target_language.value == "Python"

                    for _fix_round in range(1, _TEST_FIX_MAX + 2):  # 1 normal + 2 fix rounds
                        # ── Checker LLM (with parse-failure retries) ──────────────
                        for _attempt in range(1, _CHECKER_MAX_RETRIES + 1):
                            yield self._event(run_id, StreamEventType.AGENT_START, "checker")
                            self._audit.log(run_id, "agent_start", agent="checker",
                                            data={"iteration": i, "attempt": _attempt,
                                                  "fix_round": _fix_round})
                            try:
                                async for _item in self._checker.run_streaming(
                                    source_code=code_candidate.source_code,
                                    language=request.target_language.value,
                                    standard=request.safety_standard.value,
                                    retry_hint=_checker_retry_hint,
                                    dafny_contracts=dafny_contracts,
                                    dafny_unverified_spec=dafny_unverified_spec,
                                    dafny_solver_output=dafny_solver_output_hint,
                                    test_fix_hint=_test_fix_hint,
                                ):
                                    if isinstance(_item, str):
                                        yield self._event(run_id, StreamEventType.AGENT_TOKEN,
                                                          "checker", {"token": _item})
                                    else:
                                        checker_report = _item
                                break  # LLM parse success
                            except Exception as _checker_exc:
                                _checker_retry_hint = str(_checker_exc)
                                logger.warning(
                                    "Checker attempt %d/%d (fix_round %d) failed: %s",
                                    _attempt, _CHECKER_MAX_RETRIES, _fix_round, _checker_exc,
                                )
                                if _attempt == _CHECKER_MAX_RETRIES:
                                    _checker_skipped = True
                                    yield self._event(
                                        run_id, StreamEventType.AGENT_ERROR, "checker",
                                        {
                                            "error": str(_checker_exc),
                                            "skipped": True,
                                            "message": "Checker failed after 3 attempts — skipping.",
                                        },
                                    )
                                    self._audit.log(run_id, "agent_skipped", agent="checker",
                                                    data={"error": str(_checker_exc)})

                        if not _checker_skipped:
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
                            self._audit.log(run_id, "agent_output", agent="checker",
                                            data=checker_report.model_dump())

                        # ── Test runner ───────────────────────────────────────────
                        if (
                            request.run_tests
                            and is_python
                            and checker_report is not None
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

                            # Detect collection error — pytest couldn't parse the test file.
                            # Feed the error back to Checker to rewrite tests.
                            _is_collection_err = (
                                test_result.executed
                                and test_result.errors > 0
                                and test_result.total == 0
                                and test_result.passed == 0
                                and _fix_round <= _TEST_FIX_MAX
                                and not _checker_skipped
                            )
                            if _is_collection_err:
                                _test_fix_hint = test_result.pytest_output[:800]
                                _checker_retry_hint = None  # reset parse-failure hint
                                logger.warning(
                                    "Test collection error on fix_round %d/%d — "
                                    "asking Checker to rewrite tests.",
                                    _fix_round, _TEST_FIX_MAX,
                                )
                                continue  # outer loop: rerun checker with fix hint
                        elif not request.run_tests and checker_report is not None and checker_report.test_cases:
                            test_result = TestRunResult(
                                executed=False,
                                total=len(checker_report.test_cases),
                                pytest_output="Test execution disabled (run_tests=false).",
                            )
                            iteration.test_result = test_result

                        break  # tests ran (or not applicable) — exit fix loop

                    iteration.checker_report = checker_report
                    iteration.verification_result = verification_result

                # --- POLICY — requires stage >= POLICY ---
                policy_verdict = None

                if stage_enabled(PipelineStage.POLICY, max_stage):
                    yield self._event(run_id, StreamEventType.AGENT_START, "policy")

                    try:
                        policy_context = await self._retriever.retrieve(
                            query=f"{request.requirement_text} {request.safety_standard.value}",
                            standard=request.safety_standard.value,
                        )
                    except Exception as _rag_exc:
                        logger.warning(
                            "RAG retrieval failed (non-fatal) — proceeding without policy context: %s",
                            _rag_exc,
                        )
                        policy_context = ""

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
                # Use granular test pass ratio instead of binary pass/fail so
                # iteration with 8/9 tests beats one with 6/10.
                if test_result and test_result.executed and test_result.total > 0:
                    test_score = test_result.passed / test_result.total
                else:
                    test_score = 1.0 if tests_ok else 0.0

                score = (
                    (1.0 if checker_ok else 0.0)
                    + test_score
                    + (0.5 if dafny_ok else 0.0)
                    + (0.25 if has_dafny_spec else 0.0)
                )
                if max_stage == PipelineStage.POLICY:
                    policy_ok = policy_verdict is not None and policy_verdict.compliant
                    score += 1.0 if policy_ok else 0.0

                if score >= best_score:
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
                # Still expose best iteration's code so re-run commands work.
                if best_iteration > 0 and best_iteration <= len(state.iterations):
                    best_iter = state.iterations[best_iteration - 1]
                    if best_iter.code_candidate is not None:
                        state.final_code = best_iter.code_candidate.source_code
                        state.final_proof = best_iter.code_candidate.dafny_spec

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
