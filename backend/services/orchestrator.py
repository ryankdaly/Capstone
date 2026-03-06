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
    VerificationResult,
)
from backend.api.schemas.pipeline import (
    IterationRecord,
    PipelineRequest,
    PipelineState,
    PipelineStatus,
    StreamEvent,
    StreamEventType,
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

    async def run(
        self, request: PipelineRequest
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the pipeline, yielding SSE events as it progresses."""
        state = PipelineState(
            request=request,
            status=PipelineStatus.RUNNING,
        )
        run_id = state.run_id

        self._audit.log(run_id, "pipeline_start", data=request.model_dump())

        feedback: FeedbackMessage | None = None

        try:
            for i in range(1, request.max_iterations + 1):
                state.current_iteration = i

                iteration = IterationRecord(iteration=i)

                # --- ACTOR ---
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
                    {"language": code_candidate.language, "has_dafny": bool(code_candidate.dafny_spec)},
                )
                self._audit.log(
                    run_id, "agent_output", agent="actor",
                    data=code_candidate.model_dump(),
                )

                # --- CHECKER + DAFNY (parallel) ---
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
                    {"verdict": checker_report.verdict.value, "issues": len(checker_report.issues)},
                )
                yield self._event(
                    run_id, StreamEventType.AGENT_OUTPUT, "dafny_verifier",
                    {"verified": verification_result.verified},
                )

                self._audit.log(run_id, "agent_output", agent="checker", data=checker_report.model_dump())
                self._audit.log(run_id, "verification_result", data=verification_result.model_dump())

                # --- POLICY ---
                yield self._event(run_id, StreamEventType.AGENT_START, "policy")

                # Retrieve relevant standard clauses via RAG
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
                    {"compliant": policy_verdict.compliant, "risk_level": policy_verdict.risk_level.value},
                )
                self._audit.log(run_id, "agent_output", agent="policy", data=policy_verdict.model_dump())

                # --- CONVERGENCE CHECK ---
                all_pass = (
                    checker_report.verdict == CheckerVerdict.PASS
                    and verification_result.verified
                    and policy_verdict.compliant
                )

                # Compose feedback for next iteration (or final summary)
                feedback = compose_feedback(
                    iteration=i,
                    checker_report=checker_report,
                    verification_result=verification_result,
                    policy_verdict=policy_verdict,
                )
                iteration.feedback = feedback
                iteration.completed_at = datetime.now(timezone.utc)
                state.iterations.append(iteration)

                yield self._event(
                    run_id, StreamEventType.ITERATION_COMPLETE, data={
                        "iteration": i, "all_pass": all_pass,
                        "summary": feedback.priority_summary,
                    },
                )

                if all_pass:
                    state.status = PipelineStatus.AWAITING_APPROVAL
                    state.final_code = code_candidate.source_code
                    state.final_proof = code_candidate.dafny_spec
                    break
            else:
                # Max iterations exhausted without convergence
                state.status = PipelineStatus.FAILED
                state.error = f"Failed to converge after {request.max_iterations} iterations"

        except Exception as e:
            logger.exception("Pipeline error in run %s", run_id)
            state.status = PipelineStatus.FAILED
            state.error = str(e)
            yield self._event(
                run_id, StreamEventType.AGENT_ERROR, data={"error": str(e)},
            )

        state.completed_at = datetime.now(timezone.utc)
        self._audit.log(run_id, "pipeline_complete", data={
            "status": state.status.value,
            "iterations": len(state.iterations),
        })

        yield self._event(
            run_id, StreamEventType.PIPELINE_COMPLETE, data={
                "status": state.status.value,
                "iterations": len(state.iterations),
                "run_id": str(run_id),
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
