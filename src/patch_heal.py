"""AgentTune closed-loop self-healing extensions for FinCEN/FATF/OFAC compliance.

Fixes AgentTune v1.1.3 closed-loop gaps while preserving full compatibility:
1. Closed-loop submodule imports (.failure_classifier, .replay_validator, etc.)
2. KYCFailureDetector: inspects both AuditWriter JSONL and ReAct trajectories to
   surface loop_collapse, first-try wrong_tool, hallucinated_output, wrong_routing,
   and incomplete_reasoning without false-flagging distinct-arg or multi-party queries.
3. KYCFailureClassifier: locks deterministic observable invariants pre/post LLM.
4. ComplianceReplayValidator: validates schemas + ISO-2/DOB + SQLite watchlists +
   FinCEN oracle rules in-process with HARD REJECT anti-poisoning.
5. KYCTrainingExampleGenerator: supports all 5 root causes, reconstructs real
   dossier + OpenAI tool schemas, keeps real student failures as rejected, blocks
   reward>=0 fallback poisoning, and exports ChatML message-list DPO datasets.
6. BehavioralDiversityMonitor bridge normalizing dict tool calls -> list[str].
"""

from __future__ import annotations

import asyncio
from collections import Counter
import inspect
import json
import os
from pathlib import Path
import re
from typing import Any, Callable
import uuid

from agenttune.agentic.events import EventLog
from agenttune.agentic.heal_loop import (
    SelfHealLoop,
    as_sync_classifier,
    as_sync_generator,
    build_dataset,
)
from agenttune.agentic.project import Project
from agenttune.decide.closed_loop.behavioral_diversity_monitor import (
    BehavioralDiversityMonitor,
)
from agenttune.decide.closed_loop.contracts import (
    AgenticEvalResult,
    BufferHealth,
    ClassifiedFailure,
    Failure,
    TrainingExample,
)
from agenttune.decide.closed_loop.deployment_gate import (
    DeploymentGate,
    GateDecision,
)
from agenttune.decide.closed_loop.failure_classifier import FailureClassifier
from agenttune.decide.closed_loop.failure_detector import FailureDetector
from agenttune.decide.closed_loop.replay_validator import ReplayValidator
from agenttune.decide.closed_loop.retraining_trigger import (
    RetrainingTrigger,
    TrainingBuffer,
    TriggerConfig,
)
from agenttune.decide.closed_loop.training_example_generator import (
    TrainingExampleGenerator,
    _parse_tool_call,
)
from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator
from auditkit.metrics.structured import JsonSchemaValid
from auditkit.sample import Sample
import litellm

from src.kyc_agent_runner import (
    _build_system_prompt,
    _extract_inline_tool_calls,
    _extract_json_object,
    _format_case_prompt,
    _parse_verdict_from_text,
    canonical_tool_call_key,
    format_messages_for_cohere,
)
from src.kyc_oracle import (
    compute_required_tools,
    evaluate_case_oracle,
    is_valid_dob,
    is_valid_iso2,
)
from src.kyc_tools import (
    DEFAULT_WATCHLIST_DB,
    build_kyc_tool_registry,
    execute_kyc_tool_call,
    get_openai_tool_schemas,
)

TERMINAL_VERDICT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["verdict", "reason"],
    "properties": {
        "verdict": {"type": "string", "enum": ["APPROVE", "REVIEW", "DENY"]},
        "reason": {"type": "string", "minLength": 1},
        "cited_tools": {"type": "array", "items": {"type": "string"}},
        "edd_required": {"type": "boolean"},
        "escalate_to_edd": {"type": "boolean"},
        "escalation_required": {"type": "boolean"},
        "reason_codes": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": True,
}

_LEAK_RE = re.compile(
    r"(expected(?:_verdict)?|oracle_verdict)\s*[:=]\s*[^\s,;)\]}]+",
    re.IGNORECASE,
)

ALL_ROOT_CAUSES: tuple[str, ...] = (
    "wrong_tool",
    "loop_collapse",
    "wrong_routing",
    "incomplete_reasoning",
    "hallucinated_output",
)


def detect_canonical_loop(tool_calls: list[dict[str, Any]], max_revisits: int = 3) -> dict[str, Any]:
    """Detect canonical identical-arg repeats (>=3) or alternating cycles A(x)->B(y)->A(x).

    Distinct-argument self-corrections (e.g. 'USA' -> 'US') and multi-party screening
    queries (Entity A -> Entity B) have distinct canonical keys and are NEVER flagged.
    """
    keys = [
        canonical_tool_call_key(str(tc.get("name") or ""), tc.get("arguments", {}))
        for tc in tool_calls
    ]
    counts = Counter(keys)
    max_repeat = max(counts.values()) if counts else 0

    alternating_cycle = False
    for idx in range(len(keys) - 2):
        if keys[idx] == keys[idx + 2] and keys[idx] != keys[idx + 1]:
            alternating_cycle = True
            break

    is_loop = max_repeat >= max_revisits or alternating_cycle
    return {
        "is_loop": is_loop,
        "max_repeat": max_repeat,
        "alternating_cycle": alternating_cycle,
        "repeat_counts": dict(counts),
    }


def normalize_trajectories_for_diversity(
    trajectories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert tool_calls lists of dicts into list[str] tool names for BehavioralDiversityMonitor."""
    normalized: list[dict[str, Any]] = []
    for traj in trajectories:
        raw_calls = traj.get("tool_calls", [])
        str_calls: list[str] = []
        for item in raw_calls:
            if isinstance(item, str):
                str_calls.append(item)
            elif isinstance(item, dict):
                fn = item.get("function") or {}
                str_calls.append(str(item.get("name") or fn.get("name") or "unknown"))
        normalized.append({**traj, "tool_calls": str_calls})
    return normalized


class KYCFailureDetector:
    """Detects compliance, tool-calling, loop, and hallucination failures from runs & audit logs."""

    def __init__(self, max_revisits: int = 3) -> None:
        self.max_revisits = max_revisits
        self.audit_detector = FailureDetector(max_revisits=max_revisits)

    def scan_audit_log(self, audit_log_path: str | Path) -> list[Failure]:
        """Delegate DECIDE AuditWriter JSONL scanning cleanly to AgentTune's FailureDetector."""
        return list(self.audit_detector.scan_audit_log(str(audit_log_path)))

    def detect_from_event_logs(self, event_logs: list[EventLog]) -> list[Failure]:
        """Run AgentTune Project.add_trajectory + Project.heal(max_revisits=3) over EventLogs."""
        if not event_logs:
            return []
        project = Project()
        for log in event_logs:
            project.add_trajectory(log)
        return list(project.heal(max_revisits=self.max_revisits))

    def detect_from_runs(self, run_results: list[dict[str, Any]]) -> list[Failure]:
        """Extract observable Failures from KYCAgentRunner results, including 1st-try tool errors."""
        failures: list[Failure] = []
        for run in run_results:
            case = run["case"]
            policy = run.get("tool_policy") or compute_required_tools(case)
            tool_calls = run.get("tool_calls", [])
            first_try_errors = run.get("first_try_tool_errors", [])
            loop_info = detect_canonical_loop(tool_calls, max_revisits=self.max_revisits)
            is_loop = bool(run.get("loop_collapse_detected") or loop_info["is_loop"])

            executed_valid_tools = {tc["name"] for tc in tool_calls if tc.get("success")}
            all_called_tools = [tc["name"] for tc in tool_calls]
            forbidden_called = [
                tc for tc in tool_calls if tc.get("name") in set(policy.get("forbidden_tools", []))
            ]
            abstain_violation = bool(policy.get("abstain")) and len(tool_calls) > 0
            required_tools = set(policy.get("required_tools", []))
            missing_required = sorted(required_tools - executed_valid_tools)

            verdict = run.get("verdict")
            oracle_verdict = run.get("oracle_verdict")
            dangling = run.get("dangling_tool_calls", [])
            has_unscreened_approve = (
                not policy.get("abstain")
                and verdict == "APPROVE"
                and ("screen_sanctions" not in executed_valid_tools or "screen_pep_adverse_media" not in executed_valid_tools)
            )
            verdict_mismatch = verdict != oracle_verdict

            if not (
                is_loop
                or first_try_errors
                or forbidden_called
                or abstain_violation
                or dangling
                or has_unscreened_approve
                or missing_required
                or verdict_mismatch
            ):
                continue

            # Determine failing assistant message as real rejected behavior
            failing_msg = self._select_failing_assistant_message(
                run=run,
                is_loop=is_loop,
                first_try_errors=first_try_errors,
                forbidden_called=forbidden_called,
            )

            error_notes: list[str] = []
            if is_loop:
                error_notes.append(f"loop_collapse(max_repeat={loop_info['max_repeat']})")
            if first_try_errors:
                error_notes.append(f"first_try_tool_errors={len(first_try_errors)}")
            if forbidden_called:
                error_notes.append(
                    f"forbidden_tools_called={[t['name'] for t in forbidden_called]}"
                )
            if abstain_violation:
                error_notes.append("abstain_violation_on_incomplete_cip")
            if has_unscreened_approve:
                error_notes.append("hallucinated_approve_without_watchlist_screening")
            if missing_required:
                error_notes.append(f"missing_required_tools={missing_required}")
            if verdict_mismatch:
                error_notes.append(f"compliance_disposition_mismatch(got={verdict})")

            failure_type = (
                "loop_collapse"
                if is_loop
                else ("tool_error" if (first_try_errors or forbidden_called or abstain_violation or dangling) else "policy_failure")
            )
            if first_try_errors:
                failing_step = int(first_try_errors[0].get("step") or 1)
                scoped_observations = [
                    obs for obs in run.get("tool_observations", []) if int(obs.get("step") or 999) < failing_step
                ]
            elif forbidden_called or abstain_violation:
                target_name = (forbidden_called[0].get("name") if forbidden_called else None) or (
                    tool_calls[0].get("name") if tool_calls else None
                )
                failing_step = next(
                    (int(t.get("step") or 1) for t in tool_calls if t.get("name") == target_name),
                    1,
                )
                scoped_observations = [
                    obs for obs in run.get("tool_observations", []) if int(obs.get("step") or 999) < failing_step
                ]
            else:
                failing_step = int(run.get("steps") or 1)
                scoped_observations = list(run.get("tool_observations", []))

            error_msg = "; ".join(error_notes)
            failure_id = f"fail_{run['case_id']}_{uuid.uuid4().hex[:6]}"
            failures.append(
                Failure(
                    trajectory_id=run["trajectory_id"],
                    failure_type=failure_type,
                    failed_stage_name="watchlist_screening_agent",
                    error_message=error_msg,
                    context={
                        "failure_id": failure_id,
                        "stage_type": "llm_call",
                        "error_details": error_msg,
                        "case": case,
                        "tool_policy": policy,
                        "messages": run.get("messages", []),
                        "tool_calls": tool_calls,
                        "tool_observations": scoped_observations,
                        "failing_step": failing_step,
                        "first_try_tool_errors": first_try_errors,
                        "forbidden_called": forbidden_called,
                        "abstain_violation": abstain_violation,
                        "missing_required_tools": missing_required,
                        "has_unscreened_approve": has_unscreened_approve,
                        "dangling_tool_calls": dangling,
                        "loop_info": loop_info,
                        "verdict": verdict,
                        "oracle_verdict": oracle_verdict,
                        "failing_assistant_message": failing_msg,
                        "stage_visit_counts": {"watchlist_screening_agent": run.get("steps", 1)},
                        "state_snapshot": {
                            "verdict": verdict,
                            "oracle_verdict": oracle_verdict,
                            "executed_tools": all_called_tools,
                        },
                    },
                )
            )
        return failures

    @staticmethod
    def _select_failing_assistant_message(
        run: dict[str, Any],
        is_loop: bool,
        first_try_errors: list[dict[str, Any]],
        forbidden_called: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if first_try_errors:
            msg = first_try_errors[0].get("assistant_message")
            if isinstance(msg, dict):
                return msg
        messages = [m for m in run.get("messages", []) if m.get("role") == "assistant"]
        if forbidden_called:
            target_forbidden = forbidden_called[0]["name"]
            for m in messages:
                for tc in m.get("tool_calls", []):
                    fn_name = (tc.get("function") or {}).get("name") or tc.get("name")
                    if fn_name == target_forbidden:
                        return m
                for tc in _extract_inline_tool_calls(str(m.get("content") or "")):
                    if tc.get("name") == target_forbidden:
                        return m
        if is_loop and messages:
            return messages[-1]
        if messages:
            return messages[-1]
        return {"role": "assistant", "content": json.dumps({"verdict": run.get("verdict"), "reason": "unknown"})}


class KYCFailureClassifier(FailureClassifier):
    """Root-cause classifier enforcing observable FinCEN/tool-calling invariants."""

    def __init__(
        self,
        model_name: str = "openai/Qwen/Qwen3-235B-A22B-Instruct-2507",
        api_base: str = "https://router.huggingface.co/v1",
        max_revisits: int = 3,
        llm_classifier: Callable[[Failure], str] | None = None,
    ) -> None:
        super().__init__(model_name=model_name, api_base=api_base)
        self.max_revisits = max_revisits
        self.llm_classifier = llm_classifier

    def classify_deterministic_root_cause(self, failure: Failure) -> tuple[str, str, float] | None:
        """Apply strict observable guards matching AgentTune contracts without LLM noise."""
        ctx = failure.context or {}
        tool_calls = ctx.get("tool_calls", [])
        loop_info = ctx.get("loop_info") or detect_canonical_loop(tool_calls, max_revisits=self.max_revisits)
        stage_visits = ctx.get("stage_visit_counts", {})
        max_stage_visits = max(stage_visits.values()) if stage_visits else 0

        # 1. Observable loop_collapse: ONLY identical canonical repeats >=3, A->B->A cycle, or revisit overflow
        err_text = failure.error_message or ctx.get("error_details", "") or ""
        if (
            loop_info.get("is_loop")
            or "LOOP_COLLAPSE" in err_text
            or max_stage_visits > 6
        ):
            return (
                "loop_collapse",
                f"Identical tool call repeated {loop_info.get('max_repeat', 3)}x or alternating/step-limit cycle.",
                0.98,
            )

        # 2. First-try wrong_tool: schema/ISO-2/DOB error even if turn 2 recovered, forbidden tool, or CIP abstain breach
        if (
            ctx.get("first_try_tool_errors")
            or ctx.get("forbidden_called")
            or ctx.get("abstain_violation")
            or ctx.get("dangling_tool_calls")
            or failure.failure_type == "tool_error"
        ):
            return (
                "wrong_tool",
                "Schema/ISO-2/DOB validation failure, forbidden tool invocation, or missing-CIP abstention breach.",
                0.97,
            )

        # 3. Hallucinated output: APPROVE/clean verdict without mandatory watchlist screening
        if ctx.get("has_unscreened_approve"):
            return (
                "hallucinated_output",
                "Issued APPROVE verdict without executing mandatory sanctions/PEP screening tools.",
                0.96,
            )

        # 4. Incomplete reasoning: skipped required conditional tools (e.g. SoW/export/wallet)
        if ctx.get("missing_required_tools"):
            return (
                "incomplete_reasoning",
                f"Skipped mandatory screening step(s): {ctx['missing_required_tools']}.",
                0.94,
            )

        # 5. Wrong routing: final APPROVE/REVIEW/DENY disposition mismatches deterministic FinCEN oracle
        if ctx.get("verdict") != ctx.get("oracle_verdict"):
            return (
                "wrong_routing",
                f"Final verdict {ctx.get('verdict')!r} mismatches FinCEN oracle {ctx.get('oracle_verdict')!r}.",
                0.95,
            )

        return None

    async def classify_batch(self, failures: list[Failure]) -> list[ClassifiedFailure]:
        """Classify a batch of failures using deterministic invariants first, LLM optional."""
        classified: list[ClassifiedFailure] = []
        for failure in failures:
            guard = self.classify_deterministic_root_cause(failure)
            if guard is not None:
                root_cause, reasoning, confidence = guard
            elif self.llm_classifier is not None:
                raw_rc = self.llm_classifier(failure)
                if inspect.isawaitable(raw_rc):
                    raw_rc = await raw_rc
                root_cause = raw_rc if raw_rc in self.ROOT_CAUSES else "incomplete_reasoning"
                reasoning = "Classified via custom LLM classifier."
                confidence = 0.85
            else:
                llm_res = await super()._classify_single(failure)
                root_cause = llm_res.root_cause
                reasoning = getattr(llm_res, "analysis", "") or "No analysis provided"
                confidence = llm_res.confidence

            # Post-classification observable lock so noisy LLMs cannot mislabel distinct-arg retries
            post_guard = self.classify_deterministic_root_cause(failure)
            if post_guard is not None and post_guard[0] != root_cause:
                root_cause, reasoning, confidence = post_guard

            classified.append(
                ClassifiedFailure(
                    failure=failure,
                    root_cause=root_cause,
                    confidence=confidence,
                    analysis=reasoning,
                )
            )
        return classified


class ComplianceReplayValidator(ReplayValidator):
    """Validates candidate DPO fixes against JSONSchema, SQLite watchlists, and FinCEN oracle."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        timeout: int = 30,
    ) -> None:
        super().__init__()
        self.timeout = timeout
        self.db_path = Path(db_path or DEFAULT_WATCHLIST_DB)
        self.registry = build_kyc_tool_registry(self.db_path, include_builtins=False)
        self._context_by_trajectory: dict[str, dict[str, Any]] = {}
        self.terminal_schema_validator = JsonSchemaValid(
            schema=TERMINAL_VERDICT_JSON_SCHEMA,
            backend="auto",
            strip_fences=True,
        )

    def register_context(self, trajectory_id: str, meta: dict[str, Any]) -> None:
        """Attach compliance validation metadata keyed cleanly by trajectory_id."""
        self._context_by_trajectory[trajectory_id] = meta

    @staticmethod
    def _coerce_completion_text(completion: Any) -> str:
        """Accept raw string OR OpenAI assistant message list [{'role':'assistant','content':...}]."""
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list) and completion:
            last = completion[-1]
            if isinstance(last, dict):
                return ComplianceReplayValidator._coerce_completion_text(last)
            return str(last)
        if isinstance(completion, dict):
            tool_calls = completion.get("tool_calls") or []
            if tool_calls:
                calls_payload = []
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    calls_payload.append({"tool_name": fn.get("name"), "parameters": args})
                return json.dumps(calls_payload, ensure_ascii=False)
            return str(completion.get("content") or "")
        return "" if completion is None else str(completion)

    def _parse_candidate_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse Cohere prompt-mode JSON arrays, action tags, OpenAI tool_calls, or single dicts."""
        if not isinstance(text, str) or not text.strip():
            return []
        inline_calls = _extract_inline_tool_calls(text)
        if inline_calls:
            return inline_calls
        try:
            loaded = json.loads(text.strip())
            if isinstance(loaded, dict) and "tool_calls" in loaded and isinstance(loaded["tool_calls"], list):
                parsed_calls = []
                for tc in loaded["tool_calls"]:
                    fn = tc.get("function") or {}
                    name = fn.get("name") or tc.get("name")
                    args = fn.get("arguments", tc.get("arguments", {}))
                    if isinstance(args, str):
                        args = json.loads(args)
                    if name and isinstance(args, dict):
                        parsed_calls.append({"name": name, "arguments": args})
                if parsed_calls:
                    return parsed_calls
            if isinstance(loaded, list):
                parsed_list = []
                for item in loaded:
                    if isinstance(item, dict):
                        name = item.get("tool_name") or item.get("name") or (item.get("function") or {}).get("name")
                        args = item.get("parameters", item.get("arguments", (item.get("function") or {}).get("arguments", {})))
                        if isinstance(args, str):
                            args = json.loads(args)
                        if name and isinstance(args, dict):
                            parsed_list.append({"name": str(name), "arguments": args})
                if parsed_list:
                    return parsed_list
            if isinstance(loaded, dict) and ("name" in loaded or "tool_name" in loaded) and (
                "arguments" in loaded or "parameters" in loaded
            ):
                args = loaded.get("parameters", loaded.get("arguments", {}))
                if isinstance(args, str):
                    args = json.loads(args)
                name = loaded.get("tool_name") or loaded.get("name")
                if name and isinstance(args, dict):
                    return [{"name": str(name), "arguments": args}]
        except (json.JSONDecodeError, TypeError):
            pass

        single = _parse_tool_call(text)
        if single is not None:
            return [single]
        return []

    async def validate_example_with_trace(
        self,
        example: TrainingExample,
        completion_text: str | list[dict[str, Any]] | dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Deterministically validate tool calls OR FinCEN compliance dispositions."""
        trace: list[str] = []
        meta = (
            getattr(example, "_kyc_meta", None)
            or self._context_by_trajectory.get(example.trajectory_id)
            or {}
        )
        root_cause = meta.get("root_cause") or getattr(example, "root_cause", "wrong_tool")
        case = meta.get("case") or {}
        policy = meta.get("tool_policy") or compute_required_tools(case)
        prior_observations = meta.get("tool_observations") or []

        raw_text = self._coerce_completion_text(completion_text)
        if not raw_text.strip():
            return False, ["EMPTY_COMPLETION"]

        parsed_calls = self._parse_candidate_tool_calls(raw_text)
        verdict, _ = _parse_verdict_from_text(raw_text)

        # Reject ambiguous candidate mixing a terminal verdict with a tool call in the same completion
        if parsed_calls and verdict is not None:
            trace.append("MIXED_TOOL_AND_VERDICT: candidate cannot combine tool calls and terminal verdict")
            return False, trace

        # Case 1: Candidate emits tool call(s)
        if parsed_calls:
            if policy.get("abstain"):
                trace.append("CIP_ABSTENTION_VIOLATION: cannot call screening tools when CIP incomplete")
                return False, trace

            forbidden = set(policy.get("forbidden_tools", []))
            failing_calls = meta.get("loop_info", {}).get("repeat_counts", {})
            prior_success_keys = {
                canonical_tool_call_key(str(obs.get("tool") or obs.get("name") or ""), obs.get("arguments", {}))
                for obs in prior_observations
                if obs.get("success", True)
            }
            seen_in_turn: set[str] = set()

            for call in parsed_calls:
                t_name = str(call["name"])
                t_args = call["arguments"] if isinstance(call["arguments"], dict) else {}
                if t_name in forbidden:
                    trace.append(f"FORBIDDEN_TOOL_CALLED: {t_name}")
                    return False, trace

                # Strict ISO-2 & DOB format checks prior to registry execution
                for iso_key in ("country_iso2", "offshore_jurisdiction"):
                    iso_val = t_args.get(iso_key)
                    if isinstance(iso_val, str) and iso_val.strip() and not is_valid_iso2(iso_val):
                        trace.append(f"INVALID_ISO2_ARG: {iso_key}={iso_val!r}")
                        return False, trace
                for dob_key in ("date_of_birth", "dob"):
                    dob_val = t_args.get(dob_key)
                    if isinstance(dob_val, str) and dob_val.strip() and not is_valid_dob(dob_val):
                        trace.append(f"INVALID_DOB_ARG: {dob_key}={dob_val!r}")
                        return False, trace

                c_key = canonical_tool_call_key(t_name, t_args)
                if c_key in seen_in_turn:
                    trace.append(f"REDUNDANT_INTRA_TURN_CALL: {c_key}")
                    return False, trace
                seen_in_turn.add(c_key)

                if root_cause == "loop_collapse" and (
                    failing_calls.get(c_key, 0) >= 3 or c_key in prior_success_keys
                ):
                    trace.append(f"REPEATS_LOOPING_OR_ANSWERED_CALL: {c_key}")
                    return False, trace

                exec_res = execute_kyc_tool_call(self.registry, t_name, t_args)
                trace.append(
                    f"EXEC({t_name}) -> success={exec_res['success']} err={exec_res.get('error')}"
                )
                if not exec_res["success"]:
                    return False, trace

            return True, trace

        # Case 2: Candidate emits terminal JSON verdict / escalation
        schema_score = self.terminal_schema_validator.score(
            Sample(input="", id=str(meta.get("example_id") or example.trajectory_id)),
            raw_text,
        )
        if schema_score.value < 1.0 or verdict is None:
            trace.append(
                f"INVALID_TERMINAL_JSON_OR_VERDICT({root_cause}): {schema_score.reason or 'unparseable'}"
            )
            return False, trace

        # Block unscreened terminal verdicts (APPROVE, REVIEW, OR DENY) on CIP-complete cases with missing required tools
        required_tools = set(policy.get("required_tools", []))
        ran_tools = {
            obs.get("tool") or obs.get("name")
            for obs in prior_observations
            if obs.get("success", True)
        }
        if not policy.get("abstain") and not required_tools.issubset(ran_tools):
            missing = sorted(required_tools - ran_tools)
            trace.append(f"UNSCREENED_TERMINAL_VERDICT({verdict}): missing {missing}")
            return False, trace

        if policy.get("abstain") and verdict != "REVIEW":
            trace.append(f"INCOMPLETE_CIP_MUST_REVIEW: got={verdict}")
            return False, trace

        parsed_obj = _extract_json_object(raw_text) or {}
        if case:
            oracle_out = evaluate_case_oracle(case, tool_observations=prior_observations)
            expected = oracle_out["verdict"]
            expected_edd = bool(oracle_out.get("edd_required", False))
            expected_escalation = expected != "APPROVE"
        else:
            expected = meta.get("expected_verdict")
            expected_edd = None
            expected_escalation = None

        if expected and verdict != expected:
            trace.append(f"VERDICT_MISMATCH: got={verdict}, expected={expected}")
            return False, trace

        if expected_edd is not None:
            for edd_key in ("edd_required", "escalate_to_edd"):
                if edd_key in parsed_obj and bool(parsed_obj[edd_key]) != expected_edd:
                    trace.append(f"EDD_FLAG_MISMATCH: {edd_key}={parsed_obj[edd_key]}, expected={expected_edd}")
                    return False, trace
        if expected_escalation is not None and "escalation_required" in parsed_obj:
            if bool(parsed_obj["escalation_required"]) != expected_escalation:
                trace.append(
                    f"ESCALATION_FLAG_MISMATCH: got={parsed_obj['escalation_required']}, expected={expected_escalation}"
                )
                return False, trace

        trace.append(f"ORACLE_VERDICT_VERIFIED: {verdict}")
        return True, trace


class KYCTrainingExampleGenerator(TrainingExampleGenerator):
    """Generates verified (prompt, chosen, rejected) DPO triples across all 5 root causes."""

    def __init__(
        self,
        validator: ComplianceReplayValidator,
        model_name: str = "openai/Qwen/Qwen3-235B-A22B-Instruct-2507",
        api_base: str = "https://router.huggingface.co/v1",
        num_candidates: int = 3,
        tool_schemas: list[dict[str, Any]] | None = None,
        synthesizer: Callable[[list[dict[str, str]], ClassifiedFailure], list[str]] | None = None,
        tool_mode: str = "prompt",
    ) -> None:
        self.openai_tool_schemas: list[dict[str, Any]] = tool_schemas or get_openai_tool_schemas(validator.db_path)
        param_schemas: dict[str, Any] = {
            s["function"]["name"]: s["function"]["parameters"]
            for s in self.openai_tool_schemas
            if isinstance(s, dict) and isinstance(s.get("function"), dict)
        }
        super().__init__(
            validator=validator,
            model_name=model_name,
            api_base=api_base,
            tool_schemas=param_schemas,
        )
        self.num_candidates = num_candidates
        self.synthesizer = synthesizer
        self.tool_mode = tool_mode
        self.traj_evaluator = TrajectoryEvaluator(
            model_name="none",
            api_base=None,
            tool_schemas=param_schemas,
        )

    def _build_prompt_history(self, failure: ClassifiedFailure) -> list[dict[str, str]]:
        """Reconstruct leak-free FinCEN prompt history (dossier, CIP rules, verified prior tool outputs)."""
        ctx = failure.failure.context or {}
        case = ctx.get("case") or {}
        sys_prompt = _build_system_prompt(self.openai_tool_schemas, tool_mode=self.tool_mode)

        if case:
            case_text = _format_case_prompt(case)
        else:
            safe_ctx = {
                k: v
                for k, v in ctx.items()
                if k not in {"oracle_verdict", "expected_verdict", "oracle_details"}
            }
            case_text = f"Context: {json.dumps(safe_ctx, default=str)}"

        raw_obs = ctx.get("tool_observations") or []
        first_err_ids = {o.get("id") for o in (ctx.get("first_try_tool_errors") or []) if o.get("id")}
        if failure.root_cause == "wrong_tool" and first_err_ids:
            prior_obs = [
                {k: v for k, v in obs.items() if k != "id"}
                for obs in raw_obs
                if obs.get("id") in first_err_ids or not obs.get("success", True)
            ]
        else:
            prior_obs = [
                {"tool": obs.get("tool"), "arguments": obs.get("arguments"), "result": obs.get("result")}
                for obs in raw_obs
                if obs.get("success", True)
            ]

        obs_summary = json.dumps(prior_obs, ensure_ascii=False) if prior_obs else "[]"

        user_prompt = (
            f"{case_text}\n\n"
            f"Prior tool observations: {obs_summary}\n"
            "Respond ONLY with the next valid tool call JSON array OR final compliance decision JSON."
        )
        raw_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.tool_mode == "prompt":
            return format_messages_for_cohere(raw_messages)
        return raw_messages

    @staticmethod
    def _format_assistant_message_text(msg: Any) -> str:
        if isinstance(msg, str):
            return msg
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                calls_payload = []
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    calls_payload.append({"tool_name": fn.get("name"), "parameters": args})
                return json.dumps(calls_payload, ensure_ascii=False)
            return str(msg.get("content") or json.dumps(msg))
        return str(msg)

    def _generate_sync_or_async_candidates(
        self,
        prompt_history: list[dict[str, str]],
        failure: ClassifiedFailure,
    ) -> list[str]:
        if self.synthesizer is not None:
            raw = self.synthesizer(prompt_history, failure)
            return list(raw) if isinstance(raw, list) else [str(raw)]
        return []

    async def _synthesize_candidates(
        self,
        prompt_history: list[dict[str, str]],
        failure: ClassifiedFailure,
    ) -> list[str]:
        """Synthesize corrective candidate completions via HuggingFace Router / litellm."""
        del failure
        model_id = self.model_name
        if self.api_base and "/" in model_id and not model_id.startswith("openai/"):
            model_id = f"openai/{model_id}"
        api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
        prepared = format_messages_for_cohere(prompt_history) if self.tool_mode == "prompt" else prompt_history

        async def _one() -> str | None:
            try:
                kwargs: dict[str, Any] = {
                    "model": model_id,
                    "messages": prepared,
                    "temperature": 0.6,
                    "max_tokens": 768,
                }
                if self.api_base:
                    kwargs["api_base"] = self.api_base
                if api_key:
                    kwargs["api_key"] = api_key
                resp = await litellm.acompletion(**kwargs)
                content = getattr(resp.choices[0].message, "content", None)
                return str(content).strip() if content else None
            except Exception:
                return None

        results = await asyncio.gather(*[_one() for _ in range(max(1, self.num_candidates))])
        return [r for r in results if r]

    async def _process_single(self, failure: ClassifiedFailure) -> TrainingExample | None:
        """Validate candidates with HARD REJECT anti-poisoning across all 5 root causes."""
        if failure.root_cause not in ALL_ROOT_CAUSES:
            return None

        ctx = failure.failure.context or {}
        case = ctx.get("case") or {}
        policy = ctx.get("tool_policy") or compute_required_tools(case)
        prompt_history = self._build_prompt_history(failure)

        if self.synthesizer is not None:
            candidates = self._generate_sync_or_async_candidates(prompt_history, failure)
        else:
            candidates = await self._synthesize_candidates(prompt_history, failure)

        if not candidates:
            return None

        failing_raw = ctx.get("failing_assistant_message")
        rejected_text = self._format_assistant_message_text(failing_raw).strip()
        if not rejected_text:
            rejected_text = json.dumps({"verdict": ctx.get("verdict") or "UNKNOWN", "reason": "failing_trajectory"})
        example_id = f"ex_{ctx.get('failure_id') or failure.failure.trajectory_id}"
        meta: dict[str, Any] = {
            "example_id": example_id,
            "root_cause": failure.root_cause,
            "case": case,
            "tool_policy": policy,
            "tool_observations": ctx.get("tool_observations", []),
            "loop_info": ctx.get("loop_info", {}),
            "expected_verdict": ctx.get("oracle_verdict"),
        }

        temp_example = TrainingExample(
            trajectory_id=failure.failure.trajectory_id,
            original_failure_type=failure.failure.failure_type,
            root_cause=failure.root_cause,
            prompt=prompt_history,
            completions=[[{"role": "assistant", "content": c}] for c in candidates],
            rewards=[0.0 for _ in candidates],
            chosen=None,
            rejected=[{"role": "assistant", "content": rejected_text}],
        )
        setattr(temp_example, "_kyc_meta", meta)
        self.validator.register_context(failure.failure.trajectory_id, meta)

        validated: list[tuple[str, float, list[str]]] = []
        candidate_rewards: list[float] = []
        for cand in candidates:
            if cand.strip() == rejected_text:
                candidate_rewards.append(-1.0)
                continue
            is_valid, trace = await self.validator.validate_example_with_trace(temp_example, cand)
            if not is_valid:
                candidate_rewards.append(-1.0)
                continue

            # Score verified candidates (favoring tool accuracy + efficiency)
            parsed_calls = self.validator._parse_candidate_tool_calls(cand)
            if parsed_calls:
                tac = self.traj_evaluator._calculate_tac(
                    {"tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in parsed_calls]}
                )
                step_ratio = 1.0 if len(parsed_calls) == 1 else max(0.7, 1.0 - 0.1 * len(parsed_calls))
                score = tac + 0.5 * step_ratio + 1.0
            else:
                score = 2.0
            candidate_rewards.append(score)
            validated.append((cand, score, trace))

        # HARD REJECT rule: never emit an unvalidated completion into DPO chosen!
        if not validated:
            return None

        validated.sort(key=lambda item: item[1], reverse=True)
        best_chosen, best_score, best_trace = validated[0]
        if best_chosen.strip() == rejected_text:
            return None

        final_meta: dict[str, Any] = {
            **meta,
            "reward_delta": round(best_score, 4),
            "validation_trace": best_trace,
            "valid_candidate_count": len(validated),
            "total_candidate_count": len(candidates),
        }
        final_ex = TrainingExample(
            trajectory_id=failure.failure.trajectory_id,
            original_failure_type=failure.failure.failure_type,
            root_cause=failure.root_cause,
            prompt=prompt_history,
            completions=[[{"role": "assistant", "content": c}] for c in candidates],
            rewards=candidate_rewards,
            chosen=[{"role": "assistant", "content": best_chosen}],
            rejected=[{"role": "assistant", "content": rejected_text}],
            salvaged_at_attempt=1,
            is_negative_only=False,
        )
        setattr(final_ex, "_kyc_meta", final_meta)
        self.validator.register_context(failure.failure.trajectory_id, final_meta)
        return final_ex


def export_dpo_pairs_jsonl(
    examples: list[TrainingExample],
    output_path: str | Path,
) -> Path:
    """Write validated ChatML OpenAI-message (prompt, chosen, rejected) DPO JSONL."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for ex in examples:
        if not ex.chosen or not ex.rejected:
            continue
        meta = getattr(ex, "_kyc_meta", None) or {}
        chosen_msgs = ex.chosen if isinstance(ex.chosen, list) else [{"role": "assistant", "content": str(ex.chosen)}]
        rejected_msgs = ex.rejected if isinstance(ex.rejected, list) else [{"role": "assistant", "content": str(ex.rejected)}]
        reward_delta = meta.get("reward_delta")
        if reward_delta is None and ex.rewards:
            reward_delta = round(max(ex.rewards), 4)
        record = {
            "example_id": meta.get("example_id") or f"ex_{ex.trajectory_id}",
            "trajectory_id": ex.trajectory_id,
            "root_cause": ex.root_cause or meta.get("root_cause"),
            "reward_delta": reward_delta,
            "prompt": ex.prompt,
            "chosen": chosen_msgs,
            "rejected": rejected_msgs,
        }
        lines.append(json.dumps(record))
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


async def run_self_heal_cycle(
    run_results: list[dict[str, Any]],
    db_path: str | Path | None = None,
    synthesizer: Callable[[list[dict[str, str]], ClassifiedFailure], list[str]] | None = None,
    dpo_output_path: str | Path | None = None,
    min_failures_to_trigger: int = 12,
    failure_rate_threshold: float = 0.20,
    llm_classifier: Callable[[Failure], str] | None = None,
) -> dict[str, Any]:
    """Run Detect -> Guarded Classify -> Replay-Validated Synthesize -> Trigger & Diversity Audit."""
    detector = KYCFailureDetector()
    failures = detector.detect_from_runs(run_results)
    event_logs = [r["event_log"] for r in run_results if isinstance(r.get("event_log"), EventLog)]
    project_heal_failures = detector.detect_from_event_logs(event_logs)

    classifier = KYCFailureClassifier(llm_classifier=llm_classifier)
    classified = await classifier.classify_batch(failures)

    validator = ComplianceReplayValidator(db_path=db_path)
    generator = KYCTrainingExampleGenerator(validator=validator, synthesizer=synthesizer)
    examples = await generator.generate_batch(classified)

    trigger_cfg = TriggerConfig(
        total_failures_threshold=min_failures_to_trigger,
        min_examples_ready=min_failures_to_trigger,
        dominance_ratio=0.85,
    )
    buffer = TrainingBuffer()
    for ex in examples:
        buffer.add(ex)
    trigger = RetrainingTrigger(config=trigger_cfg, buffer=buffer)

    diversity_monitor = BehavioralDiversityMonitor()
    diversity_trajs = normalize_trajectories_for_diversity(run_results)
    for traj in diversity_trajs:
        diversity_monitor.observe_trajectory(traj)
    collapse_alert = diversity_monitor.check_diversity()
    unique_signatures = len(set(diversity_monitor.trajectory_history))
    diversity_score = (
        unique_signatures / len(diversity_monitor.trajectory_history)
        if diversity_monitor.trajectory_history
        else 1.0
    )
    diversity_report = {
        "collapse_alert": collapse_alert,
        "unique_signatures": unique_signatures,
        "observed_trajectories": len(diversity_monitor.trajectory_history),
        "diversity_score": round(diversity_score, 4),
    }

    failure_rate = len(failures) / max(len(run_results), 1)
    should_fire, trigger_reason = trigger.should_trigger()
    trigger_decision = {
        "should_trigger": should_fire,
        "reason": trigger_reason,
        "failure_rate": round(failure_rate, 4),
        "threshold_rate": failure_rate_threshold,
        "buffer_health": buffer.health(),
    }

    exported_path: str | None = None
    if dpo_output_path is not None and examples:
        exported_path = str(export_dpo_pairs_jsonl(examples, dpo_output_path))

    return {
        "total_runs": len(run_results),
        "detected_failures": failures,
        "project_heal_failures": project_heal_failures,
        "classified_failures": classified,
        "examples": examples,
        "buffer": buffer,
        "trigger": trigger,
        "buffer_size": len(examples),
        "failure_rate": failure_rate,
        "trigger_decision": trigger_decision,
        "diversity_report": diversity_report,
        "dpo_jsonl_path": exported_path,
    }


def run_self_heal_cycle_sync(**kwargs: Any) -> dict[str, Any]:
    """Synchronous convenience wrapper around run_self_heal_cycle."""
    return asyncio.run(run_self_heal_cycle(**kwargs))


def run_agenttune_self_heal_loop_bridge(
    failures: list[Failure],
    classifier: KYCFailureClassifier,
    generator: KYCTrainingExampleGenerator,
) -> dict[str, Any]:
    """Invoke AgentTune's native SelfHealLoop + build_dataset cleanly via async-to-sync bridges."""
    loop = SelfHealLoop(
        classifier=as_sync_classifier(classifier),
        generator=as_sync_generator(generator),
    )
    summary = loop.run(failures)
    return {
        "summary": summary,
        "examples": summary["generated"],
        "dataset_rows": summary["dataset"],
    }


def _runs_to_agentic_eval_results(runs: list[dict[str, Any]]) -> list[AgenticEvalResult]:
    results: list[AgenticEvalResult] = []
    for r in runs:
        correct = 1.0 if r.get("verdict") == r.get("oracle_verdict") else 0.0
        no_first_try_err = 0.0 if r.get("first_try_tool_errors") else 1.0
        no_loop = 0.0 if r.get("loop_collapse_detected") else 1.0
        steps = max(1, int(r.get("steps", 1)))
        req_count = max(1, len((r.get("tool_policy") or {}).get("required_tools") or []))
        excess_steps = max(0, steps - (req_count + 1))
        unnecessary_steps_penalty = round(min(1.0, excess_steps * 0.2), 4)
        error_recovery_score = 1.0 if not r.get("first_try_tool_errors") else (0.5 if correct == 1.0 else 0.0)
        results.append(
            AgenticEvalResult(
                trajectory_id=r["trajectory_id"],
                goal_completion_score=correct,
                tool_sequence_validity=no_first_try_err,
                unnecessary_steps_penalty=unnecessary_steps_penalty,
                error_recovery_score=error_recovery_score,
                overall_judge_score=round((correct + no_first_try_err + no_loop) / 3.0, 4),
                tac_score=no_first_try_err,
                ter_score=max(0.0, 1.0 - 0.1 * max(0, steps - 2)),
                lcf_score=0 if no_loop else 1,
            )
        )
    return results


def evaluate_kyc_deployment_gate(
    baseline_runs: list[dict[str, Any]],
    candidate_runs: list[dict[str, Any]],
    min_samples: int = 30,
    min_improvement: float = 0.03,
    task_regression_tol: float = 0.0,
    trajectory_regression_tol: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Run AgentTune DeploymentGate on paired baseline vs candidate KYC runs."""
    gate = DeploymentGate(seed=seed)
    base_by_id = {r["case_id"]: r for r in baseline_runs}
    cand_by_id = {r["case_id"]: r for r in candidate_runs}
    shared_ids = [cid for cid in cand_by_id if cid in base_by_id]

    test_set = [
        {
            "pipeline_id": cid,
            "input_text": cid,
            "expected_verdict": cand_by_id[cid]["oracle_verdict"],
        }
        for cid in shared_ids
    ]
    old_traj_results = _runs_to_agentic_eval_results([base_by_id[cid] for cid in shared_ids])
    new_traj_results = _runs_to_agentic_eval_results([cand_by_id[cid] for cid in shared_ids])

    decision = gate.evaluate_decision(
        test_set=test_set,
        old_model_fn=lambda cid: base_by_id[cid]["verdict"],
        new_model_fn=lambda cid: cand_by_id[cid]["verdict"],
        old_trajectory_results=old_traj_results,
        new_trajectory_results=new_traj_results,
        task_regression_tol=task_regression_tol,
        trajectory_regression_tol=trajectory_regression_tol,
    )

    sample_ok = len(shared_ids) >= min_samples
    improvement_ok = decision.task_delta >= min_improvement or (
        decision.task_delta >= 0.0 and decision.trajectory_delta >= min_improvement
    )
    final_approved = bool(decision.approved and sample_ok and improvement_ok)

    reasons = [decision.reason]
    if not sample_ok:
        reasons.append(f"insufficient_samples({len(shared_ids)}<{min_samples})")
    if not improvement_ok:
        reasons.append(
            f"below_min_improvement(task_delta={decision.task_delta},traj_delta={decision.trajectory_delta},min={min_improvement})"
        )

    return {
        "approved": final_approved,
        "gate_decision": decision,
        "reasons": reasons,
        "shared_cases": len(shared_ids),
    }

