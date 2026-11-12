"""Compliance workflow runner integrating AgentTune DECIDE, Cohere prompt mode, EventLog, and AuditKIT.

Executes the FinCEN/FATF/OFAC KYC triage pipeline:
  1. ``extract_cip`` + ``cip_deterministic_gate`` (real DECIDE ``RulesStage`` + ``SafeEvaluator``).
  2. ``watchlist_screening_agent`` (bounded ReAct tool loop in Cohere prompt mode or native mode,
     with optional ``IsolatedTool`` process isolation).
  3. ``fincen_cdd_edd_oracle_and_router`` + ``route_*`` (real DECIDE ``RouterStage`` + ``OutputStage``).
  4. Dual telemetry: ``AuditWriter`` JSONL stage/completion logs, canonical ``EventLog(tier="full")``
     records, ``pending_reviews/`` JSON escalation tickets, and ``OBSERVED`` ``AgentEpisode`` traces.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Protocol
import uuid

from agenttune.agentic.distill_trainer import _merge_consecutive_same_role
from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.rollout_engines.rollout_factory import (
    DEFAULT_TOOLS_FALLBACK_PROMPT,
    fold_tool_messages_into_user,
)
from agenttune.agentic.rollout_engines.tool_call_parse import extract_tool_calls
from agenttune.decide.audit import AuditWriter
from agenttune.decide.config import ConfigLoader
from agenttune.decide.stages.base import flatten_stage_outputs
from agenttune.decide.stages.rules import SafeEvaluator
from agenttune.decide.graph_runner import GraphRunner
from agenttune.decide.stage_executor import StageExecutor
from agenttune.decide.state import PipelineState
from auditkit.agent_eval import episode_from_eventlog
from auditkit.agent_eval.types import OBSERVED, AgentEpisode
from auditkit.trace import parse_tool_calls
import litellm
import yaml

from src.env_utils import load_project_env
from src.kyc_oracle import compute_required_tools, evaluate_case_oracle, evaluate_cip
from src.kyc_tools import (
    DEFAULT_WATCHLIST_DB,
    build_kyc_tool_registry,
    execute_kyc_tool_call,
    get_openai_tool_schemas,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DECIDE_CFG = PROJECT_ROOT / "config" / "decide_config.yaml"
DEFAULT_PIPELINE_CFG = PROJECT_ROOT / "config" / "kyc_compliance.yaml"
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class ChatAdapter(Protocol):
    """Protocol for injectable student/teacher chat adapters or offline deterministic stubs."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stage: str = "watchlist_screening_agent",
        case: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


class CallableChatAdapter:
    """Wrap a plain callable into a ChatAdapter for deterministic tests or custom engines."""

    def __init__(self, fn: Callable[..., dict[str, Any]]) -> None:
        self._fn = fn

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stage: str = "watchlist_screening_agent",
        case: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._fn(messages=messages, tools=tools, stage=stage, case=case)


def format_messages_for_cohere(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize any multi-turn trace into strict Cohere user/assistant alternating turns.

    Cohere ``Tiny Aya`` and ``Aya Expanse`` Jinja templates ignore ``tools=``, reject ``role='tool'``,
    and enforce alternating ``user`` / ``assistant`` roles after an optional leading ``system`` turn.
    """
    if not messages:
        return []
    folded = fold_tool_messages_into_user(messages)
    cleaned: list[dict[str, Any]] = []
    for msg in folded:
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if role == "assistant" and msg.get("tool_calls") and not (content or "").strip():
            calls_payload = [
                {
                    "tool_name": tc.get("function", {}).get("name") or tc.get("name"),
                    "parameters": _safe_parse_args(
                        tc.get("function", {}).get("arguments")
                        if isinstance(tc.get("function"), dict)
                        else tc.get("arguments")
                    ),
                }
                for tc in msg["tool_calls"]
            ]
            content = json.dumps(calls_payload, ensure_ascii=False)
        cleaned.append({"role": role, "content": "" if content is None else str(content)})

    merged = _merge_consecutive_same_role(cleaned)
    # Guarantee no leading assistant message without preceding user turn
    if merged and merged[0].get("role") == "assistant":
        merged = [{"role": "user", "content": "Evaluate the KYC dossier."}] + merged
    if len(merged) >= 2 and merged[0].get("role") == "system" and merged[1].get("role") == "assistant":
        merged = [merged[0], {"role": "user", "content": "Proceed with compliance analysis."}] + merged[1:]
    return merged


class OpenAICompatibleChatAdapter:
    """Live OpenAI-compatible / HuggingFace Router chat adapter supporting Cohere prompt mode."""

    def __init__(
        self,
        model: str,
        api_base: str = "https://router.huggingface.co/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
        tool_mode: str = "prompt",
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        load_project_env()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.tool_mode = tool_mode

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stage: str = "watchlist_screening_agent",
        case: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del stage, case
        extra_kwargs: dict[str, Any] = {}
        prepared_messages = list(messages)

        if self.tool_mode == "prompt":
            prepared_messages = format_messages_for_cohere(prepared_messages)
        else:
            if tools:
                extra_kwargs["tools"] = tools
                extra_kwargs["tool_choice"] = "auto"
            extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}

        response = litellm.completion(
            model=self.model,
            messages=prepared_messages,
            api_base=self.api_base,
            api_key=self.api_key or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY"),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **extra_kwargs,
        )
        message = response.choices[0].message
        content = getattr(message, "content", None) or ""
        raw_calls = getattr(message, "tool_calls", None) or []
        parsed_calls: list[dict[str, Any]] = []
        for idx, tc in enumerate(raw_calls):
            fn = getattr(tc, "function", None) or {}
            tc_name = getattr(fn, "name", None) if not isinstance(fn, dict) else fn.get("name")
            tc_args = getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments")
            parsed_calls.append(
                {
                    "id": getattr(tc, "id", None) or f"call_{idx}_{uuid.uuid4().hex[:6]}",
                    "type": "function",
                    "function": {
                        "name": tc_name,
                        "arguments": tc_args if isinstance(tc_args, str) else json.dumps(tc_args or {}),
                    },
                }
            )
        if not parsed_calls and content:
            extracted = _extract_inline_tool_calls(content)
            for idx, ext in enumerate(extracted):
                parsed_calls.append(
                    {
                        "id": f"call_{idx}_{uuid.uuid4().hex[:6]}",
                        "type": "function",
                        "function": {
                            "name": ext["name"],
                            "arguments": json.dumps(ext["arguments"]),
                        },
                    }
                )
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": parsed_calls,
        }


def load_decide_pipeline_config(
    decide_config_path: str | Path | None = None,
    pipeline_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load, merge, and validate DECIDE configs cleanly."""
    load_project_env()
    decide_path = Path(decide_config_path or DEFAULT_DECIDE_CFG)
    pipe_path = Path(pipeline_config_path or DEFAULT_PIPELINE_CFG)

    with decide_path.open("r", encoding="utf-8") as fh:
        decide_cfg = yaml.safe_load(fh) or {}
    with pipe_path.open("r", encoding="utf-8") as fh:
        pipe_cfg = yaml.safe_load(fh) or {}

    merged = ConfigLoader.deep_merge(decide_cfg, pipe_cfg)
    ConfigLoader._validate(merged)
    ConfigLoader._generate_edges(merged)
    return merged


def compile_decide_graph(config: dict[str, Any] | None = None) -> GraphRunner:
    """Compile the DECIDE StateGraph using AgentTune GraphRunner cleanly on CPU."""
    cfg = config or load_decide_pipeline_config()
    return GraphRunner(cfg)


def canonical_tool_call_key(tool_name: str, arguments: Any) -> str:
    """Canonical string key for deduplicating identical repeated tool calls."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {"_raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"_val": arguments}
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'))}"


def _safe_parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_raw": raw}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip()
    m = JSON_BLOCK_RE.search(cleaned)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_verdict_from_text(text: str) -> tuple[str | None, str]:
    obj = _extract_json_object(text)
    if obj and isinstance(obj.get("verdict"), str):
        v = obj["verdict"].strip().upper()
        if v in {"APPROVE", "REVIEW", "DENY"}:
            return v, str(obj.get("reason") or "")
    for v in ("DENY", "REVIEW", "APPROVE"):
        if re.search(rf'\bverdict["\']?\s*[:=]\s*["\']?{v}\b', text, re.IGNORECASE):
            return v, text.strip()
    return None, text.strip()


def _extract_inline_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from Cohere bare JSON arrays, action tags, XML, or OpenAI shapes."""
    if not isinstance(text, str) or not text.strip():
        return []

    parsed: list[dict[str, Any]] = []
    # 1. Primary: AgentTune's multi-format parser (supports Cohere tool_name/parameters, XML, JSON arrays)
    try:
        extracted = extract_tool_calls(None, text) or []
    except Exception:
        extracted = []

    for item in extracted:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else None
        name = (fn.get("name") if fn else None) or item.get("name") or item.get("tool_name")
        args_raw = (
            fn.get("arguments")
            if fn
            else item.get("arguments", item.get("parameters", {}))
        )
        if name:
            parsed.append({"name": str(name), "arguments": _safe_parse_args(args_raw)})

    if parsed:
        return parsed

    # 2. Secondary: AuditKIT's Cohere/Command-R7B/Hermes trace parser
    try:
        turns = parse_tool_calls(text) or []
    except Exception:
        turns = []
    for turn in turns:
        for tc in turn:
            if tc.name:
                parsed.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments if isinstance(tc.arguments, dict) else {},
                    }
                )
    return parsed


def _build_system_prompt(tool_schemas: list[dict[str, Any]], tool_mode: str = "prompt") -> str:
    compact_tools = [
        {
            "name": s["function"]["name"],
            "description": s["function"].get("description", ""),
            "parameters": s["function"].get("parameters", {}),
        }
        for s in tool_schemas
    ]
    compliance_rules = (
        "Strict FinCEN/FATF/OFAC KYC Compliance Rules:\n"
        "1. First verify FinCEN CIP completeness (name, valid ISO-2 country, calendar-valid YYYY-MM-DD DOB "
        "for individuals, government ID). If any required field is missing or invalid, DO NOT call screening "
        "tools; immediately output JSON {\"verdict\": \"REVIEW\", \"reason\": \"...\"}.\n"
        "2. Always pass strict uppercase 2-letter ISO-2 country codes matching ^[A-Z]{2}$ "
        "(e.g. 'US', 'GB', 'KY'), NEVER 'USA' or 'UK'.\n"
        "3. Call screen_crypto_wallet ONLY when a non-empty crypto_wallet is present in the dossier.\n"
        "4. Never repeat an identical tool call after receiving its result.\n"
        "5. Confirmed OFAC PRIMARY/AKA_STRONG/UN/EU sanctions, BIS denied parties, or sanctioned wallets -> DENY.\n"
        "6. OFAC AKA_WEAK hits, offshore secrecy hubs, FATF Black/Grey, or inconsistent wealth -> REVIEW.\n"
        "7. Once done, output strict JSON: {\"verdict\": \"APPROVE\"|\"REVIEW\"|\"DENY\", \"reason\": \"...\", \"cited_tools\": [...]}."
    )
    if tool_mode == "prompt":
        prompt_header = DEFAULT_TOOLS_FALLBACK_PROMPT.format(
            tools=json.dumps(compact_tools, indent=2)
        ).replace(
            "use it to answer the user's original question as plain text.",
            "use it to answer with the required FinCEN/FATF/OFAC JSON verdict.",
        )
        return prompt_header + "\n\n" + compliance_rules
    tool_names = [t["name"] for t in compact_tools]
    return (
        "You are AutoHeal-KYC, a strict FinCEN/FATF/OFAC KYC & AML compliance agent.\n"
        f"Available tools: {', '.join(tool_names)}.\n\n"
        + compliance_rules
    )


def _format_case_prompt(case: dict[str, Any]) -> str:
    payload = {
        "case_id": case.get("case_id"),
        "customer_name": case.get("customer_name"),
        "entity_type": case.get("entity_type"),
        "raw_country": case.get("raw_country") or case.get("country_iso2"),
        "date_of_birth": case.get("date_of_birth"),
        "id_number": case.get("id_number"),
        "declared_assets_usd": case.get("declared_assets_usd"),
        "annual_income_usd": case.get("annual_income_usd"),
        "expected_monthly_tx_usd": case.get("expected_monthly_tx_usd"),
        "source_of_wealth": case.get("source_of_wealth"),
        "offshore_jurisdiction": case.get("offshore_jurisdiction"),
        "crypto_wallet": case.get("crypto_wallet"),
        "crypto_currency": case.get("crypto_currency"),
        "beneficial_ownership_verified": case.get("beneficial_ownership_verified", True),
    }
    return f"Evaluate this customer onboarding dossier:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"


def _run_coro_sync(coro: Any) -> Any:
    """Execute an async AgentTune DECIDE stage cleanly inside sync or active-loop contexts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class KYCAgentRunner:
    """Executes the DECIDE workflow + bounded prompt-mode ReAct loop with EventLog and AuditKIT telemetry."""

    def __init__(
        self,
        adapter: ChatAdapter | Callable[..., dict[str, Any]],
        db_path: str | Path | None = None,
        audit_dir: str | Path | None = None,
        pending_reviews_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
        max_steps: int | None = None,
        tool_mode: str | None = None,
        isolate_tools: bool | None = None,
    ) -> None:
        self.config = config or load_decide_pipeline_config()
        self.adapter = adapter if hasattr(adapter, "chat") else CallableChatAdapter(adapter)
        self.db_path = Path(db_path or PROJECT_ROOT / self.config["agentic"]["watchlist_db"])
        self.audit_dir = Path(audit_dir or PROJECT_ROOT / self.config["audit"]["output_dir"])
        self.pending_reviews_dir = Path(
            pending_reviews_dir or PROJECT_ROOT / self.config["agentic"]["pending_reviews_dir"]
        )
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.pending_reviews_dir.mkdir(parents=True, exist_ok=True)

        self.max_steps = int(max_steps or self.config["agentic"].get("max_steps", 6))
        self.tool_mode = str(
            tool_mode or self.config.get("models", {}).get("student_tool_call_mode", "prompt")
        )
        tool_iso_cfg = self.config.get("agentic", {}).get("tool_isolation", False)
        self.isolate_tools = (
            bool(isolate_tools)
            if isolate_tools is not None
            else (bool(tool_iso_cfg.get("enabled", False)) if isinstance(tool_iso_cfg, dict) else bool(tool_iso_cfg))
        )
        self.registry = build_kyc_tool_registry(
            self.db_path,
            include_builtins=False,
            isolate=self.isolate_tools,
        )
        self.tool_schemas = get_openai_tool_schemas(self.db_path)
        self.audit_file = self.audit_dir / "kyc_audit.jsonl"
        self.event_log_file = self.audit_dir / "kyc_events.jsonl"
        self.audit_writer = AuditWriter(str(self.audit_file))
        self.graph_runner = compile_decide_graph(self.config)
        self._stages_by_id = {s["id"]: s for s in self.config.get("stages", [])}

    def _log_audit_stage(
        self,
        pipeline_state: PipelineState,
        stage_name: str,
        stage_type: str,
        result_payload: dict[str, Any],
    ) -> None:
        stage_cfg = dict(self._stages_by_id.get(stage_name) or {"id": stage_name, "type": stage_type})
        stage_cfg["type"] = stage_type
        pipeline_state.add_stage_output(stage_name, result_payload)
        self.audit_writer.log_stage(pipeline_state, stage_cfg, result_payload)

    def _write_pending_review_ticket(
        self,
        case: dict[str, Any],
        trajectory_id: str,
        verdict: str,
        reason: str,
        oracle_res: dict[str, Any],
    ) -> Path:
        ticket_path = self.pending_reviews_dir / f"{case.get('case_id', trajectory_id)}.json"
        payload = {
            "case_id": case.get("case_id"),
            "trajectory_id": trajectory_id,
            "verdict": verdict,
            "customer_name": case.get("customer_name"),
            "escalation_reasons": oracle_res.get("reason_codes", []),
            "agent_reason": reason,
            "oracle_verdict": oracle_res.get("verdict"),
        }
        ticket_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ticket_path

    def run_case(self, case: dict[str, Any], enforce_oracle_router: bool = False) -> dict[str, Any]:
        """Execute a full KYC case trajectory via real DECIDE stages + EventLog + AuditKIT capture."""
        started = time.time()
        case_id = str(case.get("case_id") or f"case_{uuid.uuid4().hex[:8]}")
        trajectory_id = f"traj_{case_id}_{uuid.uuid4().hex[:6]}"
        formatted_input = _format_case_prompt(case)

        pipeline_state = PipelineState(
            pipeline_id=trajectory_id,
            template_id=self.config["id"],
            template_version=self.config["version"],
            input_text=formatted_input,
            input_hash=hashlib.sha256(formatted_input.encode("utf-8")).hexdigest()[:16],
            config=self.config,
        )

        executed_stages: list[str] = []

        # Stage 1 (DECIDE RulesStage): Deterministic CIP extraction & validation (zero fake llm_call!)
        extract_cfg = self._stages_by_id.get("extract_cip", {"id": "extract_cip", "type": "rules", "rules": []})
        _run_coro_sync(StageExecutor.get("rules")(extract_cfg).execute(pipeline_state, extract_cfg))
        cip_res = evaluate_cip(case)
        extracted_cip = {
            "customer_name": case.get("customer_name"),
            "entity_type": case.get("entity_type"),
            "raw_country": case.get("raw_country") or case.get("country_iso2"),
            "country_iso2": cip_res["country_iso2"],
            "date_of_birth": case.get("date_of_birth"),
            "id_number": case.get("id_number"),
            "cip_complete": cip_res["cip_complete"],
            "country_iso2_valid": cip_res["country_iso2_valid"],
            "dob_format_valid": cip_res["dob_format_valid"],
        }
        self._log_audit_stage(pipeline_state, "extract_cip", "rules", extracted_cip)
        executed_stages.append("extract_cip")

        # Stage 2 (DECIDE RulesStage + SafeEvaluator): CIP completeness gate
        gate_cfg = self._stages_by_id.get(
            "cip_deterministic_gate",
            {"id": "cip_deterministic_gate", "type": "rules", "rules": []},
        )
        gate_stage_res = _run_coro_sync(
            StageExecutor.get("rules")(gate_cfg).execute(pipeline_state, gate_cfg)
        )
        safe_eval = SafeEvaluator()
        strict_gate_ok = bool(
            safe_eval.eval(
                "extract_cip.output.cip_complete == true and extract_cip.output.country_iso2_valid == true and extract_cip.output.dob_format_valid == true",
                flatten_stage_outputs({"extract_cip": extracted_cip}),
            )
        )
        gate_passed = (not bool(gate_stage_res.get("failed", False))) and strict_gate_ok
        self._log_audit_stage(
            pipeline_state,
            "cip_deterministic_gate",
            "rules",
            {
                "passed": gate_passed,
                "missing_fields": cip_res["missing_fields"],
                "next_stage": gate_stage_res.get("goto") or gate_cfg.get("next"),
            },
        )
        executed_stages.append("cip_deterministic_gate")

        # Stage 3: Bounded Screening Agent (Prompt Mode by default for Cohere + canonical EventLog)
        sys_prompt = _build_system_prompt(self.tool_schemas, tool_mode=self.tool_mode)
        raw_messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": formatted_input},
        ]
        event_log = EventLog(tier="full")
        event_log.append(Event(EventKind.OBSERVATION, {"text": formatted_input}))

        tool_call_records: list[dict[str, Any]] = []
        tool_observations: list[dict[str, Any]] = []
        first_try_tool_errors: list[dict[str, Any]] = []
        call_key_counts: dict[str, int] = {}
        loop_collapse_detected = False
        agent_verdict: str | None = None
        agent_reason = ""
        dangling_tool_calls: list[str] = []
        steps = 0
        final_answer_text = ""

        skip_agent_turn = enforce_oracle_router and not gate_passed
        if skip_agent_turn:
            agent_verdict = "REVIEW"
            agent_reason = "DECIDE cip_deterministic_gate blocked incomplete CIP dossier prior to screening."
            final_answer_text = json.dumps({"verdict": agent_verdict, "reason": agent_reason})
            event_log.append(Event(EventKind.TURN_COMPLETE, {"step": 1, "outcome": "cip_gate_short_circuit"}))
        else:
            executed_stages.append("watchlist_screening_agent")
            while steps < self.max_steps:
                steps += 1
                outgoing_msgs = (
                    format_messages_for_cohere(raw_messages)
                    if self.tool_mode == "prompt"
                    else list(raw_messages)
                )
                assistant_msg = self.adapter.chat(
                    messages=outgoing_msgs,
                    tools=self.tool_schemas,
                    stage="watchlist_screening_agent",
                    case=case,
                )
                content = str(assistant_msg.get("content") or "")
                raw_tool_calls = list(assistant_msg.get("tool_calls") or [])

                parsed_tool_calls: list[dict[str, Any]] = []
                if raw_tool_calls:
                    for idx, tc in enumerate(raw_tool_calls):
                        fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                        t_name = tc.get("name") or fn.get("name")
                        raw_args = (
                            tc.get("arguments")
                            if "arguments" in tc
                            else fn.get("arguments", {})
                        )
                        parsed_tool_calls.append(
                            {
                                "id": tc.get("id") or f"call_{steps}_{idx}",
                                "name": str(t_name or ""),
                                "arguments": _safe_parse_args(raw_args),
                                "raw_arguments": raw_args,
                            }
                        )
                elif content:
                    for idx, tc in enumerate(_extract_inline_tool_calls(content)):
                        parsed_tool_calls.append(
                            {
                                "id": f"call_{steps}_{idx}",
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                                "raw_arguments": json.dumps(tc["arguments"]),
                            }
                        )

                raw_messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        **({"tool_calls": raw_tool_calls} if raw_tool_calls else {}),
                    }
                )

                if not parsed_tool_calls:
                    final_answer_text = content
                    verdict_found, reason_found = _parse_verdict_from_text(content)
                    if verdict_found:
                        agent_verdict = verdict_found
                        agent_reason = reason_found
                    event_log.append(Event(EventKind.TURN_COMPLETE, {"step": steps, "outcome": "terminal_text"}))
                    break

                # Execute each tool call cleanly (unwrapped action for EventLog.to_audit_records + AuditKIT pairing)
                for tc in parsed_tool_calls:
                    t_name = tc["name"]
                    tc_id = tc["id"]
                    parsed_args = tc["arguments"]

                    key = canonical_tool_call_key(t_name, parsed_args)
                    call_key_counts[key] = call_key_counts.get(key, 0) + 1
                    if call_key_counts[key] >= 3:
                        loop_collapse_detected = True

                    event_log.append(
                        Event(
                            EventKind.TOOL_CALL,
                            {"action": {"name": t_name, "arguments": parsed_args, "id": tc_id}},
                        )
                    )
                    exec_res = execute_kyc_tool_call(self.registry, t_name, parsed_args)
                    event_log.append(
                        Event(
                            EventKind.TOOL_RESULT,
                            {
                                "output": exec_res.get("output") if exec_res["success"] else exec_res.get("error"),
                                "tool": t_name,
                                "call_id": tc_id,
                                "success": exec_res["success"],
                                "error": exec_res.get("error"),
                            },
                        )
                    )
                    tool_call_records.append(
                        {
                            "id": tc_id,
                            "name": t_name,
                            "arguments": parsed_args,
                            "canonical_key": key,
                            "step": steps,
                            "success": bool(exec_res["success"]),
                            "error": exec_res.get("error"),
                        }
                    )
                    obs = {
                        "id": tc_id,
                        "tool": t_name,
                        "arguments": parsed_args,
                        "success": bool(exec_res["success"]),
                        "result": exec_res.get("output"),
                        "output": exec_res.get("output"),
                        "error": exec_res.get("error"),
                        "step": steps,
                        "assistant_message": {
                            "role": "assistant",
                            "content": content,
                            **({"tool_calls": raw_tool_calls} if raw_tool_calls else {}),
                        },
                    }
                    tool_observations.append(obs)
                    if not exec_res["success"]:
                        first_try_tool_errors.append(obs)

                    raw_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": t_name,
                            "content": json.dumps(
                                exec_res["output"] if exec_res["success"] else {"error": exec_res["error"]}
                            ),
                        }
                    )

                self._log_audit_stage(
                    pipeline_state,
                    "watchlist_screening_agent",
                    "llm_call",
                    {
                        "step": steps,
                        "tool_calls": [c["name"] for c in parsed_tool_calls],
                        "errors": [o["error"] for o in tool_observations[-len(parsed_tool_calls) :] if not o["success"]],
                    },
                )
                event_log.append(Event(EventKind.TURN_COMPLETE, {"step": steps, "outcome": "tool_turn"}))

                if loop_collapse_detected:
                    break

            if not loop_collapse_detected and steps >= self.max_steps and agent_verdict is None:
                loop_collapse_detected = True

        # Oracle computation & policy audit
        oracle_res = evaluate_case_oracle(case, tool_observations)
        oracle_truth = str(case.get("expected_verdict") or oracle_res["verdict"])
        policy = oracle_res["tool_policy"]
        reason_codes = list(oracle_res.get("reason_codes") or [])
        oracle_res["review_reasons"] = reason_codes
        oracle_res["escalation_required"] = oracle_truth != "APPROVE"

        # Dangling tool call check
        observed_ids = {o["id"] for o in tool_observations}
        for c in tool_call_records:
            if c["id"] not in observed_ids:
                dangling_tool_calls.append(c["id"])

        resolved_verdict = (
            oracle_truth
            if enforce_oracle_router
            else (agent_verdict or ("REVIEW" if loop_collapse_detected or not gate_passed else "APPROVE"))
        )
        if not final_answer_text:
            final_answer_text = json.dumps(
                {
                    "verdict": resolved_verdict,
                    "reason": agent_reason or oracle_res["explanation"],
                }
            )

        self._log_audit_stage(
            pipeline_state,
            "watchlist_screening_agent",
            "llm_call",
            {
                "resolved_verdict": resolved_verdict,
                "agent_verdict": agent_verdict,
                "oracle_verdict": oracle_truth,
                "sanctions_or_wallet_match": any("SANCTIONS" in rc or "EXPORT" in rc or "WALLET" in rc for rc in reason_codes),
                "weak_alias_review": any("AKA_WEAK" in rc for rc in reason_codes),
                "edd_required": bool(oracle_res.get("edd_required", False)),
                "bo_verification_required": any("BO_" in rc or "BENEFICIAL" in rc for rc in reason_codes),
                "escalation_required": oracle_res["escalation_required"],
                "loop_collapse_detected": loop_collapse_detected,
                "tools_called": [t["name"] for t in tool_call_records],
            },
        )

        # Stage 4 (DECIDE RouterStage): Execute real RouterStage over SafeEvaluator conditions
        router_cfg = self._stages_by_id.get(
            "fincen_cdd_edd_oracle_and_router",
            {"id": "fincen_cdd_edd_oracle_and_router", "type": "router", "default": "route_review"},
        )
        router_res = _run_coro_sync(
            StageExecutor.get("router")(router_cfg).execute(pipeline_state, router_cfg)
        )
        routed_next = router_res.get("goto") or router_res.get("next_stage")
        expected_route_map = {"DENY": "route_deny", "APPROVE": "route_approve", "REVIEW": "route_review"}
        terminal_stage = (
            routed_next
            if isinstance(routed_next, str) and routed_next in expected_route_map.values()
            else expected_route_map.get(resolved_verdict, "route_review")
        )
        if enforce_oracle_router and not gate_passed:
            terminal_stage = "route_review"
        final_verdict = (
            "DENY"
            if terminal_stage == "route_deny"
            else ("APPROVE" if terminal_stage == "route_approve" else "REVIEW")
        )

        self._log_audit_stage(
            pipeline_state,
            "fincen_cdd_edd_oracle_and_router",
            "router",
            {
                "terminal_stage": terminal_stage,
                "router_next_stage": routed_next,
                "agent_verdict": agent_verdict,
                "oracle_verdict": oracle_truth,
                "resolved_verdict": final_verdict,
                "escalation_reasons": reason_codes,
            },
        )
        executed_stages.append("fincen_cdd_edd_oracle_and_router")

        # Stage 5 (DECIDE OutputStage): Persist pending review tickets & stamp terminal state
        artifacts_list: list[str] = []
        pending_ticket_path: str | None = None
        if final_verdict == "REVIEW":
            ticket_p = self._write_pending_review_ticket(
                case, trajectory_id, final_verdict, agent_reason or oracle_res["explanation"], oracle_res
            )
            pending_ticket_path = str(ticket_p)
            artifacts_list.append(pending_ticket_path)

        output_cfg = self._stages_by_id.get(
            terminal_stage,
            {"id": terminal_stage, "type": "output", "verdict": final_verdict, "verdict_label": final_verdict, "confidence": 0.9},
        )
        _run_coro_sync(
            StageExecutor.get("output")(output_cfg).execute(pipeline_state, output_cfg)
        )
        self._log_audit_stage(
            pipeline_state,
            terminal_stage,
            "output",
            {"verdict": final_verdict, "pending_review_ticket": pending_ticket_path},
        )
        executed_stages.append(terminal_stage)

        pipeline_state.verdict = final_verdict
        pipeline_state.reason = agent_reason or oracle_res["explanation"]
        pipeline_state.is_complete = not loop_collapse_detected and len(dangling_tool_calls) == 0
        pipeline_state.elapsed_seconds = time.time() - started
        self.audit_writer.write(pipeline_state)

        # Finalize canonical EventLog & append to kyc_events.jsonl for AuditKIT episodes_from_agenttune
        event_log.append(Event(EventKind.TEXT, {"text": final_answer_text}))
        event_log.append(
            Event(
                EventKind.REWARD,
                {"value": 1.0 if final_verdict == oracle_truth else 0.0},
                scope="episode",
            )
        )
        event_payload = {
            "id": trajectory_id,
            "case_id": case_id,
            "tier": "full",
            "events": [
                {
                    "kind": ev.kind.value,
                    "payload": ev.payload,
                    "ts": getattr(ev, "ts", None),
                    "scope": getattr(ev, "scope", None),
                }
                for ev in event_log.events
            ],
        }
        with self.event_log_file.open("a", encoding="utf-8") as efh:
            efh.write(json.dumps(event_payload, ensure_ascii=False, default=str) + "\n")

        final_state_dict = {
            "verdict": final_verdict,
            "agent_verdict": agent_verdict,
            "oracle_verdict": oracle_truth,
            "cip_complete": cip_res["cip_complete"],
            "abstained_on_incomplete_cip": (not cip_res["cip_complete"]) and len(tool_call_records) == 0,
            "loop_collapse": loop_collapse_detected,
            "first_try_errors": len(first_try_tool_errors),
            "pending_review_ticket": pending_ticket_path,
            "latency_ms": int((time.time() - started) * 1000),
            "tool_mode": self.tool_mode,
        }
        base_ep = episode_from_eventlog(event_log, source_id=trajectory_id)
        full_episode: AgentEpisode = dataclasses.replace(
            base_ep,
            case_id=case_id,
            final_state=final_state_dict,
            artifacts=artifacts_list,
            coverage={
                **base_ep.coverage,
                "final_state": OBSERVED,
                "artifacts": OBSERVED,
            },
            metadata={
                **base_ep.metadata,
                "task": formatted_input,
                "scenario_tag": case.get("scenario_tag"),
                "required_tools": policy["required_tools"],
                "decide_stages": executed_stages,
            },
        )

        outgoing_final_messages = (
            format_messages_for_cohere(raw_messages) if self.tool_mode == "prompt" else raw_messages
        )

        return {
            "case_id": case_id,
            "trajectory_id": trajectory_id,
            "case": case,
            "verdict": final_verdict,
            "agent_verdict": agent_verdict,
            "oracle_verdict": oracle_truth,
            "oracle_details": oracle_res,
            "cip": cip_res,
            "tool_policy": policy,
            "reason": agent_reason,
            "messages": outgoing_final_messages,
            "raw_messages": raw_messages,
            "tool_calls": tool_call_records,
            "tool_observations": tool_observations,
            "first_try_tool_errors": first_try_tool_errors,
            "loop_collapse_detected": loop_collapse_detected,
            "dangling_tool_calls": dangling_tool_calls,
            "steps": steps,
            "terminal_stage": terminal_stage,
            "pending_review_ticket": pending_ticket_path,
            "decide_stages_executed": executed_stages,
            "event_log": event_log,
            "episode": full_episode,
        }
