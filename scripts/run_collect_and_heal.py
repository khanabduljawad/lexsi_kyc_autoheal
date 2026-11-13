"""Phase 2 CLI: Collect baseline/Split B rollouts, classify, replay-validate, and heal to DPO JSONL.

Supports:
  --mock   Run deterministic offline CPU dry-run demonstrating all 5 failure root causes.
  (default) Run live rollout & closed-loop teacher via HF Inference Providers (requires HF_TOKEN).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_utils import load_project_env
from src.kyc_agent_runner import (
    CallableChatAdapter,
    KYCAgentRunner,
    OpenAICompatibleChatAdapter,
    load_decide_pipeline_config,
)
from src.kyc_oracle import compute_required_tools, evaluate_case_oracle, evaluate_cip
from src.patch_heal import run_self_heal_cycle_sync

_FOLDED_TOOL_RE = re.compile(
    r"Tool result for ([a-zA-Z0-9_]+):\s*(.*?)(?=\nUse this result to answer the original question\.|\Z)",
    re.DOTALL,
)


def _extract_tool_history(messages: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Extract executed tool names and parsed observations from native OR Cohere-folded messages."""
    names: list[str] = []
    observations: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            t_name = str(msg.get("name") or "")
            raw = str(msg.get("content") or "")
            names.append(t_name)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"raw": raw}
            ok = not (isinstance(parsed, dict) and "error" in parsed and len(parsed) == 1)
            observations.append({"tool": t_name, "success": ok, "result": parsed if ok else None})
        elif role == "user":
            content = str(msg.get("content") or "")
            for t_name, raw in _FOLDED_TOOL_RE.findall(content):
                raw_clean = raw.strip()
                names.append(t_name)
                try:
                    parsed = json.loads(raw_clean)
                except json.JSONDecodeError:
                    parsed = {"raw": raw_clean}
                ok = not (isinstance(parsed, dict) and "error" in parsed and len(parsed) == 1)
                observations.append({"tool": t_name, "success": ok, "result": parsed if ok else None})
    return names, observations


def _tool_args_for_case(req: str, c: dict[str, Any]) -> dict[str, Any]:
    country = c.get("expected_country_iso2") or "US"
    if req == "screen_sanctions":
        return {
            "name": c["customer_name"],
            "entity_type": c["entity_type"],
            "country_iso2": country,
            "date_of_birth": c.get("date_of_birth"),
        }
    if req == "screen_export_control":
        return {"entity_name": c["customer_name"], "country_iso2": country}
    if req == "screen_pep_adverse_media":
        return {
            "name": c["customer_name"],
            "country_iso2": country,
            "date_of_birth": c.get("date_of_birth"),
        }
    if req == "check_fatf_jurisdiction_risk":
        return {"country_iso2": country}
    if req == "screen_crypto_wallet":
        return {
            "wallet_address": c["crypto_wallet"],
            "currency": c.get("crypto_currency") or "ETH",
        }
    if req == "verify_source_of_wealth":
        return {
            "customer_name": c["customer_name"],
            "annual_income_usd": float(c.get("annual_income_usd") or 0.0),
            "declared_assets_usd": float(c.get("declared_assets_usd") or 0.0),
            "expected_monthly_tx_usd": float(c.get("expected_monthly_tx_usd") or 0.0),
            "source_of_wealth": c.get("source_of_wealth") or "salary",
            "offshore_jurisdiction": c.get("offshore_jurisdiction"),
        }
    return {}


def _format_tool_response(
    tool_name: str,
    arguments: dict[str, Any],
    call_id: str,
    tool_mode: str = "prompt",
) -> dict[str, Any]:
    if tool_mode == "prompt":
        return {
            "role": "assistant",
            "content": json.dumps([{"tool_name": tool_name, "parameters": arguments}], ensure_ascii=False),
        }
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def load_jsonl_cases(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read KYC cases from a JSONL split file cleanly."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def build_deterministic_student_stub(
    healed: bool = False,
    tool_mode: str = "prompt",
) -> CallableChatAdapter:
    """Create a deterministic offline student simulator covering baseline vs healed trajectories.

    Emits Cohere prompt-mode JSON arrays (``[{"tool_name": ..., "parameters": ...}]``) by default
    while supporting native ``tool_calls`` and parsing both native ``role='tool'`` and folded
    Cohere ``user`` tool-result messages.
    """

    def _fn(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stage: str = "watchlist_screening_agent",
        case: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del tools, stage
        c = case or {}
        tag = c.get("scenario_tag", "")
        policy = compute_required_tools(c)
        executed, observations = _extract_tool_history(messages)
        country = c.get("expected_country_iso2") or "US"

        if not healed:
            # 1. Incomplete CIP failure: tries calling screen_sanctions instead of abstaining
            if tag == "abstain_missing_cip":
                if not executed:
                    return _format_tool_response(
                        "screen_sanctions",
                        {
                            "name": c["customer_name"],
                            "entity_type": c["entity_type"],
                            "country_iso2": "US",
                        },
                        "call_bad_abstain",
                        tool_mode=tool_mode,
                    )
                return {"role": "assistant", "content": json.dumps({"verdict": "REVIEW", "reason": "late"})}

            # 2. ISO-2 format error: passes 'USA' on turn 1 even if recovering on turn 2
            if tag == "iso2_format_challenge" and len(executed) == 0:
                bad_country = (
                    "USA"
                    if country == "US"
                    else ("UK" if country == "GB" else str(c.get("raw_country") or "USA"))
                )
                return _format_tool_response(
                    "screen_sanctions",
                    {
                        "name": c["customer_name"],
                        "entity_type": c["entity_type"],
                        "country_iso2": bad_country,
                    },
                    "call_bad_iso",
                    tool_mode=tool_mode,
                )

            # 3. Loop collapse: repeats identical clean sanctions call 3x on clean benign individual
            if tag == "clean_benign_individual" and len(executed) < 4:
                return _format_tool_response(
                    "screen_sanctions",
                    {
                        "name": c["customer_name"],
                        "entity_type": c["entity_type"],
                        "country_iso2": country,
                        "date_of_birth": c.get("date_of_birth"),
                    },
                    f"call_loop_{len(executed)}",
                    tool_mode=tool_mode,
                )

            # 4. Hallucinated output: APPROVE without screening on clean benign organizations
            if tag == "clean_benign_organization" and int(c["case_id"][-1]) % 2 == 0:
                return {
                    "role": "assistant",
                    "content": json.dumps(
                        {"verdict": "APPROVE", "reason": "Known domestic LLC cleared without scan."}
                    ),
                }

            # 5. Forbidden wallet call when no wallet exists
            if tag == "fatf_jurisdiction_review" and not c.get("crypto_wallet") and len(executed) == 0:
                return _format_tool_response(
                    "screen_crypto_wallet",
                    {"wallet_address": "0x1111111111111111", "currency": "ETH"},
                    "call_unneeded_wallet",
                    tool_mode=tool_mode,
                )

        # Standard compliant next-tool progression
        if policy["abstain"]:
            cip_res = evaluate_cip(c)
            missing = cip_res.get("missing_fields", [])
            return {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "verdict": "REVIEW",
                        "reason": f"CIP incomplete ({', '.join(missing)}); abstained from screening tools.",
                        "edd_required": False,
                        "escalation_required": True,
                        "cited_tools": [],
                    }
                ),
            }

        executed_ok = {
            obs["tool"] for obs in observations if obs.get("success")
        }
        for req in policy["required_tools"]:
            if req in executed_ok:
                continue
            # Simulate incomplete_reasoning in baseline offshore_sow_review by skipping verify_source_of_wealth
            if not healed and tag == "offshore_sow_review" and req == "verify_source_of_wealth":
                continue

            args = _tool_args_for_case(req, c)
            return _format_tool_response(
                req,
                args,
                f"call_{req}_{len(executed)}",
                tool_mode=tool_mode,
            )

        # Terminal disposition
        if not healed:
            if tag == "weak_alias_review":
                return {
                    "role": "assistant",
                    "content": json.dumps({"verdict": "DENY", "reason": "Over-escalated weak alias to DENY."}),
                }
            if tag in {"pep_edd_review", "offshore_sow_review"}:
                return {
                    "role": "assistant",
                    "content": json.dumps({"verdict": "APPROVE", "reason": "Approved despite EDD flags."}),
                }

        # Derive healed verdict cleanly from observed SQLite tool outputs + FinCEN oracle rules
        oracle_out = evaluate_case_oracle(c, tool_observations=observations)
        verdict = oracle_out["verdict"]
        reasons = oracle_out.get("reason_codes") or [f"scenario={tag or 'standard'}"]
        return {
            "role": "assistant",
            "content": json.dumps(
                {
                    "verdict": verdict,
                    "reason": f"{c.get('customer_name')} ({country}) [{', '.join(reasons)}] -> {verdict}.",
                    "edd_required": bool(oracle_out.get("edd_required", False)),
                    "escalation_required": verdict != "APPROVE",
                    "cited_tools": sorted(executed_ok),
                    "reason_codes": reasons,
                }
            ),
        }

    return CallableChatAdapter(_fn)


def deterministic_teacher_synthesizer(prompt_history: list[dict[str, str]], cf: Any) -> list[str]:
    """Deterministic teacher synthesizer producing case-grounded poisoned + clean candidate triples."""
    del prompt_history
    ctx = cf.failure.context or {}
    c = ctx.get("case") or {}
    rc = cf.root_cause
    country = c.get("expected_country_iso2") or "US"
    policy = ctx.get("tool_policy") or compute_required_tools(c)
    prior_obs = ctx.get("tool_observations") or []
    ran_ok = {o.get("tool") for o in prior_obs if o.get("success", True)}

    def _cohere_call(tool_name: str, params: dict[str, Any]) -> str:
        return json.dumps([{"tool_name": tool_name, "parameters": params}], ensure_ascii=False)

    # Intentionally include an invalid candidate first to exercise ComplianceReplayValidator HARD REJECT
    if rc == "wrong_tool":
        if ctx.get("abstain_violation"):
            cip_res = evaluate_cip(c)
            missing = cip_res.get("missing_fields", [])
            bad = _cohere_call(
                "screen_sanctions",
                {"name": c.get("customer_name"), "entity_type": c.get("entity_type", "individual"), "country_iso2": "US"},
            )
            good = json.dumps(
                {
                    "verdict": "REVIEW",
                    "reason": f"Missing CIP fields ({', '.join(missing)}) for {c.get('customer_name')}; abstain and escalate to REVIEW.",
                    "edd_required": False,
                    "escalation_required": True,
                    "cited_tools": [],
                }
            )
            return [bad, good]

        if ctx.get("forbidden_called"):
            bad_wallet = _cohere_call(
                "screen_crypto_wallet",
                {"wallet_address": "0x1111111111111111", "currency": "ETH"},
            )
            next_req = next((t for t in policy.get("required_tools", []) if t not in ran_ok), "screen_sanctions")
            good_req = _cohere_call(next_req, _tool_args_for_case(next_req, c))
            return [bad_wallet, good_req]

        bad_iso = _cohere_call(
            "screen_sanctions",
            {"name": c.get("customer_name"), "entity_type": c.get("entity_type", "organization"), "country_iso2": "USA"},
        )
        good_iso = _cohere_call("screen_sanctions", _tool_args_for_case("screen_sanctions", c))
        return [bad_iso, good_iso]

    if rc == "loop_collapse":
        bad_repeat = _cohere_call("screen_sanctions", _tool_args_for_case("screen_sanctions", c))
        next_req = next((t for t in policy.get("required_tools", []) if t not in ran_ok), None)
        if next_req:
            good_next = _cohere_call(next_req, _tool_args_for_case(next_req, c))
            return [bad_repeat, good_next]
        oracle_out = evaluate_case_oracle(c, tool_observations=prior_obs)
        return [
            bad_repeat,
            json.dumps(
                {
                    "verdict": oracle_out["verdict"],
                    "reason": f"Completed all required checks for {c.get('customer_name')} ({country}) -> {oracle_out['verdict']}.",
                    "edd_required": bool(oracle_out.get("edd_required", False)),
                    "escalation_required": oracle_out["verdict"] != "APPROVE",
                    "cited_tools": sorted(ran_ok),
                }
            ),
        ]

    if rc == "incomplete_reasoning":
        missing_reqs = [t for t in policy.get("required_tools", []) if t not in ran_ok]
        target_req = missing_reqs[0] if missing_reqs else "verify_source_of_wealth"
        bad_unscreened = json.dumps({"verdict": "REVIEW", "reason": "Incomplete check."})
        good_req = _cohere_call(target_req, _tool_args_for_case(target_req, c))
        return [bad_unscreened, good_req]

    if rc == "hallucinated_output":
        unscreened_approve = json.dumps({"verdict": "APPROVE", "reason": "Clean without scan."})
        first_req = (policy.get("required_tools") or ["screen_sanctions"])[0]
        valid_tool = _cohere_call(first_req, _tool_args_for_case(first_req, c))
        return [unscreened_approve, valid_tool]

    # wrong_routing: compute genuine FinCEN/FATF/OFAC disposition from tool observations
    oracle_out = evaluate_case_oracle(c, tool_observations=prior_obs)
    expected_v = oracle_out["verdict"]
    reasons = oracle_out.get("reason_codes") or [c.get("scenario_tag") or "compliance_policy"]
    wrong_v = "APPROVE" if expected_v != "APPROVE" else "DENY"
    return [
        json.dumps({"verdict": wrong_v, "reason": f"Mismatched disposition for {c.get('customer_name')}."}),
        json.dumps(
            {
                "verdict": expected_v,
                "reason": f"FinCEN compliance analysis for {c.get('customer_name')} ({country}) [{', '.join(reasons)}] -> {expected_v}.",
                "edd_required": bool(oracle_out.get("edd_required", False)),
                "escalation_required": expected_v != "APPROVE",
                "cited_tools": sorted(ran_ok),
                "reason_codes": reasons,
            }
        ),
    ]


def _normalize_router_model(model: str, api_base: str | None) -> str:
    if api_base and "/" in model and not model.startswith("openai/"):
        return f"openai/{model}"
    return model


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description="Collect Split B trajectories and generate verified DPO pairs.")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "data" / "splits" / "split_b.jsonl")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--mock", action="store_true", help="Use offline deterministic CPU stub instead of HF API.")
    parser.add_argument("--mode", choices=["mock", "live"], default=None, help="Execution mode (mock or live).")
    parser.add_argument(
        "--student-model",
        type=str,
        default=None,
        help="Student model ID for rollout collection (defaults to config models.student_base).",
    )
    parser.add_argument("--dpo-out", type=Path, default=PROJECT_ROOT / "artifacts" / "dpo_pairs" / "dpo_train.jsonl")
    parser.add_argument("--runs-out", type=Path, default=PROJECT_ROOT / "artifacts" / "runs" / "split_b_runs.json")
    args = parser.parse_args()

    cfg = load_decide_pipeline_config()
    cases = load_jsonl_cases(args.split, limit=args.limit)
    use_mock = bool(args.mock or args.mode == "mock")
    tool_mode = str(
        cfg.get("models", {}).get("student_tool_call_mode")
        or cfg.get("models", {}).get("tool_mode")
        or "prompt"
    )
    student_model = args.student_model or cfg["models"]["student_base"]

    if use_mock:
        adapter = build_deterministic_student_stub(healed=False, tool_mode=tool_mode)
        synthesizer = deterministic_teacher_synthesizer
    else:
        api_base = cfg["models"]["teacher_api_base"]
        adapter = OpenAICompatibleChatAdapter(
            model=_normalize_router_model(student_model, api_base),
            api_base=api_base,
            enable_thinking=bool(cfg["models"].get("student_enable_thinking", False)),
            tool_mode=tool_mode,
        )
        synthesizer = None

    runner = KYCAgentRunner(adapter=adapter, config=cfg, tool_mode=tool_mode)
    runs = []
    for idx, c in enumerate(cases):
        print(f"[{idx+1}/{len(cases)}] Case {c.get('case_id')} ({c.get('scenario_tag')})...", flush=True)
        runs.append(runner.run_case(c))

    args.runs_out.parent.mkdir(parents=True, exist_ok=True)
    serializable = [{k: v for k, v in r.items() if k not in {"episode", "event_log"}} for r in runs]
    args.runs_out.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")

    heal_out = run_self_heal_cycle_sync(
        run_results=runs,
        db_path=runner.db_path,
        synthesizer=synthesizer,
        dpo_output_path=args.dpo_out,
        min_failures_to_trigger=cfg["closed_loop"]["total_failures_threshold"],
        failure_rate_threshold=cfg["closed_loop"]["failure_rate_threshold"],
    )
    print(
        json.dumps(
            {
                "mode": "offline_stub" if use_mock else "live_model",
                "student_model": "deterministic_stub" if use_mock else student_model,
                "tool_mode": tool_mode,
                "total_runs": heal_out["total_runs"],
                "detected_failures": len(heal_out["detected_failures"]),
                "project_heal_failures": len(heal_out.get("project_heal_failures", [])),
                "dpo_examples_validated": heal_out["buffer_size"],
                "trigger_decision": {
                    "should_trigger": heal_out["trigger_decision"]["should_trigger"],
                    "reason": heal_out["trigger_decision"]["reason"],
                },
                "diversity_report": heal_out["diversity_report"],
                "dpo_jsonl_path": heal_out["dpo_jsonl_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
