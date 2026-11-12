"""Phase 2/3 CLI: Evaluate baseline vs candidate student across Split C, reliability, & BFCL."""

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

from scripts.run_collect_and_heal import (
    _normalize_router_model,
    build_deterministic_student_stub,
    load_jsonl_cases,
)
from src.audit_eval import (
    bfcl_dicts_to_samples,
    compare_baseline_vs_healed,
    evaluate_bfcl_general_regression,
    evaluate_kyc_runs_with_auditkit,
)
from src.env_utils import load_project_env
from src.kyc_agent_runner import (
    KYCAgentRunner,
    OpenAICompatibleChatAdapter,
    load_decide_pipeline_config,
)
from src.patch_heal import evaluate_kyc_deployment_gate


def _build_offline_bfcl_predictors(samples_raw: list[dict[str, Any]]) -> tuple[Any, Any]:
    """Build deterministic baseline (name-drop/first-arg bug) vs healed Cohere prompt BFCL predictors."""
    samples = bfcl_dicts_to_samples(samples_raw)
    by_prompt_fragment: list[tuple[str, Any]] = []
    for s in samples:
        key_frag = s.input[:80] if s.input else (s.id or "")
        by_prompt_fragment.append((key_frag, s))

    def _pick_sample(prompt: Any) -> Any:
        text = json.dumps(prompt) if not isinstance(prompt, str) else prompt
        for frag, s in by_prompt_fragment:
            if frag and frag in text:
                return s
        return samples[0]

    def _pick_bfcl_value(alts: Any, corrupt: bool = False) -> Any:
        if isinstance(alts, list) and alts:
            chosen = next((v for v in alts if v != ""), alts[0])
        else:
            chosen = alts
        if isinstance(chosen, dict):
            return {dk: _pick_bfcl_value(dv, corrupt=corrupt) for dk, dv in chosen.items()}
        if corrupt and isinstance(chosen, str) and len(chosen) > 2:
            return "INVALID_ARG"
        if corrupt and isinstance(chosen, (int, float)) and not isinstance(chosen, bool):
            return -999999
        return chosen

    def _emit_cohere_from_ref(ref_calls: list[dict[str, Any]], corrupt: bool = False) -> str:
        out: list[dict[str, Any]] = []
        for idx, tc in enumerate(ref_calls):
            if corrupt and idx > 0:
                continue
            args = {}
            raw_args = tc.get("arguments") or {}
            if isinstance(raw_args, dict):
                for k, alts in raw_args.items():
                    args[k] = _pick_bfcl_value(alts, corrupt=corrupt)
            out.append({"tool_name": tc.get("name"), "parameters": args})
        return json.dumps(out, ensure_ascii=False)

    def _baseline_fn(prompts: list[Any]) -> list[str]:
        replies: list[str] = []
        for idx, p in enumerate(prompts):
            s = _pick_sample(p)
            turns = s.expected_tool_calls or []
            first_turn = turns[0] if turns and isinstance(turns[0], list) else []
            # Simulate 50% broken tool arg / omitted call in baseline BFCL prediction
            replies.append(_emit_cohere_from_ref(first_turn, corrupt=(idx % 2 == 0)))
        return replies

    def _healed_fn(prompts: list[Any]) -> list[str]:
        replies: list[str] = []
        for p in prompts:
            s = _pick_sample(p)
            turns = s.expected_tool_calls or []
            first_turn = turns[0] if turns and isinstance(turns[0], list) else []
            replies.append(_emit_cohere_from_ref(first_turn, corrupt=False))
        return replies

    return _baseline_fn, _healed_fn


def _make_live_bfcl_predictor(adapter: OpenAICompatibleChatAdapter) -> Any:
    """Wrap an OpenAICompatibleChatAdapter for AuditKIT evaluate_bfcl_general_regression."""

    def _predict(prompts: list[Any]) -> list[str]:
        replies: list[str] = []
        for p in prompts:
            messages = p if isinstance(p, list) else [{"role": "user", "content": str(p)}]
            resp = adapter.chat(messages)
            content = str(resp.get("content") or "").strip()
            if not content and resp.get("tool_calls"):
                calls = [
                    {
                        "tool_name": tc.get("function", {}).get("name"),
                        "parameters": json.loads(tc.get("function", {}).get("arguments") or "{}"),
                    }
                    for tc in resp["tool_calls"]
                ]
                content = json.dumps(calls, ensure_ascii=False)
            replies.append(content)
        return replies

    return _predict


def _serialize_bfcl_result(bfcl_res: dict[str, Any]) -> dict[str, Any]:
    """Serialize BFCL regression summary metrics into JSON-safe dicts."""
    bfcl_summary: dict[str, Any] = {}
    for k, v in bfcl_res.items():
        if k in {"baseline_run", "candidate_run", "comparison"}:
            continue
        if k in {"baseline_stats", "candidate_stats"} and isinstance(v, dict):
            bfcl_summary[k] = {
                mk: (
                    mv.to_dict()
                    if hasattr(mv, "to_dict")
                    else (dict(vars(mv)) if hasattr(mv, "__dict__") else mv)
                )
                for mk, mv in v.items()
            }
        else:
            bfcl_summary[k] = v
    return bfcl_summary


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description="Evaluate baseline vs candidate AutoHeal-KYC on Split C & BFCL.")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "data" / "splits" / "split_c.jsonl")
    parser.add_argument("--bfcl", type=Path, default=PROJECT_ROOT / "data" / "splits" / "bfcl_subset.json")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--trials", type=int, default=1, help="Number of independent evaluation trials (e.g. 3 for pass^3).")
    parser.add_argument("--mock", action="store_true", help="Run offline CPU baseline vs healed stub.")
    parser.add_argument("--baseline-model", type=str, default=None)
    parser.add_argument("--candidate-model", type=str, default=None)
    parser.add_argument("--curated-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "curated")
    parser.add_argument("--sidecar-out", type=Path, default=PROJECT_ROOT / "artifacts" / "runs" / "eval_sidecar.jsonl")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "runs" / "eval_report.json")
    args = parser.parse_args()

    if not args.mock and not args.candidate_model:
        parser.error("--candidate-model is required unless --mock is specified.")

    cfg = load_decide_pipeline_config()
    cases = load_jsonl_cases(args.split, limit=args.limit)
    tool_mode = str(
        cfg.get("models", {}).get("student_tool_call_mode")
        or cfg.get("models", {}).get("tool_mode")
        or "prompt"
    )
    n_trials = max(1, int(args.trials))

    if args.mock:
        base_adapter = build_deterministic_student_stub(healed=False, tool_mode=tool_mode)
        cand_adapter = build_deterministic_student_stub(healed=True, tool_mode=tool_mode)
        base_model_label = "deterministic_stub_baseline"
        cand_model_label = "deterministic_stub_healed"
    else:
        api_base = cfg["models"]["teacher_api_base"]
        base_model_label = args.baseline_model or cfg["models"]["student_base"]
        cand_model_label = str(args.candidate_model)
        base_adapter = OpenAICompatibleChatAdapter(
            model=_normalize_router_model(base_model_label, api_base),
            api_base=api_base,
            enable_thinking=bool(cfg["models"].get("student_enable_thinking", False)),
            tool_mode=tool_mode,
        )
        cand_adapter = OpenAICompatibleChatAdapter(
            model=_normalize_router_model(cand_model_label, api_base),
            api_base=api_base,
            enable_thinking=bool(cfg["models"].get("student_enable_thinking", False)),
            tool_mode=tool_mode,
        )

    base_runner = KYCAgentRunner(adapter=base_adapter, config=cfg, tool_mode=tool_mode)
    cand_runner = KYCAgentRunner(adapter=cand_adapter, config=cfg, tool_mode=tool_mode)

    base_trials: list[list[dict[str, Any]]] = []
    cand_trials: list[list[dict[str, Any]]] = []
    for _ in range(n_trials):
        base_trials.append([base_runner.run_case(c) for c in cases])
        cand_trials.append([cand_runner.run_case(c) for c in cases])

    base_runs = base_trials[0]
    cand_runs = cand_trials[0]

    base_eval = evaluate_kyc_runs_with_auditkit(
        base_runs,
        name="baseline",
        curated_dir=args.curated_dir,
        trial_runs=base_trials if n_trials >= 2 else None,
        reset_confirmed=True if n_trials >= 2 else False,
    )
    cand_eval = evaluate_kyc_runs_with_auditkit(
        cand_runs,
        name="healed",
        curated_dir=args.curated_dir,
        sidecar_path=args.sidecar_out,
        trial_runs=cand_trials if n_trials >= 2 else None,
        reset_confirmed=True if n_trials >= 2 else False,
    )
    comparison = compare_baseline_vs_healed(base_eval, cand_eval)
    gate = evaluate_kyc_deployment_gate(
        baseline_runs=base_runs,
        candidate_runs=cand_runs,
        min_samples=min(len(cases), cfg["closed_loop"]["min_samples_for_gate"]),
        min_improvement=cfg["closed_loop"]["min_improvement"],
    )

    bfcl_summary: dict[str, Any] | None = None
    if args.bfcl.exists():
        bfcl_raw = json.loads(args.bfcl.read_text(encoding="utf-8"))
        if args.mock:
            b_pred, h_pred = _build_offline_bfcl_predictors(bfcl_raw)
        else:
            b_pred = _make_live_bfcl_predictor(base_adapter)
            h_pred = _make_live_bfcl_predictor(cand_adapter)
        bfcl_res = evaluate_bfcl_general_regression(
            bfcl_raw,
            baseline_model=b_pred,
            candidate_model=h_pred,
            tool_mode=tool_mode,
        )
        bfcl_summary = _serialize_bfcl_result(bfcl_res)

    report: dict[str, Any] = {
        "mode": "offline_stub" if args.mock else "live_model",
        "tool_mode": tool_mode,
        "baseline_model": base_model_label,
        "candidate_model": cand_model_label,
        "split": str(args.split),
        "num_cases": len(cases),
        "trials": n_trials,
        "baseline_kpis": {k: v for k, v in base_eval["kpis"].items() if k != "per_case"},
        "healed_kpis": {k: v for k, v in cand_eval["kpis"].items() if k != "per_case"},
        "auditkit_summary": {
            "baseline": base_eval["auditkit_summary"],
            "healed": cand_eval["auditkit_summary"],
        },
        "custom_predicates": {
            "baseline": base_eval["custom_predicates"]["predicate_rates"],
            "healed": cand_eval["custom_predicates"]["predicate_rates"],
        },
        "reliability": {
            "baseline": {k: v for k, v in base_eval["reliability"].items() if k != "per_case"},
            "healed": {k: v for k, v in cand_eval["reliability"].items() if k != "per_case"},
        },
        "perf": {
            "baseline": base_eval["perf"],
            "healed": cand_eval["perf"],
        },
        "curatorkit_evidence": {
            "manifest_loaded": cand_eval["curatorkit_evidence"]["manifest_loaded"],
            "joined_episodes": cand_eval["curatorkit_evidence"]["joined_episodes"],
            "curated_train_case_ids_count": cand_eval["curatorkit_evidence"].get("curated_train_case_ids_count", 0),
            "split_zero_leakage_from_curated": cand_eval["curatorkit_evidence"].get("split_zero_leakage_from_curated"),
        },
        "sidecar_path": cand_eval["sidecar_path"],
        "paired_bootstrap": comparison["bootstrap"],
        "bfcl_general_regression": bfcl_summary,
        "deployment_gate": {
            "approved": gate["approved"],
            "reasons": gate["reasons"],
            "task_delta": gate["gate_decision"].task_delta,
            "trajectory_delta": gate["gate_decision"].trajectory_delta,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
