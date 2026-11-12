"""AuditKIT evaluation harness, compliance metrics, reliability, bootstrap tests, and BFCL evaluator.

Uses AuditKIT v1.1.6's AgentCase / AgentEpisode / AgentEvalRunner, FinalStateAssertion,
CustomPredicate, JsonSchemaValid, LatencyStats / Throughput, reliability (pass@k / pass^k),
curatorkit_manifest / curatorkit_data_sample / join_evidence / write_sidecar,
auditkit._bootstrap.paired_bootstrap, and BFCL ToolCallAdapter(mode="prompt") + ToolCallF1.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from auditkit import compare as ak_compare
from auditkit import evaluate as ak_evaluate
from auditkit import load_bfcl
from auditkit.runner import RunConfig
from auditkit._bootstrap import paired_bootstrap, score_pairs
from auditkit.adapter import ToolCallAdapter
from auditkit.agent_eval import (
    FAILURE,
    SUCCESS,
    AgentCase,
    AgentEpisode,
    AgentEvalRunner,
    AgentEvalSpec,
    CustomPredicate,
    FinalStateAssertion,
    aggregate_verdict,
    curatorkit_data_sample,
    curatorkit_manifest,
    join_evidence,
    reliability,
    write_sidecar,
)
from auditkit.bfcl import bfcl_arg_match
from auditkit.metrics.agent import ToolCallF1
from auditkit.metrics.perf import LatencyStats, Throughput
from auditkit.metrics.structured import JsonSchemaValid
from auditkit.sample import Sample

from src.kyc_oracle import ALL_TOOL_NAMES, compute_required_tools, evaluate_case_oracle

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OFFLINE_SCORERS: tuple[str, ...] = (
    "agent_loop_detection",
    "tool_permission",
    "redundant_tool_calls",
    "tool_call_validity",
    "tool_call_f1",
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

_VERDICT_SCHEMA_SCORER = JsonSchemaValid(
    schema=TERMINAL_VERDICT_JSON_SCHEMA,
    backend="auto",
    strip_fences=True,
)


def _episode_called_tools(episode: AgentEpisode) -> list[tuple[str, dict[str, Any], bool]]:
    """Extract (tool_name, arguments, ok) tuples observed inside an AgentEpisode trace."""
    calls: list[tuple[str, dict[str, Any], bool]] = []
    results_by_id: dict[str, dict[str, Any]] = {}
    for ev in getattr(episode, "events", []) or []:
        if getattr(ev, "type", None) == "tool_result" and getattr(ev, "call_id", None):
            payload = getattr(ev, "payload", None)
            payload_dict = payload if isinstance(payload, dict) else {}
            results_by_id[str(ev.call_id)] = {
                "error": payload_dict.get("error"),
                "output": payload_dict.get("output"),
                "success": payload_dict.get("success", True),
            }
    for ev in getattr(episode, "events", []) or []:
        if getattr(ev, "type", None) != "tool_call":
            continue
        payload = getattr(ev, "payload", None)
        if not isinstance(payload, dict) or not payload:
            continue
        fn = payload.get("function") if isinstance(payload.get("function"), dict) else None
        raw_name = fn.get("name") if fn is not None else payload.get("name")
        if not raw_name:
            continue
        t_name = str(raw_name)
        raw_args = fn.get("arguments") if fn is not None else payload.get("arguments")
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
                t_args = parsed_args if isinstance(parsed_args, dict) else {}
            except Exception:
                t_args = {}
        elif isinstance(raw_args, dict):
            t_args = raw_args
        else:
            t_args = {}
        call_id = str(getattr(ev, "call_id", None) or payload.get("id") or "")
        res = results_by_id.get(call_id, {})
        ok = (res.get("error") is None) and (res.get("success", True) is not False)
        calls.append((t_name, t_args, ok))
    return calls


def _check_cip_abstention(episode: AgentEpisode, case: AgentCase | None = None) -> str | bool:
    abstain_expected = bool((case.metadata if case else {}).get("abstain_expected", False))
    if not abstain_expected:
        return True
    called = _episode_called_tools(episode)
    verdict = (getattr(episode, "final_state", None) or {}).get("verdict")
    return len(called) == 0 and verdict == "REVIEW"


def _check_loop_free(episode: AgentEpisode, case: AgentCase | None = None) -> str | bool:
    del case
    state = getattr(episode, "final_state", None) or {}
    if state.get("loop_collapse"):
        return False
    counts: dict[str, int] = {}
    for t_name, t_args, _ in _episode_called_tools(episode):
        key = f"{t_name}:{json.dumps(t_args, sort_keys=True, default=str)}"
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= 3:
            return False
    return True


def _check_required_tools(episode: AgentEpisode, case: AgentCase | None = None) -> str | bool:
    meta = case.metadata if case else {}
    if meta.get("abstain_expected"):
        return True
    required = set(meta.get("required_tools") or [])
    if not required:
        return True
    called_ok = {t_name for t_name, _, ok in _episode_called_tools(episode) if ok}
    return required.issubset(called_ok)


CIP_ABSTENTION_PREDICATE = CustomPredicate(
    _check_cip_abstention,
    version="kyc_v1",
    name="cip_abstention_invariant",
)
LOOP_FREE_PREDICATE = CustomPredicate(
    _check_loop_free,
    version="kyc_v1",
    name="loop_free_invariant",
)
REQUIRED_TOOLS_PREDICATE = CustomPredicate(
    _check_required_tools,
    version="kyc_v1",
    name="required_tool_coverage_invariant",
)
KYC_CUSTOM_PREDICATES: tuple[CustomPredicate, ...] = (
    CIP_ABSTENTION_PREDICATE,
    LOOP_FREE_PREDICATE,
    REQUIRED_TOOLS_PREDICATE,
)


def _reference_turns_for_case(case: dict[str, Any], oracle_verdict: str) -> list[dict[str, Any]]:
    """Build deterministic reference turns for AuditKIT trajectory/tool-F1 scorers."""
    policy = compute_required_tools(case)
    turns: list[dict[str, Any]] = []
    name = case.get("customer_name") or "Unknown"
    entity_type = case.get("entity_type") or "individual"
    country = case.get("expected_country_iso2") or case.get("country_iso2") or "US"
    dob = case.get("date_of_birth")

    for t_name in policy["required_tools"]:
        if t_name == "screen_sanctions":
            args = {"name": name, "entity_type": entity_type, "country_iso2": country, "date_of_birth": dob}
        elif t_name == "screen_export_control":
            args = {"entity_name": name, "country_iso2": country}
        elif t_name == "screen_pep_adverse_media":
            args = {"name": name, "country_iso2": country, "date_of_birth": dob}
        elif t_name == "check_fatf_jurisdiction_risk":
            args = {"country_iso2": country}
        elif t_name == "screen_crypto_wallet":
            args = {
                "wallet_address": case.get("crypto_wallet") or "",
                "currency": case.get("crypto_currency") or "ETH",
            }
        elif t_name == "verify_source_of_wealth":
            args = {
                "customer_name": name,
                "annual_income_usd": float(case.get("annual_income_usd") or 0.0),
                "declared_assets_usd": float(case.get("declared_assets_usd") or 0.0),
                "expected_monthly_tx_usd": float(case.get("expected_monthly_tx_usd") or 0.0),
                "source_of_wealth": case.get("source_of_wealth") or "salary",
                "offshore_jurisdiction": case.get("offshore_jurisdiction"),
            }
        else:
            args = {}
        turns.append({"role": "assistant", "tool_calls": [{"name": t_name, "arguments": args}]})

    turns.append({"role": "assistant", "content": json.dumps({"verdict": oracle_verdict})})
    return turns


def build_agent_cases(run_results: list[dict[str, Any]]) -> list[AgentCase]:
    """Convert KYCAgentRunner outputs into validated AuditKIT AgentCase specs."""
    cases: list[AgentCase] = []
    for r in run_results:
        case = r["case"]
        policy = r.get("tool_policy") or compute_required_tools(case)
        expected_verdict = str(
            r.get("oracle_verdict")
            or case.get("expected_verdict")
            or evaluate_case_oracle(case)["verdict"]
        )
        forbidden = set(policy.get("forbidden_tools", []))
        allowed = [t for t in ALL_TOOL_NAMES if t not in forbidden]
        cases.append(
            AgentCase(
                id=str(r["case_id"]),
                task=f"Screen KYC onboarding dossier for {case.get('customer_name') or r['case_id']}",
                category=str(case.get("scenario_tag") or case.get("entity_type") or "kyc"),
                allowed_tools=allowed,
                outcome=FinalStateAssertion(key="verdict", equals=expected_verdict),
                reference_turns=_reference_turns_for_case(case, expected_verdict),
                metadata={
                    "oracle_verdict": expected_verdict,
                    "abstain_expected": bool(policy.get("abstain")),
                    "required_tools": list(policy.get("required_tools", [])),
                    "forbidden_tools": list(policy.get("forbidden_tools", [])),
                },
            )
        )
    return cases


def compute_compliance_kpis(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact domain KPIs, JsonSchemaValid checks, and per-case pass indicators."""
    n = len(run_results)
    if n == 0:
        return {"num_cases": 0, "verdict_accuracy": 0.0, "per_case": {}}

    verdict_ok = 0
    first_try_ok = 0
    loop_collapse = 0
    dangling = 0
    hallucinated_approve = 0
    schema_valid_count = 0
    abstain_total = 0
    abstain_ok = 0
    total_tool_calls = 0

    per_case: dict[str, dict[str, float]] = {}

    for r in run_results:
        cid = str(r["case_id"])
        policy = r.get("tool_policy") or compute_required_tools(r["case"])
        executed_valid = {tc["name"] for tc in r.get("tool_calls", []) if tc.get("success")}
        is_verdict_ok = r.get("verdict") == r.get("oracle_verdict")
        is_first_try_ok = (len(r.get("first_try_tool_errors", [])) == 0) and not any(
            tc["name"] in set(policy.get("forbidden_tools", []))
            for tc in r.get("tool_calls", [])
        )
        is_loop = bool(r.get("loop_collapse_detected"))
        is_dangling = len(r.get("dangling_tool_calls", [])) > 0
        is_hallucinated = (
            not policy.get("abstain")
            and r.get("verdict") == "APPROVE"
            and (
                "screen_sanctions" not in executed_valid
                or "screen_pep_adverse_media" not in executed_valid
            )
        )

        # Score terminal assistant text with AuditKIT JsonSchemaValid
        ep = r.get("episode")
        final_text = getattr(ep, "final_answer", None) if ep is not None else None
        if not final_text:
            for msg in reversed(r.get("raw_messages") or r.get("messages") or []):
                if msg.get("role") == "assistant" and msg.get("content"):
                    final_text = str(msg["content"])
                    break
        if not final_text:
            final_text = json.dumps({"verdict": r.get("verdict"), "reason": r.get("reason") or "completed"})
        schema_score = _VERDICT_SCHEMA_SCORER.score(Sample(input="", id=cid), str(final_text))
        is_schema_valid = schema_score.value >= 1.0

        if policy.get("abstain"):
            abstain_total += 1
            if len(r.get("tool_calls", [])) == 0 and r.get("verdict") == "REVIEW":
                abstain_ok += 1

        verdict_ok += int(is_verdict_ok)
        first_try_ok += int(is_first_try_ok)
        loop_collapse += int(is_loop)
        dangling += int(is_dangling)
        hallucinated_approve += int(is_hallucinated)
        schema_valid_count += int(is_schema_valid)
        total_tool_calls += len(r.get("tool_calls", []))

        per_case[cid] = {
            "verdict_pass": 1.0 if is_verdict_ok else 0.0,
            "first_try_tool_pass": 1.0 if is_first_try_ok else 0.0,
            "no_loop": 0.0 if is_loop else 1.0,
            "schema_valid": 1.0 if is_schema_valid else 0.0,
            "composite_pass": 1.0 if (is_verdict_ok and is_first_try_ok and not is_loop) else 0.0,
        }

    return {
        "num_cases": n,
        "verdict_accuracy": round(verdict_ok / n, 4),
        "first_try_tool_accuracy": round(first_try_ok / n, 4),
        "loop_collapse_rate": round(loop_collapse / n, 4),
        "dangling_tool_rate": round(dangling / n, 4),
        "hallucination_rate": round(hallucinated_approve / n, 4),
        "terminal_json_schema_rate": round(schema_valid_count / n, 4),
        "abstain_accuracy": round(abstain_ok / abstain_total, 4) if abstain_total else None,
        "mean_tool_calls": round(total_tool_calls / n, 4),
        "per_case": per_case,
    }


def _evaluate_custom_predicates(
    episodes: list[AgentEpisode],
    cases: list[AgentCase],
) -> dict[str, Any]:
    per_predicate_pass: dict[str, int] = {pred.name: 0 for pred in KYC_CUSTOM_PREDICATES}
    per_case_predicates: dict[str, dict[str, str]] = {}
    all_passed = 0
    for ep, case in zip(episodes, cases):
        row_res: dict[str, str] = {}
        case_ok = True
        for pred in KYC_CUSTOM_PREDICATES:
            outcome = pred.evaluate(ep, case)
            row_res[pred.name] = outcome.verdict
            if outcome.verdict == SUCCESS:
                per_predicate_pass[pred.name] += 1
            else:
                case_ok = False
        per_case_predicates[str(case.id)] = row_res
        all_passed += int(case_ok)

    total = max(len(cases), 1)
    return {
        "num_cases": len(cases),
        "all_invariants_rate": round(all_passed / total, 4),
        "predicate_rates": {k: round(v / total, 4) for k, v in per_predicate_pass.items()},
        "per_case": per_case_predicates,
    }


def _compute_perf_stats(run_results: list[dict[str, Any]], episodes: list[AgentEpisode]) -> dict[str, Any]:
    lat = LatencyStats()
    tp = Throughput()
    for r, ep in zip(run_results, episodes):
        ms = float((getattr(ep, "final_state", None) or {}).get("latency_ms") or r.get("latency_ms") or 0.0)
        lat.record(ms)
        tp.record(ms, num_requests=1)
    return {
        "concurrency": 1,
        "latency_ms": lat.stats(),
        "throughput": tp.stats(),
    }


def join_curatorkit_evidence(
    episodes: list[AgentEpisode],
    curated_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Attach CuratorKIT manifest & per-row lineage sidecar evidence via AuditKIT lexsi importers."""
    cdir = Path(curated_dir or (PROJECT_ROOT / "artifacts" / "curated"))
    summary: dict[str, Any] = {
        "curated_dir": str(cdir),
        "manifest_loaded": False,
        "manifest_evidence": None,
        "joined_episodes": 0,
        "join_errors": [],
    }
    manifest_path = cdir / "manifest.json"
    prov_path = cdir / "lexsi_provenance.json"
    sidecar_path = cdir / "auditkit_lineage_sidecar.json"

    provenance: dict[str, Any] | None = None
    if prov_path.exists():
        provenance = json.loads(prov_path.read_text(encoding="utf-8"))

    if manifest_path.exists():
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_ev = curatorkit_manifest(manifest_payload, provenance=provenance)
        summary["manifest_loaded"] = True
        summary["manifest_evidence"] = manifest_ev.to_dict()

    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        by_case_id: dict[str, tuple[str, dict[str, Any]]] = {}
        for row_id, entry in sidecar.items():
            cid = str((entry.get("metadata") or {}).get("case_id") or "")
            if cid and not entry.get("rejected"):
                by_case_id.setdefault(cid, (str(row_id), entry))

        summary["curated_train_case_ids_count"] = len(by_case_id)
        for ep in episodes:
            ep_cid = str(getattr(ep, "case_id", "") or "")
            match = by_case_id.get(ep_cid)
            if not match:
                continue
            row_id, entry = match
            sample_payload = {
                "id": row_id,
                "source_uri": entry.get("source_uri"),
                "task_type": entry.get("task_type"),
                "metadata": entry.get("metadata") or {},
                "provenance_chain": entry.get("provenance_chain") or [],
            }
            sample_ev = curatorkit_data_sample(sample_payload, provenance=provenance)
            jres = join_evidence(ep, sample_ev, by={"row_id": row_id})
            if jres.ok:
                summary["joined_episodes"] += 1
            else:
                summary["join_errors"].append(jres.reason)
        summary["split_zero_leakage_from_curated"] = summary["joined_episodes"] == 0

    return summary


def compute_multi_trial_reliability(
    trial_runs: Sequence[Sequence[dict[str, Any]]],
    *,
    reset_confirmed: bool = False,
) -> dict[str, Any]:
    """Compute AuditKIT multi-trial reliability (pass@k / pass^k / aggregate_verdict) per case."""
    if not trial_runs:
        return {"n_trials": 0, "status": "unknown", "reason": "no trials provided", "per_case": {}}

    k_trials = len(trial_runs)
    by_case: dict[str, list[tuple[str, str]]] = {}
    for t_idx, runs in enumerate(trial_runs):
        for r in runs:
            cid = str(r["case_id"])
            policy = r.get("tool_policy") or compute_required_tools(r["case"])
            executed_valid = {tc["name"] for tc in r.get("tool_calls", []) if tc.get("success")}
            is_verdict_ok = r.get("verdict") == r.get("oracle_verdict")
            is_first_try_ok = (len(r.get("first_try_tool_errors", [])) == 0) and not any(
                tc["name"] in set(policy.get("forbidden_tools", []))
                for tc in r.get("tool_calls", [])
            )
            is_loop = bool(r.get("loop_collapse_detected"))
            passed = is_verdict_ok and is_first_try_ok and not is_loop and (
                policy.get("abstain") or set(policy.get("required_tools", [])).issubset(executed_valid)
            )
            trial_id = str(r.get("trajectory_id") or f"{cid}_t{t_idx}")
            by_case.setdefault(cid, []).append((SUCCESS if passed else FAILURE, trial_id))

    per_case: dict[str, Any] = {}
    pass_at_k_vals: list[float] = []
    pass_hat_k_vals: list[float] = []
    agg_success = 0
    for cid, items in sorted(by_case.items()):
        verdicts = [v for v, _ in items]
        trial_ids = [tid for _, tid in items]
        rel = reliability(
            verdicts,
            k=k_trials,
            independent=reset_confirmed,
            trial_ids=trial_ids,
            n_requested=k_trials,
        )
        agg_v, agg_reason = aggregate_verdict(verdicts)
        rel["aggregate_verdict"] = agg_v
        rel["aggregate_reason"] = agg_reason
        per_case[cid] = rel
        if isinstance(rel.get("pass_at_k"), (int, float)):
            pass_at_k_vals.append(float(rel["pass_at_k"]))
        hat_val = rel.get("all_k") if rel.get("all_k") is not None else rel.get("pass_hat_k")
        if isinstance(hat_val, (int, float)):
            pass_hat_k_vals.append(float(hat_val))
        if agg_v == SUCCESS:
            agg_success += 1

    n_cases = max(len(by_case), 1)
    return {
        "n_trials": k_trials,
        "independent": bool(reset_confirmed),
        "status": "evaluated" if (pass_at_k_vals and k_trials >= 2) else "unknown",
        "num_cases": len(by_case),
        "aggregate_success_rate": round(agg_success / n_cases, 4),
        f"mean_pass_at_{k_trials}": round(sum(pass_at_k_vals) / len(pass_at_k_vals), 4)
        if pass_at_k_vals
        else None,
        f"mean_pass_hat_{k_trials}": round(sum(pass_hat_k_vals) / len(pass_hat_k_vals), 4)
        if pass_hat_k_vals
        else None,
        "pass_at_k_mean": round(sum(pass_at_k_vals) / len(pass_at_k_vals), 4)
        if pass_at_k_vals
        else None,
        "pass_hat_k_mean": round(sum(pass_hat_k_vals) / len(pass_hat_k_vals), 4)
        if pass_hat_k_vals
        else None,
        "per_case": per_case,
    }


def evaluate_kyc_runs_with_auditkit(
    run_results: list[dict[str, Any]],
    scorers: list[str] | None = None,
    judge: Any = None,
    name: str = "autoheal-kyc-eval",
    *,
    sidecar_path: str | Path | None = None,
    curated_dir: str | Path | None = None,
    trial_runs: Sequence[Sequence[dict[str, Any]]] | None = None,
    reset_confirmed: bool = False,
) -> dict[str, Any]:
    """Run AuditKIT AgentEvalRunner, CustomPredicates, CuratorKIT evidence join, perf, & sidecar."""
    cases = build_agent_cases(run_results)
    episodes: list[AgentEpisode] = [r["episode"] for r in run_results]
    curatorkit_evidence = join_curatorkit_evidence(episodes, curated_dir=curated_dir)

    chosen_scorers = list(scorers or DEFAULT_OFFLINE_SCORERS)
    if judge is not None and "task_completion" not in chosen_scorers:
        chosen_scorers.append("task_completion")

    runner = AgentEvalRunner()
    spec = AgentEvalSpec(
        cases=cases,
        mode="recorded",
        episodes=episodes,
        scorers=chosen_scorers,
        judge=judge,
        name=name,
    )
    auditkit_result = runner.run(spec)
    kpis = compute_compliance_kpis(run_results)
    custom_predicates = _evaluate_custom_predicates(episodes, cases)
    perf = _compute_perf_stats(run_results, episodes)

    if trial_runs is not None:
        reliability_summary = compute_multi_trial_reliability(trial_runs, reset_confirmed=reset_confirmed)
    else:
        single_rel = reliability(
            [
                SUCCESS if kpis["per_case"][str(r["case_id"])]["composite_pass"] >= 1.0 else FAILURE
                for r in run_results[:1]
            ],
            independent=reset_confirmed,
        )
        reliability_summary = {
            "n_trials": 1,
            "independent": bool(reset_confirmed),
            "status": single_rel.get("status", "unknown"),
            "reason": single_rel.get("reason", "single trial; pass@k/pass^k require >=2 trials"),
            "flags": single_rel.get("flags", []),
            "pass_at_k_mean": None,
            "pass_hat_k_mean": None,
        }

    emitted_sidecar: str | None = None
    if sidecar_path is not None:
        sp = Path(sidecar_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        write_sidecar(auditkit_result, str(sp))
        emitted_sidecar = str(sp)

    return {
        "name": name,
        "kpis": kpis,
        "custom_predicates": custom_predicates,
        "perf": perf,
        "reliability": reliability_summary,
        "curatorkit_evidence": curatorkit_evidence,
        "sidecar_path": emitted_sidecar,
        "auditkit_result": auditkit_result,
        "auditkit_summary": (
            auditkit_result.summary()
            if callable(getattr(auditkit_result, "summary", None))
            else getattr(auditkit_result, "summary", None)
        ),
        "auditkit_rows_count": len(getattr(auditkit_result, "rows", []) or []),
        "auditkit_outcomes": {
            verdict: sum(
                1
                for row in (getattr(auditkit_result, "rows", []) or [])
                if (row.outcome or {}).get("verdict") == verdict
            )
            for verdict in ("success", "failure", "unknown")
        },
        "eligibility": {
            "ineligible_rows": sum(
                1 for row in (getattr(auditkit_result, "rows", []) or []) if getattr(row, "ineligible", None)
            )
        },
    }


def compare_baseline_vs_healed(
    baseline_eval: dict[str, Any],
    healed_eval: dict[str, Any],
    seed: int = 42,
    n_resamples: int = 1000,
) -> dict[str, Any]:
    """Compute paired bootstrap significance across baseline vs healed KYC evaluations."""
    b_cases = baseline_eval["kpis"]["per_case"]
    h_cases = healed_eval["kpis"]["per_case"]
    shared_ids = sorted(cid for cid in b_cases if cid in h_cases)

    if not shared_ids:
        return {"shared_cases": 0, "metrics": {}}

    metrics_out: dict[str, Any] = {}
    for metric_key in ("verdict_pass", "first_try_tool_pass", "no_loop", "schema_valid", "composite_pass"):
        pairs = [(b_cases[cid][metric_key], h_cases[cid][metric_key]) for cid in shared_ids]
        metrics_out[metric_key] = paired_bootstrap(pairs, n_resamples=n_resamples, seed=seed)

    return {
        "shared_cases": len(shared_ids),
        "baseline_kpis": {k: v for k, v in baseline_eval["kpis"].items() if k != "per_case"},
        "healed_kpis": {k: v for k, v in healed_eval["kpis"].items() if k != "per_case"},
        "bootstrap": metrics_out,
    }


def load_bfcl_compliance_subset(
    output_path: str | Path | None = None,
    limit: int = 25,
    revision: str = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
    local_dir: str | Path | None = None,
    categories: tuple[str, ...] = ("simple", "multiple"),
) -> list[dict[str, Any]]:
    """Download/load pinned BFCL single-turn categories with possible_answer references."""
    if local_dir is None:
        from huggingface_hub import snapshot_download

        patterns: list[str] = []
        for cat in categories:
            patterns.append(f"BFCL_v3_{cat}.json")
            patterns.append(f"possible_answer/BFCL_v3_{cat}.json")
        snap_path = snapshot_download(
            "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
        )
        bfcl_root = Path(snap_path)
    else:
        bfcl_root = Path(local_dir)

    all_samples: list[Any] = []
    per_cat_limit = max(1, (limit + len(categories) - 1) // max(len(categories), 1))
    for cat in categories:
        q_file = bfcl_root / f"BFCL_v3_{cat}.json"
        ans_file = bfcl_root / "possible_answer" / f"BFCL_v3_{cat}.json"
        if not q_file.exists():
            matches = [
                p for p in bfcl_root.rglob(f"BFCL_v3_{cat}.json")
                if "possible_answer" not in str(p)
            ]
            if not matches:
                continue
            q_file = matches[0]
        if not ans_file.exists():
            ans_matches = list(bfcl_root.rglob(f"possible_answer/*BFCL_v3_{cat}.json"))
            ans_file = ans_matches[0] if ans_matches else None
        if ans_file is None or not ans_file.exists():
            raise FileNotFoundError(f"Missing BFCL possible_answer file for category '{cat}' under {bfcl_root}")

        loaded = load_bfcl(
            str(q_file),
            answers=str(ans_file),
            revision=revision,
            limit=per_cat_limit,
        )
        all_samples.extend(loaded if isinstance(loaded, list) else list(loaded))

    if not all_samples:
        raise FileNotFoundError(f"No BFCL {categories} files found under {bfcl_root}")

    serialized: list[dict[str, Any]] = []
    for idx, item in enumerate(all_samples[:limit]):
        if isinstance(item, dict):
            serialized.append(item)
        elif hasattr(item, "__dict__"):
            row = {k: v for k, v in vars(item).items() if not k.startswith("_")}
            row["kind"] = str(getattr(item.kind, "value", item.kind))
            serialized.append(row)
        else:
            serialized.append({"id": f"bfcl_{idx}", "raw": str(item)})

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(serialized, indent=2, default=str), encoding="utf-8")

    return serialized


def bfcl_dicts_to_samples(items: list[dict[str, Any] | Sample]) -> list[Sample]:
    """Convert serialized bfcl_subset.json dicts back into typed AuditKIT Sample objects."""
    samples: list[Sample] = []
    for idx, item in enumerate(items):
        if isinstance(item, Sample):
            samples.append(item)
            continue
        samples.append(
            Sample(
                id=str(item.get("id") or f"bfcl_{idx}"),
                input=str(item.get("input") or ""),
                target=item.get("target"),
                tools=item.get("tools"),
                expected_tool_calls=item.get("expected_tool_calls"),
                metadata=dict(item.get("metadata") or {}),
                tags=list(item.get("tags") or []),
            )
        )
    return samples


def evaluate_bfcl_general_regression(
    bfcl_items: list[dict[str, Any] | Sample] | str | Path,
    baseline_model: Any,
    candidate_model: Any | None = None,
    *,
    tool_mode: str = "prompt",
    seed: int = 42,
    n_resamples: int = 500,
) -> dict[str, Any]:
    """Run real AuditKIT ToolCallAdapter(mode='prompt') + ToolCallF1(arg_match=bfcl_arg_match) BFCL eval."""
    if isinstance(bfcl_items, (str, Path)):
        raw_items = json.loads(Path(bfcl_items).read_text(encoding="utf-8"))
        samples = bfcl_dicts_to_samples(raw_items)
    else:
        samples = bfcl_dicts_to_samples(bfcl_items)

    samples_sha = hashlib.sha256(
        json.dumps(
            [(s.id, s.input, s.expected_tool_calls) for s in samples],
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]

    def _run_config_for(model_obj: Any) -> RunConfig:
        extra: dict[str, Any] = {"samples_sha": samples_sha}
        if callable(model_obj):
            parts: list[str] = [getattr(model_obj, "__name__", type(model_obj).__name__)]
            try:
                mod = inspect.getmodule(model_obj)
                if mod and getattr(mod, "__file__", None):
                    parts.append(Path(mod.__file__).read_text(encoding="utf-8", errors="ignore"))
                else:
                    parts.append(inspect.getsource(model_obj))
            except Exception:
                parts.append(str(getattr(getattr(model_obj, "__code__", None), "co_code", "")))
            extra["callable_sha"] = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
        return RunConfig(extra=extra)

    adapter = ToolCallAdapter(mode=tool_mode)
    scorer = ToolCallF1(arg_match=bfcl_arg_match)
    base_res = ak_evaluate(
        samples,
        model=baseline_model,
        adapter=adapter,
        scorers=[scorer],
        config=_run_config_for(baseline_model),
    )
    f1_key = next(
        (k for k in base_res.headline if k == "tool_call_f1" or k.startswith("tool_call_f1_")),
        None,
    )
    base_f1 = round(float(base_res.headline.get(f1_key, 0.0) if f1_key else 0.0), 4)

    out: dict[str, Any] = {
        "num_samples": len(samples),
        "tool_mode": tool_mode,
        "categories": sorted(
            {
                str(s.metadata.get("bfcl_category") or s.metadata.get("category") or "unknown")
                for s in samples
            }
        ),
        "f1_metric_key": f1_key or "tool_call_f1",
        "baseline_tool_call_f1": base_f1,
        "baseline_stats": base_res.stats,
        "baseline_run": base_res,
    }
    if candidate_model is not None:
        cand_res = ak_evaluate(
            samples,
            model=candidate_model,
            adapter=adapter,
            scorers=[scorer],
            config=_run_config_for(candidate_model),
        )
        cand_f1_key = next(
            (k for k in cand_res.headline if k == "tool_call_f1" or k.startswith("tool_call_f1_")),
            f1_key,
        )
        cand_f1 = round(float(cand_res.headline.get(cand_f1_key, 0.0) if cand_f1_key else 0.0), 4)
        pairs = score_pairs(base_res, cand_res, metric=f1_key) if f1_key else []
        if not pairs:
            pairs = score_pairs(base_res, cand_res)
        boot = paired_bootstrap(pairs, n_resamples=n_resamples, seed=seed) if len(pairs) >= 2 else None
        out.update(
            {
                "candidate_tool_call_f1": cand_f1,
                "delta_tool_call_f1": round(cand_f1 - base_f1, 4),
                "candidate_stats": cand_res.stats,
                "candidate_run": cand_res,
                "bootstrap": boot,
                "comparison": ak_compare(base_res, cand_res),
            }
        )
    return out


def compare_bfcl_run_results(baseline_run: Any, candidate_run: Any) -> Any:
    """Compare two AuditKIT RunResult objects using auditkit.compare cleanly."""
    return ak_compare(baseline_run, candidate_run)
