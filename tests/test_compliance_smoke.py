"""Comprehensive offline CPU smoke & integration tests for AutoHeal-KYC.

Uses actual downloaded HF seeds, real SQLite watchlists (data/watchlists.db),
and disjoint Split A (40), Split B (120), Split C (60) datasets alongside
deterministic LLM stubs — spending zero paid API or GPU credits.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agenttune.agentic.distill_trainer import create_distill_trainer
from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.rewards.composite import combine_rewards
from agenttune.decide.closed_loop.contracts import Failure, TrainingExample
from agenttune.decide.stages.base import flatten_stage_outputs
from agenttune.decide.stages.rules import SafeEvaluator
from auditkit.agent_eval import OBSERVED, SUCCESS, reliability
from datasets import Dataset
from transformers import AutoTokenizer

from scripts.build_splits import SPLIT_SIZES, verify_zero_leakage, _verify_oracle_for_case
from scripts.run_collect_and_heal import (
    _extract_tool_history,
    build_deterministic_student_stub,
    deterministic_teacher_synthesizer,
    load_jsonl_cases,
)
from scripts.run_full_eval import _build_offline_bfcl_predictors
from src.audit_eval import (
    compare_baseline_vs_healed,
    evaluate_bfcl_general_regression,
    evaluate_kyc_runs_with_auditkit,
)
from src.curator_pipeline import _format_dossier_text, run_kyc_curatorkit_pipeline
from src.kyc_agent_runner import (
    CallableChatAdapter,
    KYCAgentRunner,
    _format_case_prompt,
    compile_decide_graph,
    format_messages_for_cohere,
    load_decide_pipeline_config,
)
from src.kyc_oracle import compute_required_tools, evaluate_case_oracle
from src.kyc_tools import (
    DEFAULT_WATCHLIST_DB,
    ScopedToolRegistry,
    build_kyc_tool_registry,
    execute_kyc_tool_call,
    get_openai_tool_schemas,
)
from src.patch_heal import (
    ComplianceReplayValidator,
    KYCFailureClassifier,
    KYCFailureDetector,
    KYCTrainingExampleGenerator,
    detect_canonical_loop,
    evaluate_kyc_deployment_gate,
    normalize_trajectories_for_diversity,
    run_agenttune_self_heal_loop_bridge,
    run_self_heal_cycle_sync,
)


def test_config_merge_and_decide_graph_compilation() -> None:
    cfg = load_decide_pipeline_config()
    assert cfg["default_model"] is None
    assert cfg["id"] == "kyc_compliance_v1"
    assert cfg["models"]["student_base"] == "CohereLabs/tiny-aya-global"
    assert cfg["models"]["student_comparison"] == "CohereLabs/aya-expanse-8b"
    assert cfg["models"]["student_tool_call_mode"] == "prompt"
    assert len(cfg["stages"]) == 7
    assert len(cfg["edges"]) >= 6

    runner = compile_decide_graph(cfg)
    assert runner is not None

    evaluator = SafeEvaluator()
    assert evaluator.eval(
        "extract_cip.output.cip_complete == true and extract_cip.output.country_iso2_valid == true",
        flatten_stage_outputs({"extract_cip": {"cip_complete": True, "country_iso2_valid": True}}),
    ) is True
    assert evaluator.eval(
        "watchlist_screening_agent.output.sanctions_or_wallet_match == true",
        flatten_stage_outputs({"watchlist_screening_agent": {"sanctions_or_wallet_match": True}}),
    ) is True


def test_real_watchlists_db_populated_and_tools_enforced() -> None:
    assert DEFAULT_WATCHLIST_DB.exists(), "Run mock_watchlists.py before running tests"
    conn = sqlite3.connect(str(DEFAULT_WATCHLIST_DB))
    try:
        conn.row_factory = sqlite3.Row
        counts = {
            tbl: conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            for tbl in ("sanctions", "export_control", "pep_adverse_media", "crypto_wallets")
        }
        assert counts["sanctions"] >= 4000
        assert counts["export_control"] >= 1500
        assert counts["pep_adverse_media"] >= 1000
        assert counts["crypto_wallets"] >= 900

        sdn_primary = dict(
            conn.execute(
                "SELECT * FROM sanctions WHERE name_type IN ('PRIMARY', 'AKA_STRONG') AND length(name) >= 6 LIMIT 1"
            ).fetchone()
        )
        sdn_weak = dict(
            conn.execute(
                "SELECT * FROM sanctions WHERE name_type = 'AKA_WEAK' AND length(name) >= 6 LIMIT 1"
            ).fetchone()
        )
        bis_row = dict(
            conn.execute("SELECT * FROM export_control WHERE length(name) >= 6 LIMIT 1").fetchone()
        )
        pep_row = dict(
            conn.execute(
                "SELECT * FROM pep_adverse_media WHERE birth_date IS NOT NULL AND length(name) >= 6 LIMIT 1"
            ).fetchone()
        )
        wallet_row = dict(
            conn.execute("SELECT * FROM crypto_wallets LIMIT 1").fetchone()
        )
    finally:
        conn.close()

    registry = build_kyc_tool_registry(DEFAULT_WATCHLIST_DB, include_builtins=False)
    assert len(get_openai_tool_schemas()) == 6

    # Strict ISO-2 & DOB rejection (no silent coercion)
    bad_iso = execute_kyc_tool_call(
        registry,
        "screen_sanctions",
        {"name": "Acme Corp", "entity_type": "organization", "country_iso2": "USA"},
    )
    assert bad_iso["success"] is False
    assert "SCHEMA_VALIDATION_ERROR" in bad_iso["error"] or "ISO2" in bad_iso["error"]

    bad_dob = execute_kyc_tool_call(
        registry,
        "screen_sanctions",
        {"name": "Acme Corp", "entity_type": "individual", "country_iso2": "US", "date_of_birth": "1985/04/12"},
    )
    assert bad_dob["success"] is False

    # Real sanctions PRIMARY -> DENY (use matching DOB if present so disambiguation does not clear it)
    res_primary = execute_kyc_tool_call(
        registry,
        "screen_sanctions",
        {
            "name": sdn_primary["name"],
            "entity_type": sdn_primary["entity_type"],
            "country_iso2": "US",
        },
    )
    assert res_primary["success"] is True and res_primary["output"]["matched"] is True
    oracle_p = evaluate_case_oracle(
        {
            "customer_name": sdn_primary["name"],
            "entity_type": sdn_primary["entity_type"],
            "country_iso2": "US",
            "date_of_birth": sdn_primary.get("birth_date") or "1980-01-01",
            "id_number": "ID1",
        },
        tool_observations=[{"tool": "screen_sanctions", "result": res_primary["output"]}],
    )
    assert oracle_p["verdict"] in {"DENY", "REVIEW"}

    res_weak = execute_kyc_tool_call(
        registry,
        "screen_sanctions",
        {"name": sdn_weak["name"], "entity_type": sdn_weak["entity_type"], "country_iso2": "US"},
    )
    assert res_weak["success"] is True and res_weak["output"]["matched"] is True
    assert any(m["match_strength"] == "AKA_WEAK" for m in res_weak["output"]["matches"])

    # Real BIS export control hit
    res_bis = execute_kyc_tool_call(
        registry,
        "screen_export_control",
        {"entity_name": bis_row["name"], "country_iso2": "CN"},
    )
    assert res_bis["success"] is True and res_bis["output"]["matched"] is True

    # Real crypto wallet hit & negative
    res_wal = execute_kyc_tool_call(
        registry,
        "screen_crypto_wallet",
        {"wallet_address": wallet_row["address"], "currency": wallet_row["currency"]},
    )
    assert res_wal["success"] is True and res_wal["output"]["sanctioned"] is True

    res_wal_clean = execute_kyc_tool_call(
        registry,
        "screen_crypto_wallet",
        {"wallet_address": "0x000000000000000000000000000000000000dead", "currency": "ETH"},
    )
    assert res_wal_clean["success"] is True and res_wal_clean["output"]["sanctioned"] is False

    # PEP + DOB disambiguation (matching DOB -> REVIEW vs conflicting DOB -> APPROVE)
    res_pep = execute_kyc_tool_call(
        registry,
        "screen_pep_adverse_media",
        {"name": pep_row["name"], "country_iso2": "GB", "date_of_birth": "1901-01-01"},
    )
    assert res_pep["success"] is True and res_pep["output"]["matched"] is True
    # Only disambiguate if record birth_date != query DOB
    if str(pep_row["birth_date"])[:4] != "1901":
        oracle_fp = evaluate_case_oracle(
            {"customer_name": pep_row["name"], "entity_type": "individual", "country_iso2": "GB", "date_of_birth": "1901-01-01", "id_number": "P1"},
            tool_observations=[{"tool": "screen_pep_adverse_media", "result": res_pep["output"]}],
        )
        assert oracle_fp["verdict"] == "APPROVE"
        assert "DOB_FALSE_POSITIVE_CLEARED" in oracle_fp["reason_codes"]

    # FATF & SoW checks
    res_fatf = execute_kyc_tool_call(registry, "check_fatf_jurisdiction_risk", {"country_iso2": "KP"})
    assert res_fatf["success"] is True and res_fatf["output"]["risk_level"] == "CRITICAL"

    res_sow = execute_kyc_tool_call(
        registry,
        "verify_source_of_wealth",
        {
            "customer_name": "Test Client",
            "annual_income_usd": 60000,
            "declared_assets_usd": 4000000,
            "expected_monthly_tx_usd": 90000,
            "source_of_wealth": "shell_company_dividends",
            "offshore_jurisdiction": "KY",
        },
    )
    assert res_sow["success"] is True and res_sow["output"]["consistent"] is False


def test_splits_zero_leakage_and_oracle_consistency() -> None:
    splits_dir = PROJECT_ROOT / "data" / "splits"
    split_a = load_jsonl_cases(splits_dir / "split_a.jsonl")
    split_b = load_jsonl_cases(splits_dir / "split_b.jsonl")
    split_c = load_jsonl_cases(splits_dir / "split_c.jsonl")

    assert len(split_a) == SPLIT_SIZES["split_a"]
    assert len(split_b) == SPLIT_SIZES["split_b"]
    assert len(split_c) == SPLIT_SIZES["split_c"]
    verify_zero_leakage({"split_a": split_a, "split_b": split_b, "split_c": split_c})

    registry = build_kyc_tool_registry(DEFAULT_WATCHLIST_DB, include_builtins=False)
    for sample in (split_a[:12] + split_c[:12]):
        recomputed = _verify_oracle_for_case(sample, registry)
        assert recomputed["verdict"] == sample["expected_verdict"]

    # Verify 316usman seed omits noisy rejected field
    match_prompts_path = PROJECT_ROOT / "data" / "seeds" / "sanctions_match_prompts.jsonl"
    assert match_prompts_path.exists()
    first_prompt = json.loads(match_prompts_path.read_text(encoding="utf-8").splitlines()[0])
    assert "prompt" in first_prompt and "chosen" in first_prompt and "rejected" not in first_prompt

    # Verify BFCL subset exists & non-empty
    bfcl_path = splits_dir / "bfcl_subset.json"
    assert bfcl_path.exists()
    bfcl_items = json.loads(bfcl_path.read_text(encoding="utf-8"))
    assert isinstance(bfcl_items, list) and len(bfcl_items) > 0


def test_runner_audit_logs_pending_reviews_and_full_coverage_episode(tmp_path: Path) -> None:
    split_a = load_jsonl_cases(PROJECT_ROOT / "data" / "splits" / "split_a.jsonl")
    # Pick 6 diverse Split A cases
    by_tag: dict[str, dict] = {}
    for row in split_a:
        by_tag.setdefault(row["scenario_tag"], row)
    sample = list(by_tag.values())[:6]

    runner = KYCAgentRunner(
        adapter=build_deterministic_student_stub(healed=True),
        db_path=DEFAULT_WATCHLIST_DB,
        audit_dir=tmp_path / "audit",
        pending_reviews_dir=tmp_path / "pending_reviews",
    )
    runs = [runner.run_case(c) for c in sample]
    assert runner.audit_file.exists()

    audit_lines = [
        json.loads(line)
        for line in runner.audit_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stage_entries = [rec for rec in audit_lines if "stage_name" in rec]
    assert len(stage_entries) >= len(sample) * 3
    for rec in stage_entries:
        assert {"trajectory_id", "stage_name", "stage_type", "status", "state_snapshot"}.issubset(rec.keys())

    review_runs = [r for r in runs if r["verdict"] == "REVIEW"]
    assert review_runs, "Expected at least one REVIEW case to generate ticket artifact"
    for r in review_runs:
        assert r["pending_review_ticket"] is not None
        assert Path(r["pending_review_ticket"]).exists()

    for r in runs:
        assert r["verdict"] == r["oracle_verdict"]
        assert r["dangling_tool_calls"] == []
        assert r["episode"].coverage["final_state"] == OBSERVED
        assert r["episode"].coverage["artifacts"] == OBSERVED


def test_classifier_invariants_no_false_loops_and_first_try_tool_capture(tmp_path: Path) -> None:
    split_b = load_jsonl_cases(PROJECT_ROOT / "data" / "splits" / "split_b.jsonl")
    clean_org = next(c for c in split_b if c["scenario_tag"] == "clean_benign_organization")

    # 1. Distinct-arg self-correction ('USA' on turn 1 -> 'US' on turn 2) must be wrong_tool, NOT loop_collapse!
    def _self_correcting_stub(messages, tools=None, stage="", case=None):
        executed_hist, observations = _extract_tool_history(messages)
        if len(executed_hist) == 0:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_try1",
                        "type": "function",
                        "function": {
                            "name": "screen_sanctions",
                            "arguments": json.dumps({"name": case["customer_name"], "entity_type": "organization", "country_iso2": "USA"}),
                        },
                    }
                ],
            }
        policy = compute_required_tools(case)
        executed = {obs["tool"] for obs in observations if obs.get("success")}
        for req in policy["required_tools"]:
            if req not in executed:
                args = (
                    {"name": case["customer_name"], "entity_type": "organization", "country_iso2": case["expected_country_iso2"], "date_of_birth": None}
                    if req == "screen_sanctions"
                    else (
                        {"entity_name": case["customer_name"], "country_iso2": case["expected_country_iso2"]}
                        if req == "screen_export_control"
                        else (
                            {"name": case["customer_name"], "country_iso2": case["expected_country_iso2"], "date_of_birth": None}
                            if req == "screen_pep_adverse_media"
                            else {"country_iso2": case["expected_country_iso2"]}
                        )
                    )
                )
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": f"call_{req}", "type": "function", "function": {"name": req, "arguments": json.dumps(args)}}],
                }
        return {"role": "assistant", "content": json.dumps({"verdict": "APPROVE", "reason": "Recovered."})}

    runner = KYCAgentRunner(
        adapter=CallableChatAdapter(_self_correcting_stub),
        audit_dir=tmp_path / "audit",
        pending_reviews_dir=tmp_path / "reviews",
    )
    run_sc = runner.run_case(clean_org)
    assert run_sc["verdict"] == "APPROVE"
    assert len(run_sc["first_try_tool_errors"]) == 1
    assert run_sc["loop_collapse_detected"] is False

    # 2. Multi-party distinct args must NEVER trigger loop detection
    loop_check = detect_canonical_loop(
        [
            {"name": "screen_sanctions", "arguments": {"name": "Alpha Holdings", "entity_type": "organization", "country_iso2": "US"}},
            {"name": "screen_sanctions", "arguments": {"name": "Beta Director", "entity_type": "individual", "country_iso2": "GB"}},
            {"name": "screen_sanctions", "arguments": {"name": "Gamma Beneficiary", "entity_type": "individual", "country_iso2": "CA"}},
        ]
    )
    assert loop_check["is_loop"] is False

    detector = KYCFailureDetector()
    failures = detector.detect_from_runs([run_sc])
    assert len(failures) == 1

    # Noisy LLM tries mislabeling it as loop_collapse -> post-guard must override to wrong_tool!
    classifier = KYCFailureClassifier(llm_classifier=lambda f: "loop_collapse")
    classified = asyncio.run(classifier.classify_batch(failures))
    assert classified[0].root_cause == "wrong_tool"

    # Identical 3x repeat -> must lock loop_collapse even against wrong LLM label!
    loop_failure = Failure(
        trajectory_id="t_loop",
        failure_type="loop_collapse",
        failed_stage_name="watchlist_screening_agent",
        error_message="LOOP_COLLAPSE",
        context={
            "failure_id": "f_loop",
            "tool_calls": [
                {"name": "screen_sanctions", "arguments": {"name": "X", "country_iso2": "US"}},
                {"name": "screen_sanctions", "arguments": {"name": "X", "country_iso2": "US"}},
                {"name": "screen_sanctions", "arguments": {"name": "X", "country_iso2": "US"}},
            ],
        },
    )
    classified_loop = asyncio.run(KYCFailureClassifier(llm_classifier=lambda f: "wrong_routing").classify_batch([loop_failure]))
    assert classified_loop[0].root_cause == "loop_collapse"


def test_replay_validator_hard_reject_and_generator_all_5_root_causes(tmp_path: Path) -> None:
    split_b = load_jsonl_cases(PROJECT_ROOT / "data" / "splits" / "split_b.jsonl")
    by_tag: dict[str, dict] = {}
    for row in split_b:
        by_tag.setdefault(row["scenario_tag"], row)

    # Select subset producing ALL 5 root causes cleanly
    target_cases = [
        by_tag["iso2_format_challenge"],     # wrong_tool
        by_tag["clean_benign_individual"],   # loop_collapse
        by_tag["weak_alias_review"],         # wrong_routing
        by_tag["offshore_sow_review"],       # incomplete_reasoning
        by_tag["clean_benign_organization"], # hallucinated_output (ensure even suffix case id ending)
    ]
    # Make sure clean_benign_organization has even final digit to exercise hallucinated_output in stub
    for c in split_b:
        if c["scenario_tag"] == "clean_benign_organization" and int(c["case_id"][-1]) % 2 == 0:
            target_cases[-1] = c
            break

    runner = KYCAgentRunner(
        adapter=build_deterministic_student_stub(healed=False),
        audit_dir=tmp_path / "audit",
        pending_reviews_dir=tmp_path / "reviews",
    )
    runs = [runner.run_case(c) for c in target_cases]

    validator = ComplianceReplayValidator(db_path=DEFAULT_WATCHLIST_DB)
    # Direct HARD REJECT test: generator given only INVALID candidates must return [] (zero poisoning!)
    detector = KYCFailureDetector()
    failures = detector.detect_from_runs(runs)
    classifier = KYCFailureClassifier()
    classified = asyncio.run(classifier.classify_batch(failures))
    seen_causes = {cf.root_cause for cf in classified}
    assert seen_causes == {
        "wrong_tool",
        "loop_collapse",
        "wrong_routing",
        "incomplete_reasoning",
        "hallucinated_output",
    }

    poison_only_gen = KYCTrainingExampleGenerator(
        validator=validator,
        synthesizer=lambda prompt, cf: [
            json.dumps({"name": "screen_sanctions", "arguments": {"name": "Bad", "entity_type": "organization", "country_iso2": "USA"}}),
            json.dumps({"verdict": "APPROVE", "reason": "Unscreened."}),
        ],
    )
    poisoned_examples = asyncio.run(poison_only_gen.generate_batch([classified[0]]))
    assert poisoned_examples == [], "Invalid candidates must NEVER survive via reward=0 fallback!"

    # Full valid DPO generation across all 5 root causes + prompt history & Cohere prompt verification
    good_gen = KYCTrainingExampleGenerator(
        validator=validator,
        synthesizer=deterministic_teacher_synthesizer,
    )
    examples = asyncio.run(good_gen.generate_batch(classified))
    assert len(examples) == 5
    for ex in examples:
        dumped_prompt = json.dumps(ex.prompt)
        assert "Task context leading to failure" not in dumped_prompt
        assert "oracle_verdict" not in dumped_prompt
        assert "expected=" not in dumped_prompt
        assert isinstance(ex.prompt, list) and len(ex.prompt) >= 2
        assert ex.chosen != ex.rejected
        # Re-verify winning chosen is genuinely valid under validator
        is_ok, _ = asyncio.run(validator.validate_example_with_trace(ex, ex.chosen))
        assert is_ok is True

    # Verify hardened replay validator rejects unscreened REVIEW/DENY, mixed tool+verdict, and wrong edd_required
    unscreened_ex = next(ex for ex in examples if ex.root_cause == "hallucinated_output")
    for bad_verdict in ("APPROVE", "REVIEW", "DENY"):
        ok_v, reason_v = asyncio.run(
            validator.validate_example_with_trace(
                unscreened_ex,
                json.dumps({"verdict": bad_verdict, "reason": "No screening executed."}),
            )
        )
        assert ok_v is False, f"Unscreened {bad_verdict} should be rejected ({reason_v})"

    mixed_ok, _ = asyncio.run(
        validator.validate_example_with_trace(
            examples[0],
            json.dumps([{"tool_name": "screen_sanctions", "parameters": {"name": "A", "country_iso2": "US"}}])
            + "\n"
            + json.dumps({"verdict": "APPROVE", "reason": "Mixed block"}),
        )
    )
    assert mixed_ok is False

    routing_ex = next(ex for ex in examples if ex.root_cause == "wrong_routing")
    bad_edd_ok, _ = asyncio.run(
        validator.validate_example_with_trace(
            routing_ex,
            json.dumps(
                {
                    "verdict": routing_ex._kyc_meta["expected_verdict"],
                    "reason": "Flagged review.",
                    "edd_required": False,
                }
            ),
        )
    )
    assert bad_edd_ok is False

    # Test AgentTune SelfHealLoop bridge & BehavioralDiversityMonitor dict normalization bridge
    bridge = run_agenttune_self_heal_loop_bridge(failures[:2], classifier, good_gen)
    assert len(bridge["dataset_rows"]) == 2
    norm_trajs = normalize_trajectories_for_diversity(runs)
    assert all(isinstance(tc, str) for t in norm_trajs for tc in t["tool_calls"])


def test_end_to_end_collect_heal_auditkit_bootstrap_and_gate(tmp_path: Path) -> None:
    split_b = load_jsonl_cases(PROJECT_ROOT / "data" / "splits" / "split_b.jsonl", limit=36)
    split_c = load_jsonl_cases(PROJECT_ROOT / "data" / "splits" / "split_c.jsonl")

    base_runner = KYCAgentRunner(
        adapter=build_deterministic_student_stub(healed=False),
        audit_dir=tmp_path / "base_audit",
        pending_reviews_dir=tmp_path / "base_reviews",
    )
    base_b_runs = [base_runner.run_case(c) for c in split_b]

    heal_out = run_self_heal_cycle_sync(
        run_results=base_b_runs,
        db_path=DEFAULT_WATCHLIST_DB,
        synthesizer=deterministic_teacher_synthesizer,
        dpo_output_path=tmp_path / "dpo_train.jsonl",
        min_failures_to_trigger=5,
        failure_rate_threshold=0.15,
    )
    assert heal_out["buffer_size"] >= 8
    assert heal_out["trigger_decision"]["should_trigger"] is True
    assert Path(heal_out["dpo_jsonl_path"]).exists()

    dpo_rows = load_jsonl_cases(Path(heal_out["dpo_jsonl_path"]))
    assert len(dpo_rows) == heal_out["buffer_size"]
    assert isinstance(dpo_rows[0]["prompt"], list)
    assert isinstance(dpo_rows[0]["chosen"], list)
    assert isinstance(dpo_rows[0]["rejected"], list)

    # Full Split C (60 cases): Baseline vs Healed with AuditKIT & paired_bootstrap & DeploymentGate
    healed_runner = KYCAgentRunner(
        adapter=build_deterministic_student_stub(healed=True),
        audit_dir=tmp_path / "healed_audit",
        pending_reviews_dir=tmp_path / "healed_reviews",
    )
    base_c_runs = [base_runner.run_case(c) for c in split_c]
    healed_c_runs = [healed_runner.run_case(c) for c in split_c]

    base_eval = evaluate_kyc_runs_with_auditkit(base_c_runs, name="split-c-baseline")
    healed_eval = evaluate_kyc_runs_with_auditkit(healed_c_runs, name="split-c-healed")
    cmp_res = compare_baseline_vs_healed(base_eval, healed_eval, seed=42, n_resamples=500)

    assert healed_eval["kpis"]["verdict_accuracy"] == 1.0
    assert healed_eval["kpis"]["first_try_tool_accuracy"] == 1.0
    assert healed_eval["kpis"]["loop_collapse_rate"] == 0.0
    assert base_eval["kpis"]["verdict_accuracy"] < healed_eval["kpis"]["verdict_accuracy"]
    assert cmp_res["bootstrap"]["composite_pass"]["significant"] is True

    gate_ok = evaluate_kyc_deployment_gate(
        baseline_runs=base_c_runs,
        candidate_runs=healed_c_runs,
        min_samples=30,
        min_improvement=0.03,
    )
    assert gate_ok["approved"] is True

    # Regression check: flipping baseline & candidate must BLOCK deploy!
    gate_blocked = evaluate_kyc_deployment_gate(
        baseline_runs=healed_c_runs,
        candidate_runs=base_c_runs,
        min_samples=30,
        min_improvement=0.03,
    )
    assert gate_blocked["approved"] is False


def test_cloud_uv_jobs_self_contained_ast() -> None:
    for job_rel in ("jobs/train_agenttune_job.py", "jobs/eval_hf_job.py"):
        job_path = PROJECT_ROOT / job_rel
        assert job_path.exists()
        src_text = job_path.read_text(encoding="utf-8")
        assert "# /// script" in src_text
        assert "agenttune==1.1.3" in src_text
        assert "auditkit==1.1.6" in src_text
        assert "CohereLabs/tiny-aya-global" in src_text
        assert "Qwen3-4B" not in src_text and "Qwen3.5" not in src_text
        tree = ast.parse(src_text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("src"), f"Forbidden src import in {job_rel}"
                assert not node.module.startswith("scripts"), f"Forbidden scripts import in {job_rel}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("src"), f"Forbidden src import in {job_rel}"
                    assert not alias.name.startswith("scripts"), f"Forbidden scripts import in {job_rel}"


def test_cohere_tokenizer_isolated_tool_decide_and_project_heal(tmp_path: Path) -> None:
    # 1. Missing tool -> clean ToolNotFound error dict (not raw KeyError crash)
    registry = build_kyc_tool_registry(DEFAULT_WATCHLIST_DB, include_builtins=False)
    assert isinstance(registry, ScopedToolRegistry)
    missing_res = execute_kyc_tool_call(registry, "nonexistent_tool", {"x": 1})
    assert missing_res["success"] is False
    assert "nonexistent_tool" in str(missing_res["error"])

    # 2. Real IsolatedTool subprocess execution + DECIDE stages in KYCAgentRunner
    split_a = load_jsonl_cases(PROJECT_ROOT / "data" / "splits" / "split_a.jsonl")
    clean_org = next(c for c in split_a if c["scenario_tag"] == "clean_benign_organization")
    iso_runner = KYCAgentRunner(
        adapter=build_deterministic_student_stub(healed=True),
        db_path=DEFAULT_WATCHLIST_DB,
        audit_dir=tmp_path / "iso_audit",
        pending_reviews_dir=tmp_path / "iso_reviews",
        isolate_tools=True,
    )
    iso_run = iso_runner.run_case(clean_org)
    assert iso_run["verdict"] == iso_run["oracle_verdict"] == "APPROVE"
    assert len(iso_run["tool_calls"]) >= 4

    audit_lines = [
        json.loads(line)
        for line in iso_runner.audit_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stage_types = {rec["stage_name"]: rec["stage_type"] for rec in audit_lines if "stage_name" in rec}
    assert stage_types["extract_cip"] == "rules"
    assert stage_types["cip_deterministic_gate"] == "rules"
    assert stage_types["fincen_cdd_edd_oracle_and_router"] == "router"

    # 3. Project.heal(max_revisits=3) from canonical EventLog loop trace (>3 revisits triggers loop_collapse)
    loop_log = EventLog(tier="full")
    for _ in range(4):
        loop_log.append(Event(EventKind.TOOL_CALL, {"action": {"name": "screen_sanctions", "arguments": {"name": "X"}}}))
        loop_log.append(Event(EventKind.TOOL_RESULT, {"output": {"matched": False}, "tool": "screen_sanctions"}))
    loop_log.append(Event(EventKind.TURN_COMPLETE, {"step": 4}))
    healed_fails = KYCFailureDetector(max_revisits=3).detect_from_event_logs([loop_log])
    assert len(healed_fails) >= 1
    assert healed_fails[0].failure_type == "loop_collapse"

    # 4. Real cached Cohere tiny-aya-global tokenizer rendering of prompt-mode DPO rows
    tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global", local_files_only=True)
    curated_dpo = load_jsonl_cases(PROJECT_ROOT / "artifacts" / "curated" / "dpo.jsonl", limit=3)
    for row in curated_dpo:
        prompt_msgs = format_messages_for_cohere(row["prompt"])
        full_chosen = format_messages_for_cohere(prompt_msgs + row["chosen"])
        rendered = tok.apply_chat_template(
            full_chosen,
            tokenize=False,
            add_generation_prompt=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        assert isinstance(rendered, str) and len(rendered) > 50


def test_curatorkit_auditkit_reliability_and_bfcl_regression(tmp_path: Path) -> None:
    # 1. Verify CuratorKIT artifacts contain zero vc940 or Qwen3-4B refs and pass smoke curate_kyc_datasets
    manifest_text = (PROJECT_ROOT / "artifacts" / "curated" / "manifest.json").read_text(encoding="utf-8")
    assert "vc940" not in manifest_text
    assert "CohereLabs/tiny-aya-global" in (
        PROJECT_ROOT / "artifacts" / "curated" / "lexsi_provenance.json"
    ).read_text(encoding="utf-8")

    cur_smoke = run_kyc_curatorkit_pipeline(
        output_dir=tmp_path / "curated_smoke",
        db_path=DEFAULT_WATCHLIST_DB,
        max_split_cases=6,
        live=False,
    )
    assert cur_smoke["passed_count"] > 0
    assert cur_smoke["manifest_path"].exists()
    assert cur_smoke["auditkit_lineage_sidecar_path"].exists()

    # 2. Verify AuditKIT CustomPredicate, JsonSchemaValid, multi-trial reliability, perf, and sidecar on Split C subset
    split_c = load_jsonl_cases(PROJECT_ROOT / "data" / "splits" / "split_c.jsonl", limit=12)
    healed_runner = KYCAgentRunner(
        adapter=build_deterministic_student_stub(healed=True),
        db_path=DEFAULT_WATCHLIST_DB,
        audit_dir=tmp_path / "eval_audit",
        pending_reviews_dir=tmp_path / "eval_reviews",
    )
    trial_1 = [healed_runner.run_case(c) for c in split_c]
    trial_2 = [healed_runner.run_case(c) for c in split_c]
    sidecar_out = tmp_path / "eval_sidecar.jsonl"
    eval_res = evaluate_kyc_runs_with_auditkit(
        trial_1,
        name="auditkit-smoke",
        sidecar_path=sidecar_out,
        trial_runs=[trial_1, trial_2],
        reset_confirmed=False,
    )
    assert eval_res["kpis"]["terminal_json_schema_rate"] == 1.0
    assert eval_res["custom_predicates"]["all_invariants_rate"] == 1.0
    assert eval_res["reliability"]["status"] == "evaluated"
    assert eval_res["reliability"]["mean_pass_at_2"] == 1.0
    assert eval_res["reliability"]["mean_pass_hat_2"] == 1.0
    assert eval_res["reliability"]["independent"] is False
    assert eval_res["perf"]["concurrency"] == 1
    assert eval_res["curatorkit_evidence"]["manifest_loaded"] is True
    assert eval_res["curatorkit_evidence"]["curated_train_case_ids_count"] > 0
    assert eval_res["curatorkit_evidence"]["split_zero_leakage_from_curated"] is True
    assert sidecar_out.exists()

    # 3. Verify real BFCL simple/multiple subset + ToolCallAdapter(mode='prompt') + ToolCallF1 regression eval
    bfcl_path = PROJECT_ROOT / "data" / "splits" / "bfcl_subset.json"
    bfcl_items = json.loads(bfcl_path.read_text(encoding="utf-8"))
    base_pred, cand_pred = _build_offline_bfcl_predictors(bfcl_items)
    bfcl_res = evaluate_bfcl_general_regression(
        bfcl_items,
        baseline_model=base_pred,
        candidate_model=cand_pred,
        tool_mode="prompt",
        n_resamples=100,
    )
    assert any(cat in {"simple", "multiple"} for cat in bfcl_res["categories"])
    assert bfcl_res["baseline_tool_call_f1"] < bfcl_res["candidate_tool_call_f1"]
    assert bfcl_res["candidate_tool_call_f1"] >= 0.95
    assert bfcl_res["comparison"] is not None


def _import_job_module(name: str, rel_path: str):
    path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cloud_training_job_dry_run_contracts(tmp_path: Path) -> None:
    train_mod = _import_job_module("train_agenttune_job_test", "jobs/train_agenttune_job.py")
    curated_dir = PROJECT_ROOT / "artifacts" / "curated"
    grpo_rows = load_jsonl_cases(curated_dir / "grpo.jsonl", limit=20)
    dpo_rows = load_jsonl_cases(curated_dir / "dpo.jsonl")
    dpo_heal_rows = load_jsonl_cases(PROJECT_ROOT / "artifacts" / "dpo_pairs" / "dpo_train.jsonl")
    sft_rows = load_jsonl_cases(curated_dir / "sft_messages.jsonl", limit=25)

    # 1. Rollout prompt prep (GRPO + DPO rollout) produces hashable str prompts & non-None expected_verdict
    rollout_rows = train_mod.build_rollout_prompt_rows(grpo_rows)
    assert len(rollout_rows) == len(grpo_rows)
    ds_rollout = Dataset.from_list(rollout_rows)
    prompt_index = {row["prompt"]: row["expected_verdict"] for row in ds_rollout}
    assert len(prompt_index) > 0
    for r in rollout_rows:
        assert isinstance(r["prompt"], str)
        assert not r["prompt"].startswith("You can call these tools:")
        assert r["expected_verdict"] in {"APPROVE", "REVIEW", "DENY"}
    # Also handle legacy grpo rows containing pre-baked tool schema headers
    stripped = train_mod._extract_dossier_user_prompt(
        'You can call these tools:\n[]\n\nEvaluate this customer onboarding dossier:\n{"case_id": "X"}'
    )
    assert stripped.startswith("Evaluate this customer onboarding dossier:")

    # 2. Composite rewards accept BOTH list[str] and list[list[dict]] with zero litellm calls
    reward_fn = combine_rewards(
        [train_mod.kyc_compliance_reward, train_mod.kyc_process_reward],
        weights=[0.6, 0.4],
    )
    good_str = (
        '[{"tool_name": "screen_sanctions", "parameters": {"name": "Alice", "entity_type": "individual", "country_iso2": "US"}}]\n'
        '{"verdict": "APPROVE", "reason": "Clean checks."}'
    )
    bad_str = '[{"tool_name": "screen_sanctions", "parameters": {"name": "Alice", "entity_type": "individual", "country_iso2": "USA"}}] Unclear.'
    str_scores = reward_fn(
        prompts=["Evaluate dossier", "Evaluate dossier"],
        completions=[good_str, bad_str],
        expected_verdict=["APPROVE", "APPROVE"],
    )
    assert str_scores[0] > str_scores[1]

    good_msgs = [
        {"role": "assistant", "content": '[{"tool_name": "check_fatf_jurisdiction_risk", "parameters": {"country_iso2": "KY"}}]'},
        {"role": "user", "content": 'Tool Response (check_fatf_jurisdiction_risk): {"edd_required": true}'},
        {"role": "assistant", "content": '{"verdict": "REVIEW", "reason": "Offshore secrecy hub KY requires EDD."}'},
    ]
    bad_repeat_msgs = [
        {"role": "assistant", "content": '[{"tool_name": "screen_sanctions", "parameters": {"name": "A", "entity_type": "individual", "country_iso2": "USA"}}]'},
        {"role": "user", "content": 'Tool Response (screen_sanctions): {"error": "INVALID_ISO2"}'},
        {"role": "assistant", "content": '[{"tool_name": "screen_sanctions", "parameters": {"name": "A", "entity_type": "individual", "country_iso2": "USA"}}]'},
        {"role": "assistant", "content": '[{"tool_name": "screen_sanctions", "parameters": {"name": "A", "entity_type": "individual", "country_iso2": "USA"}}]'},
        {"role": "user", "content": 'Tool Response (screen_sanctions): {"error": "INVALID_ISO2"}'},
    ]
    msg_scores = reward_fn(
        prompts=["Evaluate", "Evaluate"],
        completions=[good_msgs, bad_repeat_msgs],
        expected_verdict=["REVIEW", "REVIEW"],
    )
    assert msg_scores[0] > msg_scores[1]

    # 3. Offline DPO prep + distill_sft EventLog conversion + Cohere template rendering
    tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global", local_files_only=True)
    dpo_clean = train_mod.build_offline_dpo_rows(dpo_rows) + train_mod.build_offline_dpo_rows(dpo_heal_rows)
    assert len(dpo_clean) == len(dpo_rows) + len(dpo_heal_rows)
    for row in dpo_clean:
        rendered = tok.apply_chat_template(row["prompt"] + row["chosen"], tokenize=False, add_generation_prompt=False)
        assert "<|SYSTEM_TOKEN|>" in rendered

    logs = train_mod._records_to_event_logs(sft_rows)
    assert any(any(e.kind is EventKind.TOOL_CALL for e in log.events) for log in logs)
    assert any(any(e.kind is EventKind.TOOL_RESULT for e in log.events) for log in logs)
    dt = create_distill_trainer(
        student="CohereLabs/tiny-aya-global",
        trajectories=logs,
        output_dir=str(tmp_path / "distill_dry"),
        lora_targets="all-linear",
    )
    sft_ds = dt.sft_dataset()
    assert len(sft_ds) == len(logs)
    assert any(row["messages"][0]["role"] == "system" for row in sft_ds)
    for row in sft_ds:
        rendered_sft = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        assert isinstance(rendered_sft, str) and len(rendered_sft) > 40

    # 4. Standalone SQLite BaseTools parity
    job_tools = {t.name: t for t in train_mod._build_sqlite_kyc_tools(DEFAULT_WATCHLIST_DB)}
    assert job_tools["check_fatf_jurisdiction_risk"].execute(country_iso2="KY").output["edd_required"] is True
    assert not job_tools["screen_sanctions"].execute(
        name="Test Person", entity_type="individual", country_iso2="US", date_of_birth="1985-02-30"
    ).success


def test_cloud_eval_job_and_regenerated_artifacts_parity() -> None:
    eval_mod = _import_job_module("eval_hf_job_test", "jobs/eval_hf_job.py")

    # 1. Verdict parser must NEVER default unparseable text to REVIEW
    assert eval_mod._parse_verdict_from_reply("") is None
    assert eval_mod._parse_verdict_from_reply("We should review this customer document manually.") is None
    assert eval_mod._parse_verdict_from_reply('{"verdict": "DENY", "reason": "OFAC match"}') == "DENY"

    # 2. Standalone query_sqlite_tool parity with src/kyc_tools.py (KY, AKA_WEAK, invalid DOB)
    ky_res = eval_mod.query_sqlite_tool(DEFAULT_WATCHLIST_DB, "check_fatf_jurisdiction_risk", {"country_iso2": "KY"})
    assert ky_res["success"] is True and ky_res["edd_required"] is True

    bad_dob = eval_mod.query_sqlite_tool(
        DEFAULT_WATCHLIST_DB,
        "screen_sanctions",
        {"name": "Jane Doe", "entity_type": "individual", "country_iso2": "US", "date_of_birth": "1990-02-30"},
    )
    assert bad_dob["success"] is False

    conn = sqlite3.connect(str(DEFAULT_WATCHLIST_DB))
    conn.row_factory = sqlite3.Row
    try:
        weak_row = conn.execute(
            "SELECT name FROM sanctions WHERE upper(coalesce(name_type, '')) = 'AKA_WEAK' LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert weak_row is not None
    standalone_weak = eval_mod.query_sqlite_tool(
        DEFAULT_WATCHLIST_DB,
        "screen_sanctions",
        {"name": weak_row["name"], "entity_type": "individual", "country_iso2": "US"},
    )
    assert standalone_weak["matched"] is True
    assert standalone_weak["matches"][0]["match_strength"] == "AKA_WEAK"

    # 3. Eval prompt format matches training/curator compact dossier and runner JSON payload
    split_c = load_jsonl_cases(PROJECT_ROOT / "data" / "splits" / "split_c.jsonl", limit=3)
    for c in split_c:
        prompt = eval_mod._format_dossier_prompt(c)
        assert prompt == _format_dossier_text(c)
        runner_prompt = _format_case_prompt(c)
        assert json.loads(prompt.split("\n", 1)[1]) == json.loads(runner_prompt.split("\n", 1)[1])
        assert "expected_verdict" not in prompt and "oracle_verdict" not in prompt

    # 4. AuditKIT reliability returns all_k/pass_at_k cleanly
    rel = reliability([SUCCESS, SUCCESS], k=2, independent=True, n_requested=2)
    assert float(rel["pass_at_k"]) == 1.0
    assert float(rel.get("all_k", rel.get("pass_hat_k"))) == 1.0

    # 5. Regenerated artifacts quality checks
    dpo_train_path = PROJECT_ROOT / "artifacts" / "dpo_pairs" / "dpo_train.jsonl"
    dpo_train_rows = load_jsonl_cases(dpo_train_path)
    assert len(dpo_train_rows) > 0
    for row in dpo_train_rows:
        prompt_blob = json.dumps(row.get("prompt", []))
        assert '"arguments": null' not in prompt_blob
        assert "Detected failure diagnostics" not in prompt_blob

    manifest = json.loads((PROJECT_ROOT / "artifacts" / "curated" / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest.get("export_file_counts") or {}
    for fname in ("sft_messages.jsonl", "dpo.jsonl", "grpo.jsonl"):
        actual = sum(1 for ln in (PROJECT_ROOT / "artifacts" / "curated" / fname).read_text(encoding="utf-8").splitlines() if ln.strip())
        assert counts.get(fname) == actual and actual > 0

    eval_report = json.loads((PROJECT_ROOT / "artifacts" / "runs" / "eval_report.json").read_text(encoding="utf-8"))
    healed_summary = str(eval_report["auditkit_summary"]["healed"])
    assert "Verified success: 60/60" in healed_summary
    assert "tool names only 0" in healed_summary
    assert "56/56 eligible" in healed_summary


def test_env_secrets_gate_anti_hack_rewards_and_policy_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    from curatorkit.schema import DataSample
    from src.curator_pipeline import _WatchlistSafeSecretsGate, _prompt_mode_system_text
    from src.env_utils import load_project_env
    from src.kyc_agent_runner import _build_system_prompt

    # 1. load_project_env respects override=False, strips comments/quotes, handles missing .env
    assert load_project_env(tmp_path / "nonexistent.env") is None
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nHF_TOKEN="hf_dummy_test_123"\nexport OTHER_KEY=unquoted_val # inline comment\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_TOKEN", "already_exported_token")
    monkeypatch.delenv("OTHER_KEY", raising=False)
    assert load_project_env(env_file, override=False) == env_file
    assert os.environ["HF_TOKEN"] == "already_exported_token"
    assert os.environ["OTHER_KEY"] == "unquoted_val"
    assert load_project_env(env_file, override=True) == env_file
    assert os.environ["HF_TOKEN"] == "hf_dummy_test_123"

    # 2. _WatchlistSafeSecretsGate allows verified public OFAC wallet while rejecting real secrets
    conn = sqlite3.connect(str(DEFAULT_WATCHLIST_DB))
    try:
        wallet_addr = conn.execute("SELECT address FROM crypto_wallets LIMIT 1").fetchone()[0]
    finally:
        conn.close()

    safe_sample = DataSample(
        id="wallet-safe-1",
        source_uri="unit_test://wallet-safe-1",
        task_type="sft",
        input=f"Check public OFAC wallet {wallet_addr}",
        output=f'[{{"tool_name": "screen_crypto_wallet", "parameters": {{"wallet_address": "{wallet_addr}", "currency": "BTC"}}}}]',
        metadata={"allowed_public_wallets": [wallet_addr]},
    )
    unsafe_sample = DataSample(
        id="wallet-unsafe-1",
        source_uri="unit_test://wallet-unsafe-1",
        task_type="sft",
        input=f"Check wallet {wallet_addr} with leaked ghp_1234567890abcdefghijklmnopqrstuvwxyzABCD",
        output='{"verdict": "DENY", "reason": "test"}',
        metadata={"allowed_public_wallets": [wallet_addr]},
    )
    gate = _WatchlistSafeSecretsGate(allowed_public_wallets={wallet_addr})
    passed, rejected = gate.run([safe_sample, unsafe_sample])
    assert len(passed) == 1 and passed[0].id == "wallet-safe-1"
    assert wallet_addr in passed[0].output
    assert len(rejected) == 1 and rejected[0].id == "wallet-unsafe-1"

    # Verify regenerated curated SFT retains all 16 split crypto_wallet traces and drops 0 wallet cases
    curated_dir = PROJECT_ROOT / "artifacts" / "curated"
    sft_rows = load_jsonl_cases(curated_dir / "sft_messages.jsonl")
    rej_rows = load_jsonl_cases(curated_dir / "rejected.jsonl")
    split_wallet_sft = sum(
        any(m.get("role") == "user" and '"crypto_wallet": "' in m.get("content", "") for m in r.get("messages", []))
        for r in sft_rows
    )
    asst_wallet_sft = sum(
        any(m.get("role") == "assistant" and "screen_crypto_wallet" in m.get("content", "") for m in r.get("messages", []))
        for r in sft_rows
    )
    assert split_wallet_sft == 16
    assert asst_wallet_sft >= 16
    assert len(rej_rows) == 2

    # 3. No plain-text instruction contradiction across prompt headers
    tool_schemas = get_openai_tool_schemas()
    assert "as plain text" not in _build_system_prompt(tool_schemas).lower()
    assert "as plain text" not in _prompt_mode_system_text(tool_schemas).lower()

    # 4. Oracle policy parity across ALL 220 split cases & 160 GRPO rollout rows
    train_mod = _import_job_module("train_agenttune_job_parity", "jobs/train_agenttune_job.py")
    eval_mod = _import_job_module("eval_hf_job_parity", "jobs/eval_hf_job.py")
    assert "as plain text" not in train_mod.COHERE_TOOLS_FALLBACK_PROMPT.lower()
    assert "as plain text" not in eval_mod.COHERE_TOOLS_FALLBACK_PROMPT.lower()

    cases_by_id: dict[str, dict] = {}
    for split_name in ("split_a.jsonl", "split_b.jsonl", "split_c.jsonl"):
        for c in load_jsonl_cases(PROJECT_ROOT / "data" / "splits" / split_name):
            cases_by_id[c["case_id"]] = c
            oracle_pol = compute_required_tools(c)
            expected_pol = {
                "abstain": oracle_pol["abstain"],
                "required_tools": oracle_pol["required_tools"],
                "forbidden_tools": oracle_pol["forbidden_tools"],
            }
            assert eval_mod._evaluate_cip_and_policy(c) == expected_pol
            dossier = train_mod._parse_dossier_dict(eval_mod._format_dossier_prompt(c))
            assert train_mod._compute_tool_policy_standalone(dossier) == expected_pol

    grpo_rows = load_jsonl_cases(curated_dir / "grpo.jsonl")
    rollout_rows = train_mod.build_rollout_prompt_rows(grpo_rows)
    assert len(rollout_rows) == len(grpo_rows)
    for rr in rollout_rows:
        d = train_mod._parse_dossier_dict(rr["prompt"])
        orig_pol = compute_required_tools(cases_by_id[d["case_id"]])
        assert rr["abstain"] == orig_pol["abstain"]
        assert rr["required_tools"] == orig_pol["required_tools"]
        assert rr["forbidden_tools"] == orig_pol["forbidden_tools"]

    # 5. Anti-reward-hacking & duplicate assistant extraction guards
    dup_turn = [
        {
            "role": "assistant",
            "content": '[{"tool_name": "screen_sanctions", "parameters": {"name": "Alice", "entity_type": "individual", "country_iso2": "US"}}]',
            "tool_calls": [{"function": {"name": "screen_sanctions", "arguments": '{"name": "Alice", "entity_type": "individual", "country_iso2": "US"}'}}],
        }
    ]
    _, ext_calls, _ = train_mod._extract_trajectory_from_completion(dup_turn)
    assert len(ext_calls) == 1

    reward_fn = combine_rewards([train_mod.kyc_compliance_reward, train_mod.kyc_process_reward], weights=[0.6, 0.4])
    req = ["screen_sanctions", "screen_pep_adverse_media", "check_fatf_jurisdiction_risk"]
    clean_traj = [
        {
            "role": "assistant",
            "content": json.dumps([
                {"tool_name": "screen_sanctions", "parameters": {"name": "Alice", "entity_type": "individual", "country_iso2": "US"}},
                {"tool_name": "screen_pep_adverse_media", "parameters": {"name": "Alice", "country_iso2": "US"}},
                {"tool_name": "check_fatf_jurisdiction_risk", "parameters": {"country_iso2": "US"}},
            ]),
        },
        {"role": "user", "content": 'Tool Response (screen_sanctions): {"success": true, "matched": false}'},
        {"role": "assistant", "content": '{"verdict": "APPROVE", "reason": "Clean screening."}'},
    ]
    empty_guess = '{"verdict": "APPROVE", "reason": "Guessing without calling tools."}'
    repeat_spam = [
        {
            "role": "assistant",
            "content": json.dumps([
                {"tool_name": "screen_sanctions", "parameters": {"name": "Alice", "entity_type": "individual", "country_iso2": "US"}},
                {"tool_name": "screen_sanctions", "parameters": {"name": "Alice", "entity_type": "individual", "country_iso2": "US"}},
                {"tool_name": "screen_sanctions", "parameters": {"name": "Alice", "entity_type": "individual", "country_iso2": "US"}},
            ]),
        },
        {"role": "assistant", "content": '{"verdict": "APPROVE", "reason": "Looping."}'},
    ]
    s_clean, s_empty, s_spam = reward_fn(
        prompts=["p1", "p2", "p3"],
        completions=[clean_traj, empty_guess, repeat_spam],
        expected_verdict=["APPROVE", "APPROVE", "APPROVE"],
        required_tools=[req, req, req],
        forbidden_tools=[["screen_crypto_wallet"]] * 3,
        abstain=[False, False, False],
    )
    assert s_clean > 0.5 and s_empty < 0.0 and s_clean > s_spam

    abstain_clean = '{"verdict": "REVIEW", "reason": "Incomplete CIP."}'
    abstain_violation = (
        '[{"tool_name": "screen_sanctions", "parameters": {"name": "Bob", "entity_type": "individual", "country_iso2": "US"}}]\n'
        '{"verdict": "REVIEW", "reason": "Screened anyway."}'
    )
    s_abs_ok, s_abs_bad = reward_fn(
        prompts=["a1", "a2"],
        completions=[abstain_clean, abstain_violation],
        expected_verdict=["REVIEW", "REVIEW"],
        required_tools=[[], []],
        forbidden_tools=[["screen_sanctions"], ["screen_sanctions"]],
        abstain=[True, True],
    )
    assert s_abs_ok > 0.5 and s_abs_bad < 0.0 and s_abs_ok > s_abs_bad
