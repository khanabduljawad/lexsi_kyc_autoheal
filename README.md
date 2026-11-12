# AutoHeal-KYC

Self-Healing FinCEN / FATF / OFAC Compliance & Tool-Calling Agent for **Track 3 — Agentic Reasoning** (Problem Statements: *Business workflow and decision agents* and *Self-healing agents*), built on **CuratorKIT**, **AgentTune**, and **AuditKIT** with Cohere open-weight dense student models (`CohereLabs/tiny-aya-global` 3.35B and `CohereLabs/aya-expanse-8b`).

> 📖 **Comprehensive Engineering Architecture & Mechanisms:** For a complete walkthrough of the 3-phase architecture, internal DECIDE state machine, closed-loop self-healing classifiers, composite anti-reward-hacking functions, and the 3-library contribution matrix, see [**`project_details.md`**](project_details.md).

## Architecture

1. **Deterministic Compliance Oracle (`src/kyc_oracle.py`)**:
   - Enforces FinCEN 31 CFR 1020.220 CIP completeness gates (`customer_name`, ISO-2 country, calendar `YYYY-MM-DD` DOB for individuals, `id_number`).
   - Validates strict ISO-3166-1 alpha-2 country codes (`^[A-Z]{2}$`) without silent tool-level coercion.
   - Scores FATF black/grey lists, OFAC embargoes, offshore secrecy hubs, DOB false-positive disambiguation, Source-of-Wealth inconsistencies, and FinCEN CDD/EDD dispositions (`APPROVE` | `REVIEW` | `DENY`).
2. **Schema-Enforced Isolated SQLite Tools (`src/kyc_tools.py`)**:
   - Six `AgentTune` `BaseTool` implementations wrapped with `IsolatedTool` process-isolation support and safe `ToolRegistry.get` `KeyError` handling, backed by local SQLite watchlists (`data/watchlists.db`):
     - `screen_sanctions` (OFAC SDN/Consolidated, UN SC, EU FSF, `PRIMARY`/`AKA_STRONG` vs `AKA_WEAK`)
     - `screen_export_control` (US BIS Entity List & Denied Persons)
     - `screen_pep_adverse_media` (Global PEP registries + DOB disambiguation)
     - `check_fatf_jurisdiction_risk`
     - `screen_crypto_wallet` (OFAC SDN digital currency addresses — conditional only when wallet present)
     - `verify_source_of_wealth`
3. **CuratorKIT Hygiene, Curation & Lineage Pipeline (`src/curator_pipeline.py`)**:
   - Runs `SecretsGate`, `PIIPseudonymizer(spacy_model="en_core_web_sm")`, `SchemaGate`, and semantic `DiversityGate` (plus optional live `AdversarialPreferenceTask` / `HallucinationGate` / `RewardGate`) over FinCEN/FATF/OFAC regulatory text and formatted Cohere prompt-mode Split A/B traces.
   - Exports `sft_messages.jsonl`, `dpo.jsonl`, `grpo.jsonl`, `manifest.json`, `rejected.jsonl`, `lexsi_provenance.json`, and `AuditKIT` `lineage_sidecar` metadata.
4. **DECIDE GraphRunner + Cohere Prompt-Mode Runner (`src/kyc_agent_runner.py`)**:
   - Compiles and executes `GraphRunner` (`extract_cip` / `cip_deterministic_gate` `rules`, `watchlist_screening_agent` prompt-mode ReAct loop, `fincen_cdd_edd_oracle_and_router` `router`, `route_*` `output` stages) with `AuditWriter` JSONL logging.
   - Renders multi-turn conversations strictly under Cohere's prompt-mode protocol (`DEFAULT_TOOLS_FALLBACK_PROMPT`, `fold_tool_messages_into_user`, `_merge_consecutive_same_role`, `extract_tool_calls`, `ToolCallAdapter(mode="prompt")`) — never passing unsupported `role="tool"` turns or `tools=` to Cohere templates.
   - Emits `EventLog(tier="full")` (with leading `OBSERVATION` and unwrapped bare `{"name", "arguments"}` `TOOL_CALL`s + `TURN_COMPLETE`) alongside `AuditKIT` `episode_from_eventlog` full-coverage episodes and `pending_reviews/` tickets.
5. **Closed-Loop Self-Healing (`src/patch_heal.py`)**:
   - Integrates `EventLog.to_audit_records()`, `Project.heal()`, `KYCFailureDetector`, `KYCFailureClassifier`, `ComplianceReplayValidator` (HARD REJECT anti-poisoning across all 5 root causes + JSONSchema/oracle checks), `KYCTrainingExampleGenerator` (zero expected-verdict prompt leakage, case-grounded repair candidates), `RetrainingTrigger`, `BehavioralDiversityMonitor`, and `DeploymentGate` (`>= 30` held-out cases).
6. **AuditKIT Evaluation & Statistics (`src/audit_eval.py`)**:
   - Runs `AgentEvalRunner` (`FinalStateAssertion`, `CustomPredicate`, `ToolCallValidity`, `ToolCallF1`, `RedundantToolCalls`, `AgentLoopDetection`), `JsonSchemaValid`, multi-trial `reliability()` (`pass@3` / `pass^3`), `LatencyStats`, `Throughput`, `paired_bootstrap`, `curatorkit_manifest` + `join_evidence`, and separate `BFCL_v3_simple` + `BFCL_v3_multiple` regression checks.

## Quickstart

### Phase 1: Offline Seeds, Watchlists, Curation & Verification (CPU)
```powershell
python scripts/load_hf_seeds.py
python scripts/mock_watchlists.py
python scripts/build_splits.py
python scripts/run_curate_kyc.py
pytest -v tests/test_compliance_smoke.py
```

### Phase 2: Collection & Self-Healing
```powershell
# Offline CPU smoke dry-run
python scripts/run_collect_and_heal.py --mock
python scripts/run_full_eval.py --mock

# Live HF Inference Provider rollout & CuratorKIT / DPO synthesis (requires HF_TOKEN)
python scripts/run_curate_kyc.py --live-curator
python scripts/run_collect_and_heal.py
```

### Phase 3: Self-Contained Cloud GPU Training & Evaluation (Cohere Open-Weight Models)
```bash
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN jobs/train_agenttune_job.py \
  --mode agentic_dpo_offline \
  --base-model CohereLabs/tiny-aya-global \
  --dataset <hf_user>/autoheal-kyc-dpo-pairs \
  --output-repo <hf_user>/autoheal-kyc-tiny-aya-dpo

hf jobs uv run --flavor l4x1 --secrets HF_TOKEN jobs/eval_hf_job.py \
  --base-model CohereLabs/tiny-aya-global \
  --adapter-repo <hf_user>/autoheal-kyc-tiny-aya-dpo \
  --split-c <hf_user>/autoheal-kyc-splits \
  --watchlist-db <hf_user>/autoheal-kyc-watchlists
```
