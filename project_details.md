# AutoHeal-KYC: Architecture, Internal Mechanisms & Three-Library Integration Guide

**Problem Statements (`maxChoices: 2`):**
1. **Business workflow and decision agents** (FinCEN CIP -> OFAC/UN/EU/BIS/PEP/Crypto screening -> FinCEN CDD/EDD triage & escalation)
2. **Self-healing agents** (closed-loop failure detection, invariant-locked root-cause classification, replay-validated corrective `(prompt, chosen, rejected)` synthesis across 5 failure classes, LoRA DPO/GRPO retraining, and statistical deployment gating)

---

## 1. Mission & Core Design Philosophy

AutoHeal-KYC trains and evaluates an open-weight **Cohere** compliance agent that triages customer onboarding dossiers into **`APPROVE`**, **`REVIEW`**, or **`DENY`** while invoking real local watchlist screening tools safely, concisely, and without retry loops.

### Two Non-Negotiable Architectural Principles

1. **Deterministic Regulatory Oracle + Learnable Agent Policy:**  
   Regulatory thresholds (FinCEN CIP mandatory fields, strict uppercase `^[A-Z]{2}$` ISO-2 country codes, calendar-valid `YYYY-MM-DD` DOBs, OFAC/FATF country tiers, assets-to-income EDD ratios, and exact SQLite watchlist hit severity) are computed deterministically in [`src/kyc_oracle.py`](src/kyc_oracle.py) and enforced deterministically in DECIDE `RulesStage` / `RouterStage` nodes and `ComplianceReplayValidator`. The LLM agent does **not** invent compliance math; it must learn *when to abstain*, *which tools to run*, *how to format valid tool arguments on the first attempt*, *never to repeat answered/failing calls*, and *how to synthesize the observed tool evidence into a schema-valid JSON disposition*.
2. **Strict Cohere Open-Weight Compatibility (Prompt Mode):**  
   Track 3 requires all trained/submitted agent models to be **Cohere open-weight dense models**:
   - **Primary Student:** `CohereLabs/tiny-aya-global` (3.35B)
   - **Scaling Baseline:** `CohereLabs/aya-expanse-8b` (`CohereLabs/aya-expanse-32b` omitted due to gated 403 access)
   - **Teacher / Cross-Family Reference (diagnostics/synthesis only):** `openai/Qwen/Qwen3-235B-A22B-Instruct-2507` and `openai/deepseek-ai/DeepSeek-V3.2` via Hugging Face Router (`https://router.huggingface.co/v1`).

   Because both `CohereLabs/tiny-aya-global` and `CohereLabs/aya-expanse-8b` store `tokenizer.chat_template` as a `dict` (`{'default': ...}`), ignore OpenAI `tools=` arguments during template rendering, reject `role="tool"`, enforce strict `"user"` / `"assistant"` alternation after an optional `"system"` message, and fail if LoRA targets only standard Qwen/Llama attention names, the entire codebase standardizes on:
   - `student_tool_call_mode: prompt` (Cohere prompt-mode JSON arrays: `[{"tool_name": "...", "parameters": {...}}]`)
   - `fold_tool_messages_into_user` + `_merge_consecutive_same_role` (from `AgentTune`)
   - `target_modules: "all-linear"` (for PEFT LoRA across Cohere architectures)

---

## 2. End-to-End 3-Phase Architecture & Dataflow

```text
==========================================================================================
PHASE 1: OFFLINE DATA, WATCHLISTS, HYGIENE & CPU VERIFICATION (Implemented & Verified)
==========================================================================================
  HF Public Compliance Datasets + Faker (PIIPseudonymizer) + BFCL_v3 (simple/multiple)
        │  scripts/load_hf_seeds.py -> data/seeds/*
        ▼
  scripts/mock_watchlists.py ──▶ data/watchlists.db
        │   (sanctions: 5,997 | export_control: 2,064 | pep: 1,800 | crypto_wallets: 949)
        ▼
  scripts/build_splits.py + src/kyc_oracle.py
        ├──▶ data/splits/split_a.jsonl (40)   [SFT Warmup]
        ├──▶ data/splits/split_b.jsonl (120)  [Closed-Loop Rollout & DPO Self-Healing]
        ├──▶ data/splits/split_c.jsonl (60)   [Zero-Leakage Held-Out Test]
        └──▶ data/splits/bfcl_subset.json (25)[General Function-Calling Non-Regression]
        │
        ▼
  CuratorKIT Pipeline (src/curator_pipeline.py / scripts/run_curate_kyc.py)
        ├── Inputs: 8 Policy Seeds + Split A (40) + Split B (120) + 316usman (no test)
        ├── Multi-turn SFT trace generation against REAL data/watchlists.db SQLite tools
        ├── Gates: SecretsGate -> _CachedSysPromptPIIPseudonymizer -> SchemaGate
        │          -> OfflineSafeDiversityGate (+ optional live Adversarial/Hallucination/Reward)
        └── Exports: sft_messages.jsonl (196) | dpo.jsonl (160) | grpo.jsonl (160)
                   + manifest.json + lexsi_provenance.json + auditkit_lineage_sidecar.json

==========================================================================================
PHASE 2: CLOSED-LOOP FAILURE COLLECTION & SELF-HEALING (CPU Stubs Verified; Live Pending)
==========================================================================================
  KYCAgentRunner (src/kyc_agent_runner.py) on Split B (120)
        ├── DECIDE Stages: extract_cip -> cip_deterministic_gate -> watchlist_screening_agent
        │                  -> fincen_cdd_edd_oracle_and_router -> route_{approve,review,deny}
        └── Telemetry: kyc_audit.jsonl (AuditWriter) + kyc_events.jsonl (EventLog tier="full")
                       + pending_reviews/*.json + OBSERVED AgentEpisodes
        │
        ▼
  Self-Healing Pipeline (src/patch_heal.py / scripts/run_collect_and_heal.py)
        ├── KYCFailureDetector (AuditWriter + Project.heal() + 1st-try tool errors)
        ├── KYCFailureClassifier (pre/post observable invariant locks across all 5 causes)
        ├── KYCTrainingExampleGenerator (leak-free history + Qwen3-235B/stub candidates)
        ├── ComplianceReplayValidator (HARD REJECT against SQLite DB + FinCEN oracle)
        ├── RetrainingTrigger (count/readiness/dominance) + BehavioralDiversityMonitor
        └── Export: artifacts/dpo_pairs/dpo_train.jsonl (60 verified pairs in mock run)

==========================================================================================
PHASE 3: CLOUD POST-TRAINING, EVALUATION & GATING (Dry-Run Tested; HF Jobs GPU Pending)
==========================================================================================
  jobs/train_agenttune_job.py (PEP 723 uv script for HF Jobs)
        ├── Mode 1: distill_sft         -> create_distill_trainer() + all-linear LoRA
        ├── Mode 2: agentic_dpo_offline -> TrlAgenticDPO(use_rollouts=False) on dpo_train.jsonl
        ├── Mode 3: agentic_dpo_rollout -> TrlAgenticDPO(use_rollouts=True) + SQLite tools
        └── Mode 4: agentic_grpo        -> TrlAgenticGrpo() + compliance + process rewards
        │
        ▼
  Evaluation & Gate (src/audit_eval.py, scripts/run_full_eval.py, jobs/eval_hf_job.py)
        ├── Split C (60) AgentEvalRunner + FinalStateAssertion + 3 CustomPredicates
        ├── AuditKIT reliability (pass@k / pass^k), LatencyStats/Throughput, curatorkit_manifest
        ├── BFCL (25) via ToolCallAdapter(mode="prompt") + ToolCallF1 + paired_bootstrap
        └── AgentTune DeploymentGate (seed=42, min_samples=30, min_improvement=0.03)
==========================================================================================
```

### Repository Inventory

| Directory / File | Responsibility |
| :--- | :--- |
| [`config/decide_config.yaml`](config/decide_config.yaml) | Model IDs, HF Router endpoint, CuratorKIT settings, `agentic` limits, `closed_loop` & `retrain` LoRA hyperparameters |
| [`config/kyc_compliance.yaml`](config/kyc_compliance.yaml) | 7-stage AgentTune DECIDE workflow DAG (`kyc_compliance_v1` v2.0.0) |
| [`src/kyc_oracle.py`](src/kyc_oracle.py) | Ground-truth FinCEN/FATF/OFAC rules, ISO-2/DOB validators, risk tables, `compute_required_tools`, `evaluate_case_oracle` |
| [`src/kyc_tools.py`](src/kyc_tools.py) | 6 `KYCBaseTool` implementations, strict schema/ISO-2/DOB validation, SQLite fuzzy entity search, `IsolatedTool`, `ScopedToolRegistry` |
| [`src/kyc_agent_runner.py`](src/kyc_agent_runner.py) | DECIDE stage execution, Cohere prompt-mode ReAct loop, `EventLog(tier="full")`, `AuditWriter`, `pending_reviews/` tickets, `AgentEpisode` mapping |
| [`src/curator_pipeline.py`](src/curator_pipeline.py) | CuratorKIT multi-stage hygiene, PII pseudonymization, offline/hashembed dedup, real-DB multi-turn synthesis, SFT/DPO/GRPO export, provenance & sidecar |
| [`src/patch_heal.py`](src/patch_heal.py) | Failure detector, invariant-locked classifier, replay validator, training example generator, `SelfHealLoop` bridge, diversity monitor, `DeploymentGate` |
| [`src/audit_eval.py`](src/audit_eval.py) | AuditKIT `AgentEvalRunner`, `CustomPredicate`s, `reliability`, `LatencyStats`/`Throughput`, `join_curatorkit_evidence`, `paired_bootstrap`, BFCL evaluator |
| [`scripts/load_hf_seeds.py`](scripts/load_hf_seeds.py) | Downloads/normalizes 7 HF datasets + builds PII-sanitized synthetic controls + pinned BFCL subset |
| [`scripts/mock_watchlists.py`](scripts/mock_watchlists.py) | Indexes JSONL seeds into `data/watchlists.db` SQLite WAL database |
| [`scripts/build_splits.py`](scripts/build_splits.py) | Builds oracle-verified, zero-leakage `split_a.jsonl` (40), `split_b.jsonl` (120), `split_c.jsonl` (60) |
| [`scripts/run_curate_kyc.py`](scripts/run_curate_kyc.py) | CLI driver for `src/curator_pipeline.py` (`--live-curator` optional) |
| [`scripts/run_collect_and_heal.py`](scripts/run_collect_and_heal.py) | CLI driver collecting Split B runs (`--mock` or `--mode live`) and running self-healing to `dpo_train.jsonl` |
| [`scripts/run_full_eval.py`](scripts/run_full_eval.py) | CLI driver evaluating Split C + BFCL + `DeploymentGate` + `eval_report.json` + `eval_sidecar.jsonl` |
| [`jobs/train_agenttune_job.py`](jobs/train_agenttune_job.py) | Self-contained PEP 723 script for HF Jobs supporting all 4 AgentTune/TRL training modes |
| [`jobs/eval_hf_job.py`](jobs/eval_hf_job.py) | Self-contained PEP 723 HF Jobs evaluator running Split C + AuditKIT + BFCL under real GPU `vLLM`/HF generation |
| [`tests/test_compliance_smoke.py`](tests/test_compliance_smoke.py) | 13 offline CPU integration & contract tests covering every component |

---

## 3. Data Layer, SQLite Watchlists & Zero-Leakage Splits

### Public HF Seeds & Sanitized Synthetic Controls (`scripts/load_hf_seeds.py`)

`load_all_seeds()` builds `data/seeds/` from public, ungated datasets plus synthetic controls whose contact details are masked by CuratorKIT's `PIIPseudonymizer`:

| Seed File | Source Dataset / Generator | Count (`data/seeds/manifest.json`) | Usage |
| :--- | :--- | :--- | :--- |
| `sanctions_records.jsonl` | `proofaml/global-sanctions-regulatory-enforcement` (US OFAC SDN up to 2,500, UN SC up to 1,000, EU FSF up to 1,500) + `sahith002/ofac-screening-names-sample` (1,000 with `PRIMARY` / `AKA_STRONG` / `AKA_WEAK` labels) | **6,000** (`1,000` sahith) | `sanctions` SQLite table + Split case sourcing |
| `export_control_records.jsonl` | `proofaml/global-export-control-denied-party` | **2,064** | `export_control` SQLite table |
| `pep_records.jsonl` | `proofaml/sanctions-snapshot` | **1,800** | `pep_adverse_media` SQLite table |
| `crypto_wallets.jsonl` | `ai-decisions/ofac-sdn-crypto-addresses` | **949** | `crypto_wallets` SQLite table |
| `sanctions_match_prompts.jsonl` | `316usman/sanctions-screening-match` (`train`/`validation`/`test`, capped at `80/30/30`; **only `prompt` and `chosen`**, degenerate `rejected` intentionally dropped) | **140** | Extra `instruction_following` SFT examples in CuratorKIT (excluding `split == "test"`) |
| `sovereign_forger_kyc.jsonl` | `sovereign-forger/kyc-aml-sample-data` (`5` preview records) | **5** | Rich offshore/SoW context templates |
| `synthetic_benign_entities.jsonl` | Faker (`seed=42`) anonymized with CuratorKIT `PIIPseudonymizer` (`EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `CREDIT_CARD`) | **250** | Clean benign entities & ISO-2 challenge subjects |
| `data/splits/bfcl_subset.json` | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` via `auditkit.load_bfcl` (pinned revision `61fc0608...`, `BFCL_v3_simple` + `BFCL_v3_multiple` + `possible_answer`) | **25** | General function-calling non-regression benchmark |

### SQLite Watchlist Database (`scripts/mock_watchlists.py`)

`build_watchlists_db()` constructs `data/watchlists.db` in SQLite WAL mode with normalized name/alias columns and indexes (`idx_sanctions_norm_name`, `idx_export_norm_name`, `idx_pep_norm_name`, `idx_wallet_address`). Current row counts on disk:

- `sanctions`: **5,997**
- `export_control`: **2,064**
- `pep_adverse_media`: **1,800**
- `crypto_wallets`: **949**

### Zero-Leakage Stratified Splits (`scripts/build_splits.py`)

Every non-abstain case has its `expected_verdict` and `expected_reason_codes` derived directly by running `_verify_oracle_for_case()` against the live SQLite registry and `evaluate_case_oracle()` — never guessed from source metadata.

| Split | Total | Purpose | Verdict Distribution (Observed on Disk) |
| :--- | :---: | :--- | :--- |
| `split_a.jsonl` | **40** | Initial SFT warmup & prompt sanity | `APPROVE: 12`, `DENY: 12`, `REVIEW: 16` |
| `split_b.jsonl` | **120** | Rollout collection & closed-loop self-healing -> DPO training pairs | `APPROVE: 32`, `DENY: 38`, `REVIEW: 50` |
| `split_c.jsonl` | **60** | Strictly held-out final evaluation & `DeploymentGate` | `APPROVE: 17`, `DENY: 19`, `REVIEW: 24` |

Each split draws disjoint quotas across **12 compliance scenarios**:
`sanctions_primary_deny`, `export_control_deny`, `sanctioned_wallet_deny`, `weak_alias_review`, `pep_edd_review`, `fatf_jurisdiction_review`, `offshore_sow_review`, `abstain_missing_cip`, `dob_disambiguation_clean`, `iso2_format_challenge`, `clean_benign_organization`, and `clean_benign_individual`.

`verify_zero_leakage()` asserts pairwise disjointness across all three splits on:
1. `case_id`
2. Normalized `customer_name` (`normalize_entity_text`)
3. Lowercased `crypto_wallet`

Each split is deterministically shuffled (`random.Random(42)`) so any `--limit N` slice stays scenario-stratified without introducing overlap.

---

## 4. Compliance Domain Engine & Hardened Tool Execution

### FinCEN / FATF / OFAC Deterministic Oracle (`src/kyc_oracle.py`)

`evaluate_case_oracle(case, tool_observations)` implements a strict regulatory decision hierarchy:

1. **FinCEN CIP Check (`evaluate_cip`):**  
   Requires `customer_name`, valid ISO-2 country (`country_iso2` or `normalize_country_iso2(raw_country)`), calendar-valid `YYYY-MM-DD` DOB (when `entity_type == "individual"`), and `id_number`. If incomplete -> `tool_policy.abstain = True`, verdict **`REVIEW`** (`ABSTAIN_INCOMPLETE_CIP`).
2. **Conditional Tool Policy (`compute_required_tools`):**
   - If CIP incomplete: `required_tools = []`, `forbidden_tools = ALL_TOOL_NAMES`, `abstain = True`.
   - If CIP complete: always requires `screen_sanctions`, `screen_pep_adverse_media`, and `check_fatf_jurisdiction_risk`.
   - Requires `screen_export_control` when `entity_type == "organization"` or `case.get("check_export_control")` is true (otherwise `optional`).
   - Requires `verify_source_of_wealth` when `declared_assets_usd >= 250_000.0`, `expected_monthly_tx_usd >= 50_000.0`, `offshore_jurisdiction` is in `OFFSHORE_SECRECY_HUBS`, or `case.get("trigger_sow_check")` is true (otherwise `optional`).
   - Requires `screen_crypto_wallet` **only** when a non-empty `crypto_wallet` is present; otherwise `screen_crypto_wallet` is placed in `forbidden_tools`.
3. **Hard Confirmed Match -> `DENY`:**
   - Sanctions hit (`screen_sanctions`) not cleared by `dob_disambiguates()` and with `match_strength` / `name_type` outside `{"AKA_WEAK", "WEAK", "FUZZY_REVIEW"}`.
   - Export control hit (`screen_export_control`) with `match_strength` outside `{"WEAK", "FUZZY_REVIEW"}`.
   - Sanctioned wallet hit (`screen_crypto_wallet` returns `sanctioned == True`).
4. **Fuzzy / Weak / Jurisdiction / EDD Escalations -> `REVIEW`:**
   - Weak/fuzzy watchlist hit (`AKA_WEAK` or `FUZZY_REVIEW` in `screen_sanctions` or `screen_export_control`).
   - Active PEP or adverse media hit (`screen_pep_adverse_media`, with **`dob_disambiguates()`** clearing false-positive homonyms whenever both parsed `(year, month, day)` tuples conflict on year, month, or day).
   - High-risk jurisdiction (`FATF_BLACK_LIST` = `{KP, IR, MM}`, `FATF_GREY_LIST` = 20 jurisdictions, `OFAC_COMPREHENSIVE_EMBARGO` = `{KP, IR, CU, SY}`, `SANCTIONS_SECTORAL_HIGH_RISK` = `{RU, BY, AF, SD, LY, SO}`, `OFFSHORE_SECRECY_HUBS` = 15 jurisdictions; pinned `FATF_LIST_VERSION = "2025-2026-static"`).
   - Disproportionate Source of Wealth (`evaluate_source_of_wealth_logic`: zero income with `assets >= 100_000` or `monthly_tx >= 15_000`; `assets > 15 * income` and `assets >= 250_000` unless `inheritance`/`business_exit`/`real_estate_sale`; `(monthly_tx * 12) > 3 * income` and `monthly_tx >= 25_000`; missing/`HIGH_RISK_SOW_SOURCES` SoW; or `offshore_jurisdiction in OFFSHORE_SECRECY_HUBS`), or `beneficial_ownership_verified == False`.
5. **Clean Verification -> `APPROVE`:** All required screenings ran with zero confirmed, weak, PEP, FATF, or EDD flags.

### Schema-Enforced SQLite Tools & Subprocess Isolation (`src/kyc_tools.py`)

Six `KYCBaseTool(BaseTool)` tools back the runtime, training, and replay validator:

1. `screen_sanctions` — queries `sanctions` table (fuzzy threshold `0.80`)
2. `screen_export_control` — queries `export_control` table (fuzzy threshold `0.82`)
3. `screen_pep_adverse_media` — queries `pep_adverse_media` table with DOB disambiguation
4. `check_fatf_jurisdiction_risk` — evaluates ISO-2 against static FATF/OFAC/offshore lists
5. `screen_crypto_wallet` — case-insensitive exact match on `crypto_wallets`
6. `verify_source_of_wealth` — deterministic FinCEN CDD/EDD wealth & offshore check

Key engineering mechanisms inside `src/kyc_tools.py`:
- **Strict Zero-Coercion Argument Validation (`validate_strict_tool_args`):** Validates `jsonschema`, enforces `^[A-Z]{2}$` belonging to `OFFICIAL_ISO2_CODES` (rejecting `"USA"` / `"UK"` with an explicit error message so the model must learn ISO-2 normalization rather than relying on tool forgiveness), validates calendar-real `YYYY-MM-DD` DOBs, and checks wallet format (`len >= 12`, no spaces).
- **Two-Stage Entity Resolution (`_search_sqlite_entities`):** Runs exact `norm_name = ?` lookup plus up to 3 longest tokens via `LIKE` over `norm_name` and `norm_aliases`, then scores candidates with `fuzzy_name_score()` combining full-string `SequenceMatcher` ratio with suffix-stripped (`ltd`, `llc`, `gmbh`, `fzco`, etc.) token Jaccard + ratio (`0.7 * strip_ratio + 0.3 * token_overlap`). Exact/strong matches score `>= 0.94` (`PRIMARY`/`AKA_STRONG`/`STRONG`); weaker candidates `>= threshold` become `FUZZY_REVIEW` unless tagged `AKA_WEAK`.
- **`ScopedToolRegistry`:** Wraps `ToolRegistry` so unknown tool lookups return `None` instead of raising `KeyError` inside `AgentTune` (`agenttune/agentic/tools/registry.py` raises `KeyError`), which ensures `execute_kyc_tool_call` cleanly returns `UNKNOWN_TOOL` structured failure dicts.
- **`wrap_isolated_kyc_tools`:** Optionally wraps tools with AgentTune's `IsolatedTool` (`isolate_tools(tools, timeout_s=30.0, require_isolation=False)`) so real subprocess sandbox execution works on demand (and is verified by pytest).

---

## 5. DECIDE Workflow Runtime & Cohere Prompt-Mode Mechanism

[`config/kyc_compliance.yaml`](config/kyc_compliance.yaml) defines the 7-stage `kyc_compliance_v1` (`version: "2.0.0"`) workflow compiled into an AgentTune `GraphRunner` (`compile_decide_graph`) via `ConfigLoader.deep_merge(decide_config.yaml, kyc_compliance.yaml)`:

```text
[extract_cip] (rules)
      │
      ▼
[cip_deterministic_gate] (rules + SafeEvaluator)
      ├─ (cip_complete == false OR invalid ISO-2 OR invalid DOB) ──▶ [route_review]
      ▼ (valid CIP)
[watchlist_screening_agent] (llm_call, Cohere prompt-mode ReAct, max_steps=6)
      │
      ▼
[fincen_cdd_edd_oracle_and_router] (router + SafeEvaluator)
      ├─ loop_collapse_detected == true                           ──▶ [route_review]
      ├─ sanctions_or_wallet_match == true                        ──▶ [route_deny]
      ├─ weak_alias_review / edd_required / bo_verification / escal ──▶ [route_review]
      └─ default / clean                                          ──▶ [route_approve]
```

Inside `KYCAgentRunner.run_case(case, enforce_oracle_router=False)` (`src/kyc_agent_runner.py`):

1. **Stages 1–2 (`RulesStage` + `SafeEvaluator`):**  
   Executes real `StageExecutor.get("rules")` instances on `extract_cip` and `cip_deterministic_gate`, verifying `evaluate_cip(case)` with `SafeEvaluator().eval(...)` and logging every stage to `AuditWriter` (`artifacts/audit_logs/kyc_audit.jsonl`). Notice: unlike fake stubs, `extract_cip` is explicitly `type: rules`, **not** an unbacked `llm_call`.
2. **Stage 3 (`watchlist_screening_agent` ReAct Loop):**
   - Builds the Cohere prompt-mode system prompt using AgentTune's `DEFAULT_TOOLS_FALLBACK_PROMPT` (`_build_system_prompt`) plus 7 FinCEN/tool-calling rules.
   - Before every `adapter.chat(...)` call in `tool_mode == "prompt"`, `format_messages_for_cohere(raw_messages)` converts `raw_messages` via `fold_tool_messages_into_user()` -> converts any empty-content `tool_calls` into Cohere JSON payload text -> merges adjacent same-role turns via `_merge_consecutive_same_role()` -> prepends `"Evaluate the KYC dossier."` if a message sequence would start with `"assistant"`.
   - Parses model responses with **dual extraction**: first `agenttune.agentic.rollout_engines.tool_call_parse.extract_tool_calls()`, falling back to `auditkit.trace.parse_tool_calls()`.
   - Tracks **`canonical_tool_call_key(t_name, parsed_args)`** (normalized JSON args); repeating the exact same `(tool, args)` triple (`count >= 3`) triggers `loop_collapse_detected = True` and terminates the loop, while distinct-arg self-correction (`"USA"` -> `"US"`) gets a new key and is *never* mislabeled as a loop.
   - Records every tool invocation in `first_try_tool_errors` if `exec_res["success"]` is false — **never overwritten if step 2 succeeds** — and stores `"arguments": parsed_args` in every observation (`obs`).
   - Appends **unwrapped** `Event(EventKind.TOOL_CALL, {"action": {"name": t_name, "arguments": parsed_args, "id": tc_id}})` and `Event(EventKind.TOOL_RESULT, {"output": ..., "tool": t_name, "call_id": tc_id, ...})` to `EventLog(tier="full")`. Unwrapping `action` is essential so `EventLog.to_audit_records()` and AuditKIT's `episode_from_eventlog()` pair calls and results accurately.
3. **Stages 4–5 (`RouterStage` + `OutputStage`) & Telemetry Emission:**  
   Logs final `watchlist_screening_agent` outputs, invokes `StageExecutor.get("router")` (`fincen_cdd_edd_oracle_and_router`) and `StageExecutor.get("output")` (`route_approve`/`route_review`/`route_deny`), writes `artifacts/pending_reviews/<case_id>.json` whenever `final_verdict == "REVIEW"`, stamps `pipeline_state`, writes `AuditWriter`, appends serialized events to `artifacts/audit_logs/kyc_events.jsonl`, and upgrades `episode_from_eventlog()` via `dataclasses.replace(...)` so `coverage["final_state"] == OBSERVED` and `coverage["artifacts"] == OBSERVED`.

---

## 6. CuratorKIT Mechanism: Hygiene, Multi-Turn Synthesis & Provenance

[`src/curator_pipeline.py`](src/curator_pipeline.py) orchestrates data curation using CuratorKIT v0.1.25 (`Pipeline`, gates, exporters, provenance, and AuditKIT sidecar export):

### Input Sources (529 Raw Samples)
1. **8 `REGULATORY_POLICY_SEEDS`** (`source_uri="policy://fincen-fatf-ofac/..."`): multi-turn policy enforcement examples covering FinCEN CIP, OFAC SDN vs weak alias, ISO-2 codes, PEP DOB disambiguation, crypto wallet gating, FATF high-risk lists, loop prevention, and SoW EDD.
2. **160 Disjoint Cases (`split_a` + `split_b` only — zero `split_c` leakage):**  
   Each case emits **3 synchronized training representations**:
   - **SFT (`instruction_following`):** For non-abstain cases, `_optimal_first_action_for_case()` runs **all** required tools in `policy["required_tools"]` **against real `data/watchlists.db`**, builds raw multi-turn `(system, user, assistant -> tool -> ... -> assistant verdict)` traces, folds them with `fold_tool_messages_into_user` + `_merge_consecutive_same_role`, and stores `turns` + `system_prompt` in `metadata` for `_TaskFilteredExporter(MessagesExporter)`.
   - **Offline Preference (`preference`):** Stores Cohere prompt-mode `prompt_turns`, `chosen_turns` (valid action/verdict), and `rejected_turns` (invalid country `"USA"` / forbidden crypto tool / wrong verdict).
   - **GRPO Rollout Seed (`grpo`):** Stores clean dossier `str` in `instruction` (so `AgentTune` injects the tool/system fallback prompt cleanly once during rollout rather than double-injecting it) plus `output = {"verdict": verdict}` and `expected_verdict` metadata.
3. **`sanctions_match_prompts.jsonl` (`316usman`, excluding `test` split):** Extra `instruction_following` samples (up to index `40` processed -> `41` indices inspected, sanitized/filtered).

### Gate Chain & Engineering Hardening

```text
KYCSeedReader (529)
   │
   ▼
_WatchlistSafeSecretsGate (SecretsGate subclass; scans clone with public OFAC wallets masked)
   │  [Preserves all 16 split crypto_wallet cases unmodified; rejected 2 unallowed 316usman Hex High Entropy rows]
   ▼ (527 passed)
_CachedSysPromptPIIPseudonymizer (PIIPseudonymizer subclass)
   │  [Skips constant tool-schema prefix so Presidio/Faker doesn't spend ~2s per row scanning static prompt boilerplate]
   ▼ (527)
SchemaGate (min_tokens=8, max_tokens=4096, use_tiktoken=False)
   │  [527 passed]
   ▼
OfflineSafeDiversityGate (DiversityGate subclass)
   │  [Deduplicates per `task_type` so SFT, DPO, and GRPO share cases cleanly without stripping each other]
   │  [Loads SentenceTransformer(local_files_only=True) offline; falls back to deterministic 512-d blake2b `_HashedLocalEmbedder` when un-cached]
   ▼ (527 passed; if `--live-curator`: + AdversarialPreferenceTask + HallucinationGate + RewardGate on Qwen3-235B)
3x _TaskFilteredExporter (MessagesExporter -> sft_messages.jsonl | DPOExporter -> dpo.jsonl | GRPOExporter -> grpo.jsonl)
```

> **Why `_TaskFilteredExporter` exists:** CuratorKIT's default exporters emit every `DataSample` passing upstream gates. Without task-type filtering, `sft_messages.jsonl`, `dpo.jsonl`, and `grpo.jsonl` would each contain all 527 rows instead of only their respective formats. Wrapping each exporter in `_TaskFilteredExporter` yields clean, non-redundant exports verified in `artifacts/curated/manifest.json`:
> - `sft_messages.jsonl`: **207** rows (including all **16** split `crypto_wallet` traces)
> - `dpo.jsonl`: **160** rows
> - `grpo.jsonl`: **160** rows

Provenance & lineage outputs written alongside the JSONLs:
- `artifacts/curated/manifest.json` (`wall_clock_seconds: 51.032`, `stage_counts`, `rejected_breakdown`, `export_file_counts`)
- `artifacts/curated/rejected.jsonl`
- `artifacts/curated/checksums.txt`
- `artifacts/curated/dataset_card.md` & `README.md`
- `artifacts/curated/lexsi_provenance.json` (`schema: "lexsi.provenance/1"`, inputs `split_a.jsonl` + `split_b.jsonl`)
- `artifacts/curated/auditkit_lineage_sidecar.json` (built with `auditkit.agent_eval.lexsi.lineage_sidecar`)

---

## 7. Closed-Loop Self-Healing Mechanism (`src/patch_heal.py`)

AgentTune v1.1.3 closed-loop primitives are extended and hardened in [`src/patch_heal.py`](src/patch_heal.py) so self-healing handles **all 5 failure shapes** without label leakage, false loop diagnosis, or dataset poisoning.

### Stage A: Detection (`KYCFailureDetector`)
- Consumes completed `KYCAgentRunner` runs (`detect_from_runs`), DECIDE `AuditWriter` JSONL logs (`scan_audit_log`), and canonical `EventLog`s via `Project().add_trajectory()` + `Project.heal(max_revisits=3)`.
- Inspects `first_try_tool_errors` so a model that fails on turn 1 (e.g., sends `"USA"` instead of `"US"`) is **still detected and healed** even if it recovers cleanly on turn 2!
- Captures the exact offending assistant message via `_select_failing_assistant_message` and scopes `tool_observations` to strictly steps *before* the failing turn (`step < failing_step`) so the repair generator sees exact causal state.

### Stage B: Invariant-Locked Classification (`KYCFailureClassifier`)
Overrides `FailureClassifier` with **pre-LLM and post-LLM deterministic guards** (`classify_deterministic_root_cause`):

| Priority | Root Cause | Exact Observable Trigger | Confidence |
| :---: | :--- | :--- | :---: |
| 1 | **`loop_collapse`** | Canonical `(tool, sorted_args)` repeated `>= 3` times (`detect_canonical_loop`), or `A->B->A` alternating cycle, or `max_stage_visits > 6` / `LOOP_COLLAPSE` | `0.98` |
| 2 | **`wrong_tool`** | `first_try_tool_errors` non-empty, `forbidden_called` (e.g. `screen_crypto_wallet` with no wallet), `abstain_violation` (tools called despite incomplete CIP), `dangling_tool_calls`, or `failure_type == "tool_error"` | `0.97` |
| 3 | **`hallucinated_output`** | Issued `APPROVE` without executing mandatory `screen_sanctions` and `screen_pep_adverse_media` (`has_unscreened_approve`) | `0.96` |
| 4 | **`incomplete_reasoning`** | CIP-complete case missing one or more required conditional tools (e.g. skipped `verify_source_of_wealth` or `screen_export_control`) | `0.94` |
| 5 | **`wrong_routing`** | All required tools ran validly, but terminal `APPROVE`/`REVIEW`/`DENY` disposition mismatches `oracle_verdict` | `0.95` |

*Anti-false-loop guarantee:* Distinct-argument retries (`"USA"` -> `"US"`) and multi-party checks generate distinct `canonical_tool_call_key`s, so `detect_canonical_loop()` returns `is_loop=False` and locks them as `wrong_tool`, preventing noisy LLM judges from mislabeling them as loops.

### Stage C: Leak-Free Candidate Synthesis (`KYCTrainingExampleGenerator`)
- `_build_prompt_history()` reconstructs clean prompt context (`_build_system_prompt` + `_format_case_prompt` + prior valid JSON observations `{tool, arguments, result}`) with **zero leakage** of `expected_verdict`, `oracle_verdict`, or failure diagnostic strings (`_LEAK_RE` verified clean in generated `dpo_train.jsonl`).
- Uses injected `synthesizer` in `--mock` / test mode (`deterministic_teacher_synthesizer` in `scripts/run_collect_and_heal.py`) or async HF Router calls (`Qwen3-235B`, `num_candidates=3`, formatted for Cohere prompt mode).
- Scores passing candidates using AgentTune's `TrajectoryEvaluator(model_name="none")._calculate_tac(...)` + efficiency step ratio.
- Preserves the student's **actual failing assistant message** as `rejected` (never synthetic strawmen).

### Stage D: Hard-Reject Replay Verification (`ComplianceReplayValidator`)
Every candidate repair is executed and verified in-process via `validate_example_with_trace()` before entering DPO data:
1. Rejects empty completions or completions mixing tool calls and a terminal verdict in one turn (`MIXED_TOOL_AND_VERDICT`).
2. **If tool call(s):** enforces `abstain` blocking, `forbidden_tools` blocking, strict `ISO-2` & `DOB` checks, no intra-turn duplicates, no repeating a `loop_collapse` or already-answered `(tool, args)` key, and executes the real tool against `data/watchlists.db` (`execute_kyc_tool_call` must return `success=True`).
3. **If terminal verdict:** validates JSON against `TERMINAL_VERDICT_JSON_SCHEMA` via AuditKIT's `JsonSchemaValid`, blocks unscreened verdicts (whether `APPROVE`, `REVIEW`, or `DENY`) when CIP is complete and required tools are missing, requires `REVIEW` if CIP is incomplete, and verifies exact parity with `evaluate_case_oracle()` (including `edd_required` / `escalate_to_edd` / `escalation_required` if emitted).
4. **Zero Poisoning (`HARD REJECT`):** If *no* candidate passes validation (or `best_chosen == rejected_text`), `_process_single()` returns `None`. There is **no** fallback emitting unvalidated `reward >= 0` repairs.

### Stage E: Orchestration, Triggering, Diversity & Deployment Gate
- **`run_agenttune_self_heal_loop_bridge()`:** Bridges directly into AgentTune's native `SelfHealLoop(classifier=as_sync_classifier(classifier), generator=as_sync_generator(generator))`.
- **`RetrainingTrigger` + `TrainingBuffer`:** Uses `TriggerConfig(total_failures_threshold=min_failures_to_trigger, min_examples_ready=min_failures_to_trigger, dominance_ratio=0.85)` to avoid premature or single-mode overfitting retraining.
- **`BehavioralDiversityMonitor`:** Consumes `normalize_trajectories_for_diversity(run_results)` (converts dict `tool_calls` to `list[str]` names) to monitor trajectory collapse across iterations.
- **`export_dpo_pairs_jsonl()`:** Exports verified `(prompt, chosen, rejected)` ChatML triples to `artifacts/dpo_pairs/dpo_train.jsonl` (**60** clean pairs in the offline Split B run).
- **`evaluate_kyc_deployment_gate()`:** Invokes AgentTune `DeploymentGate(seed=42).evaluate_decision(...)` over paired baseline vs candidate runs, requiring `approved == True`, `shared_ids >= min_samples (30)`, and improvement `>= min_improvement (0.03)` before promoting a healed checkpoint.

---

## 8. Cloud Post-Training & Honest Evaluation Mechanisms

### Self-Contained HF Jobs Trainer (`jobs/train_agenttune_job.py`)

Uses PEP 723 inline dependencies (`agenttune`, `trl==1.7.1`, `peft>=0.15.0`, `transformers>=5.15,<6`) to run four AgentTune/TRL modes cleanly on cloud A10G/A100 GPUs:

| Mode | Dataset Source | AgentTune / TRL Engine | Key Cohere Hardening Applied |
| :--- | :--- | :--- | :--- |
| `distill_sft` | `sft_messages.jsonl` | `agenttune.agentic.distill_trainer.create_distill_trainer` | `normalize_cohere_messages(..., allow_assistant_start=False)`, folds `role="tool"`, ensures `"user"`-first alternation, applies `all-linear` LoRA |
| `agentic_dpo_offline` | `dpo_train.jsonl` / `dpo.jsonl` | `agenttune.backends.trl.agentic.dpo.TrlAgenticDPO(use_rollouts=False)` | Normalizes `prompt` with `allow_assistant_start=False`; normalizes `chosen`/`rejected` with **`allow_assistant_start=True`** so TRL completion lists keep pure `[{"role": "assistant", "content": ...}]` turns |
| `agentic_dpo_rollout` | `grpo.jsonl` / raw prompts | `TrlAgenticDPO(use_rollouts=True)` + `_build_sqlite_kyc_tools()` | `build_rollout_prompt_rows` emits hashable plain `str` `prompt` + `expected_verdict` + `required_tools` + `forbidden_tools` + `abstain`, avoids double-injecting tool headers |
| `agentic_grpo` | `grpo.jsonl` | `agenttune.backends.trl.agentic.grpo.TrlAgenticGrpo` | Same hashable `str` prompt + per-case tool policy columns, `tools_fallback_prompt=COHERE_TOOLS_FALLBACK_PROMPT`, and real SQLite KYC tools executed live during RL rollouts |

> **Environment & Prompt-Header Consistency:** [`src/env_utils.py`](src/env_utils.py) (`load_project_env()`) automatically loads project `.env` variables across `src/kyc_agent_runner.py`, `src/curator_pipeline.py`, and local CLI scripts without overriding explicit shell exports or printing secrets (with `.gitignore` excluding `.env`). Additionally, `DEFAULT_TOOLS_FALLBACK_PROMPT`'s `"as plain text."` suffix is replaced with `"with the required FinCEN/FATF/OFAC JSON verdict."` across `src/kyc_agent_runner.py`, `src/curator_pipeline.py`, `jobs/train_agenttune_job.py` (`COHERE_TOOLS_FALLBACK_PROMPT`), and `jobs/eval_hf_job.py`.

#### Composite Reward Signals (`train_agenttune_job.py`)
For `agentic_dpo_rollout` and `agentic_grpo`, `combine_rewards([kyc_compliance_reward, kyc_process_reward], weights=[0.6, 0.4])` combines two hardened evaluators fed via TRL/AgentTune's dataset-column forwarding (`required_tools`, `forbidden_tools`, `abstain`, `expected_verdict`):
1. **`kyc_compliance_reward`:** Extracts assistant text, non-double-counted tool calls (`_extract_trajectory_from_completion`), and tool/folded-user outputs. Awards `+0.75` for clean empty-tool `abstain` vs `-1.0 * len(calls)` when calling tools despite incomplete CIP; on non-abstain cases penalizes empty tool lists (`-1.0`), scores `required_tools` coverage (`0.80 * cov - 0.65 * missing_n`), penalizes `forbidden_tools` (`-0.75`), caps unique valid tool-schema/ISO-2 credit at `0.25`, penalizes `>= 3` loop repeats (`-1.0`, `== 2`: `-0.35`), over-budget calls (`-0.15` per extra call), and runtime tool errors (`-0.30` per `"success": false`), validates terminal JSON schema (`+0.40` / `-0.50`), and grants full verdict reward (`+1.0`) **only** when `tools_satisfied` (requiring zero calls on `abstain`, full `required_tools` on non-abstain, and zero `forbidden_tools` calls — dropping to `+0.25` otherwise, `-0.75` if verdict disagrees).
2. **`kyc_process_reward`:** Evaluates trajectories with `TrajectoryEvaluator(model_name="none", tool_schemas=_KYC_TOOL_SCHEMAS)` (`0.5 * tac + 0.3 * scsr - 0.5 * arr - 0.75 * lcf + 0.2` verdict bonus), returns `-1.0` when tools are called on an `abstain=True` case, and closes `TrajectoryEvaluator`'s default empty-call `+1.0` loophole by returning `-0.5` whenever `not calls and not outputs and not is_abstain`.

### Evaluation Harness (`src/audit_eval.py`, `scripts/run_full_eval.py`, `jobs/eval_hf_job.py`)

`evaluate_kyc_runs_with_auditkit()` and `run_full_eval.py` measure candidate checkpoints rigorously without hiding failures:

1. **Compliance KPIs (`compute_compliance_kpis`):** `verdict_accuracy`, `first_try_tool_accuracy`, `loop_collapse_rate`, `dangling_tool_rate`, `hallucination_rate`, `terminal_json_schema_rate` (via AuditKIT `JsonSchemaValid`), `abstain_accuracy`, and `mean_tool_calls`.
2. **AuditKIT `AgentEvalRunner` + Custom Predicates:**
   - Wraps every case into `AgentCase(outcome=FinalStateAssertion(key="verdict", equals=expected_verdict), reference_turns=...)`.
   - Evaluates 3 domain `CustomPredicate` invariants per episode: `cip_abstention_invariant`, `loop_free_invariant`, and `required_tool_coverage_invariant`.
3. **Multi-Trial Reliability & Performance:**
   - `compute_multi_trial_reliability()` calls AuditKIT `reliability(..., independent=reset_confirmed)` and `aggregate_verdict()`. On single-trial runs (`trials=1`), `status` is truthfully `"unknown"` (`"single trial; nothing to compare"`) rather than forging `pass@k`/`pass^k`.
   - `_compute_perf_stats()` records `LatencyStats` and `Throughput` explicitly labeled with `concurrency: 1`.
4. **CuratorKIT Zero-Leakage Audit:** `join_curatorkit_evidence()` loads `manifest.json`, `lexsi_provenance.json`, and `auditkit_lineage_sidecar.json` via `curatorkit_manifest`, `curatorkit_data_sample`, and `join_evidence()`, verifying **`joined_episodes == 0`** (`split_zero_leakage_from_curated: true`) between curated training rows (`160` unique case IDs) and Split C (`60`).
5. **Statistical Significance & General Non-Regression:**
   - `compare_baseline_vs_healed()` runs `auditkit._bootstrap.paired_bootstrap(..., n_resamples=1000, seed=42)` across `verdict_pass`, `first_try_tool_pass`, `no_loop`, `schema_valid`, and `composite_pass`.
   - `evaluate_bfcl_general_regression()` evaluates general function calling on `bfcl_subset.json` (`25` samples from `simple` & `multiple`) using `auditkit.evaluate()` with `ToolCallAdapter(mode="prompt")`, `ToolCallF1(arg_match=bfcl_arg_match)`, `paired_bootstrap`, and `auditkit.compare`.
6. **Honest Standalone HF Evaluator (`jobs/eval_hf_job.py`):**
   - Runs real GPU/Transformers inference on `CohereLabs/tiny-aya-global` / `aya-expanse-8b` + optional LoRA adapters with `COHERE_TOOLS_FALLBACK_PROMPT`, `temperature=0.0` (single trial) or `0.2` (multi-trial), **never** falls back to guessing `"APPROVE"` when output JSON is unparseable (records `"PARSE_ERROR"` and surfaces `parse_error_rate`), mirrors all 5 high-risk jurisdiction sets + complete `COUNTRY_ALIAS_TO_ISO2` + `_evaluate_cip_and_policy()` (surfacing `abstain_accuracy`, `required_tool_coverage_rate`, `policy_ok`, and honest per-trial `passed`), and logs `AgentEvalRunner.summary()`.

---

## 9. Three-Library Lexsi Contribution Matrix

| Lexsi Library | Version | Exact Modules / Symbols Used | Exact Contribution in AutoHeal-KYC |
| :--- | :---: | :--- | :--- |
| **CuratorKIT** | `0.1.25` | `Pipeline`, `DataSample`, `BaseReader`, `BaseExporter`, `SecretsGate`, `PIIPseudonymizer`, `SchemaGate`, `DiversityGate`, `AdversarialPreferenceTask`, `HallucinationGate`, `RewardGate`, `LiteLLMBackend`, `MessagesExporter`, `DPOExporter`, `GRPOExporter`, `ProvenanceManifest`, `DatasetCardGenerator`, `build_provenance`, `dataset_input` | Sanitizes Faker controls (`PIIPseudonymizer`), filters unallowed secrets & schema defects across 529 inputs (`_WatchlistSafeSecretsGate` preserved all 16 split `crypto_wallet` traces and rejected 2 unallowed `316usman` Hex High Entropy rows), deduplicates per `task_type`, provides `--live-curator` adversarial/hallucination/reward gating, exports SFT/DPO/GRPO datasets + checksums + dataset cards + `lexsi.provenance/1`. |
| **AgentTune** | `1.1.3` | `GraphRunner`, `ConfigLoader`, `StageExecutor`, `SafeEvaluator`, `flatten_stage_outputs`, `PipelineState`, `AuditWriter`, `BaseTool`, `ToolResult`, `ToolRegistry`, `IsolatedTool`, `isolate_tools`, `Event`, `EventKind`, `EventLog`, `Project` (`add_trajectory`, `heal`), `SelfHealLoop`, `as_sync_classifier`, `as_sync_generator`, `Failure`, `ClassifiedFailure`, `TrainingExample`, `FailureDetector`, `FailureClassifier`, `ReplayValidator`, `TrainingExampleGenerator`, `RetrainingTrigger`, `TriggerConfig`, `TrainingBuffer`, `BehavioralDiversityMonitor`, `DeploymentGate`, `TrajectoryEvaluator`, `DEFAULT_TOOLS_FALLBACK_PROMPT`, `fold_tool_messages_into_user`, `_merge_consecutive_same_role`, `extract_tool_calls`, `create_distill_trainer`, `TrlAgenticDPO`, `TrlAgenticGrpo` | Powers the 7-stage DECIDE compliance workflow (`RulesStage`, `RouterStage`, `OutputStage`, `AuditWriter`), Cohere prompt-mode message folding & parsing, optional `IsolatedTool` subprocess sandboxing, `EventLog(tier="full")` + `Project.heal()` failure detection, closed-loop `SelfHealLoop` + `RetrainingTrigger` + `BehavioralDiversityMonitor` + `DeploymentGate`, `TrajectoryEvaluator` rewards, and SFT/DPO/GRPO trainers. |
| **AuditKIT** | `1.1.6` | `evaluate`, `compare`, `load_bfcl`, `bfcl_arg_match`, `RunConfig`, `ToolCallAdapter`, `ToolCallF1`, `JsonSchemaValid`, `LatencyStats`, `Throughput`, `AgentCase`, `AgentEpisode`, `AgentEvalRunner`, `AgentEvalSpec`, `FinalStateAssertion`, `CustomPredicate`, `episode_from_eventlog`, `reliability`, `aggregate_verdict`, `paired_bootstrap`, `score_pairs`, `write_sidecar`, `curatorkit_manifest`, `curatorkit_data_sample`, `lineage_sidecar`, `join_evidence`, `OBSERVED`, `SUCCESS`, `FAILURE`, `parse_tool_calls` | Converts `EventLog` traces into `OBSERVED` `AgentEpisode`s, executes `AgentEvalRunner` + `FinalStateAssertion` + `CustomPredicate` invariants + `write_sidecar`, validates terminal JSON (`JsonSchemaValid`), computes `reliability` (`pass@k`/`pass^k`) & `LatencyStats`/`Throughput`, verifies zero Split C leakage via `lineage_sidecar` + `join_evidence`, runs `paired_bootstrap` significance tests, and evaluates general BFCL regression (`ToolCallAdapter(mode="prompt")` + `ToolCallF1`). |

---

## 10. Current Verification Boundary (Offline-Verified vs Cloud-Pending)

### Verified Locally on CPU (`tests/test_compliance_smoke.py` & Regenerated Artifacts)

`pytest -v tests/test_compliance_smoke.py` passed **`13 passed, 1560 warnings in 54.03s`**, verifying:

1. `test_config_merge_and_decide_graph_compilation` — merges YAML configs, checks `CohereLabs/tiny-aya-global` / `aya-expanse-8b` / `prompt` mode, compiles `GraphRunner`, tests `SafeEvaluator`.
2. `test_real_watchlists_db_populated_and_tools_enforced` — verifies all 4 SQLite tables populated and strict ISO-2/DOB/wallet/registry error handling.
3. `test_splits_zero_leakage_and_oracle_consistency` — verifies `split_a=40`, `split_b=120`, `split_c=60`, `verify_zero_leakage()`, and SQLite oracle parity.
4. `test_runner_audit_logs_pending_reviews_and_full_coverage_episode` — verifies `kyc_audit.jsonl`, `kyc_events.jsonl`, `pending_reviews/`, and `OBSERVED` `AgentEpisode` coverage.
5. `test_classifier_invariants_no_false_loops_and_first_try_tool_capture` — verifies distinct-arg self-correction (`"USA"` -> `"US"`) is classified `wrong_tool` (not `loop_collapse`), while `>= 3` repeats trigger `loop_collapse`.
6. `test_replay_validator_hard_reject_and_generator_all_5_root_causes` — verifies `HARD REJECT` anti-poisoning (bad ISO-2, forbidden crypto calls, mixed verdict+tool, unscreened verdict) and validates all 5 root causes.
7. `test_end_to_end_collect_heal_auditkit_bootstrap_and_gate` — runs end-to-end Split B collect -> heal and Split C baseline vs healed `eval_report.json` + `eval_sidecar.jsonl`.
8. `test_cloud_uv_jobs_self_contained_ast` — AST-validates both `jobs/*.py` scripts have zero `src.*` imports and valid PEP 723 dependencies.
9. `test_cohere_tokenizer_isolated_tool_decide_and_project_heal` — validates real cached `CohereLabs/tiny-aya-global` tokenizer template rendering + `IsolatedTool` subprocess call + `Project.heal()`.
10. `test_curatorkit_auditkit_reliability_and_bfcl_regression` — validates `OfflineSafeDiversityGate`, CuratorKIT artifacts, multi-trial `reliability`, `join_curatorkit_evidence`, and BFCL `ToolCallF1` regression test.
11. `test_cloud_training_job_dry_run_contracts` — instantiates non-AST dry-run normalization, reward wrappers (`kyc_compliance_reward`, `kyc_process_reward`), and hashable `str` prompt contracts from `jobs/train_agenttune_job.py`.
12. `test_cloud_eval_job_and_regenerated_artifacts_parity` — verifies `eval_hf_job.py` `PARSE_ERROR` handling, 5-set FATF parity, and parity of regenerated disk artifacts (`manifest.json`, `grpo.jsonl`, `sft_messages.jsonl`, `dpo_train.jsonl`, `eval_report.json`).
13. `test_env_secrets_gate_anti_hack_rewards_and_policy_parity` — verifies `load_project_env()` override/comment behavior, `_WatchlistSafeSecretsGate` wallet masking + real secret rejection, all `16` split wallet SFT traces retained, zero `"as plain text"` contradictions, 220/220 split + 160/160 GRPO oracle policy parity across `src/` & `jobs/`, duplicate-turn call extraction, and negative composite rewards on empty-tool / loop-spam / `abstain`-violation hacking.

#### Offline Stub Benchmark Numbers (Deterministic Stub Simulator — Not Real Model Weights)

From [`artifacts/runs/eval_report.json`](artifacts/runs/eval_report.json) on Split C (`n=60`, `trials=1`, `offline_stub` mode):

| Metric | Stub Baseline | Stub Healed | Delta / Evidence |
| :--- | :---: | :---: | :--- |
| `verdict_accuracy` | `0.6667` (`40/60`) | `1.0000` (`60/60`) | `+0.3333` (`p=0.000999`) |
| `first_try_tool_accuracy` | `0.8000` | `1.0000` | `+0.2000` (`p=0.000999`) |
| `loop_collapse_rate` | `0.0667` | `0.0000` | Eliminated (`p=0.055944`) |
| `hallucination_rate` | `0.0667` | `0.0000` | Eliminated |
| `abstain_accuracy` | `0.0000` | `1.0000` | Incomplete-CIP abstention restored |
| `cip_abstention_invariant` / `loop_free_invariant` / `required_tool_coverage_invariant` | `0.9333` / `0.9333` / `0.7833` | `1.0` / `1.0` / `1.0` | AuditKIT CustomPredicates |
| `bfcl_general_regression` (`n=25`, `simple`+`multiple`) | `0.4800` | `1.0000` | `+0.5200` (`p=0.001996`) |
| `deployment_gate` | — | **Approved (`true`)** | `task_delta: 0.3333`, `trajectory_delta: 0.0511` |
| `curatorkit_evidence.split_zero_leakage_from_curated` | — | **`true`** | `0` joined episodes across `160` curated train IDs |

> **Important Honesty Note:** The table above proves the **harness, oracle, replay validator, CuratorKIT gates, AuditKIT runner, and self-healing machinery** using deterministic CPU baseline/healed simulators. Real `CohereLabs/tiny-aya-global` (`A0`) and `CohereLabs/aya-expanse-8b` baseline metrics, `--live-curator` filtering, live Split B failure mining, cloud LoRA SFT/DPO/GRPO training (`A1`–`A5`), multi-trial (`--trials >= 3`) reliability, and real Split C + BFCL post-training numbers remain **Phase 2/3 pending execution** with live `HF_TOKEN` credentials and HF Jobs GPUs.
