"""CuratorKIT hygiene, diversity, preference curation, and provenance pipeline for AutoHeal-KYC."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any
import warnings

from agenttune.agentic.distill_trainer import _merge_consecutive_same_role
from agenttune.agentic.rollout_engines.rollout_factory import (
    DEFAULT_TOOLS_FALLBACK_PROMPT,
    fold_tool_messages_into_user,
)
from auditkit.agent_eval import (
    curatorkit_data_sample,
    curatorkit_manifest,
    lineage_sidecar,
)
from curatorkit.exporters.dpo import DPOExporter
from curatorkit.exporters.grpo import GRPOExporter
from curatorkit.exporters.messages import MessagesExporter
from curatorkit.gates.diversity import DiversityGate
from curatorkit.gates.hallucination import HallucinationGate
from curatorkit.gates.reward import RewardGate
from curatorkit.gates.schema import SchemaGate
from curatorkit.generators.adversarial_preference import AdversarialPreferenceTask
from curatorkit.hygiene.pii import PIIPseudonymizer
from curatorkit.hygiene.secrets import SecretsGate
from curatorkit.interfaces import BaseExporter, BaseReader
from curatorkit.llm.litellm import LiteLLMBackend
from curatorkit.manifest import (
    DatasetCardGenerator,
    ProvenanceManifest,
    build_provenance,
    dataset_input,
)
from curatorkit.pipeline import Pipeline, PipelineResult
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample
import numpy as np

from src.env_utils import load_project_env
from src.kyc_oracle import compute_required_tools, evaluate_case_oracle
from src.kyc_tools import (
    DEFAULT_WATCHLIST_DB,
    build_kyc_tool_registry,
    execute_kyc_tool_call,
    get_openai_tool_schemas,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REGULATORY_POLICY_SEEDS: list[dict[str, str]] = [
    {
        "clause_id": "fincen_cip_1020_220",
        "topic": "FinCEN 31 CFR 1020.220 CIP Gate",
        "instruction": (
            "Explain the mandatory FinCEN 31 CFR 1020.220 action when an individual customer "
            "dossier is missing date_of_birth or id_number."
        ),
        "output": (
            '{"verdict": "REVIEW", "reason": "Under FinCEN 31 CFR 1020.220 Customer Identification '
            "Program rules, onboarding cannot clear without required customer_name, ISO-2 country, "
            "calendar-valid YYYY-MM-DD DOB (individuals), and government ID number. Abstain from "
            'calling external screening tools and escalate to CIP remediation REVIEW."}'
        ),
    },
    {
        "clause_id": "iso2_strict_validation",
        "topic": "ISO-3166-1 Alpha-2 Country Format Enforcement",
        "instruction": (
            "When a customer dossier says domiciled in 'United Kingdom' or 'USA', what country_iso2 "
            "argument must be supplied to KYC compliance screening tools?"
        ),
        "output": (
            "Always normalize country strings to strict uppercase two-letter ISO-3166-1 alpha-2 "
            "codes ('GB' for United Kingdom, never 'UK'; 'US' for United States, never 'USA') "
            "matching ^[A-Z]{2}$ before invoking screening tools."
        ),
    },
    {
        "clause_id": "ofac_weak_alias_vs_primary",
        "topic": "OFAC SDN PRIMARY vs AKA_WEAK Escalation Policy",
        "instruction": (
            "How does FinCEN/OFAC compliance disposition differ between an OFAC SDN PRIMARY or "
            "AKA_STRONG name match versus an AKA_WEAK match?"
        ),
        "output": (
            '{"verdict": "REVIEW", "reason": "Confirmed OFAC PRIMARY and AKA_STRONG matches require '
            "immediate DENY blocking, whereas OFAC AKA_WEAK or fuzzy matches require manual "
            'compliance REVIEW to avoid false-positive customer blocking."}'
        ),
    },
    {
        "clause_id": "pep_dob_disambiguation",
        "topic": "Politically Exposed Person DOB False-Positive Disambiguation",
        "instruction": (
            "If screen_pep_adverse_media returns a PEP hit whose birth_date year differs by more "
            "than 10 years from the verified customer DOB and no other risk flags exist, how should "
            "the case be dispositioned?"
        ),
        "output": (
            '{"verdict": "APPROVE", "reason": "Verified customer DOB cleanly disambiguates the '
            "watchlist record as a false-positive homonym under FinCEN CDD standards when no "
            'sanctions, BIS export control, FATF, or SoW risks remain."}'
        ),
    },
    {
        "clause_id": "conditional_crypto_gating",
        "topic": "Conditional OFAC Crypto Wallet Tool Policy",
        "instruction": (
            "Under AutoHeal-KYC tool calling policy, when is invoking screen_crypto_wallet allowed "
            "versus forbidden?"
        ),
        "output": (
            "Invoke screen_crypto_wallet ONLY when a non-empty cryptocurrency wallet address is "
            "present in the dossier. Calling screen_crypto_wallet with fabricated addresses on "
            "fiat-only customers is a tool-permission violation."
        ),
    },
    {
        "clause_id": "fatf_cdd_edd_jurisdiction",
        "topic": "FATF High-Risk & Secrecy Jurisdiction EDD Escalation",
        "instruction": (
            "What risk screening and disposition follow when check_fatf_jurisdiction_risk reports "
            "FATF Black List, Grey List, or Offshore Secrecy Hub exposure?"
        ),
        "output": (
            '{"verdict": "REVIEW", "reason": "FATF high-risk, increased-monitoring, and offshore '
            "secrecy jurisdictions mandate Enhanced Due Diligence (EDD) REVIEW unless a confirmed "
            'OFAC/UN/EU sanctions or BIS match forces DENY."}'
        ),
    },
    {
        "clause_id": "fincen_sow_proportionality",
        "topic": "FinCEN CDD/EDD Source of Wealth Proportionality",
        "instruction": (
            "When declared assets exceed 15x annual income (above $250k) or involve offshore trust "
            "dividends in secrecy hubs such as KY or VG, which mandatory tool and disposition apply?"
        ),
        "output": (
            "Execute verify_source_of_wealth before final disposition. Unexplained wealth ratios, "
            "high monthly transaction velocity, or offshore secrecy trusts require EDD escalation to REVIEW."
        ),
    },
    {
        "clause_id": "loop_avoidance_not_found",
        "topic": "Trajectory Efficiency & Loop Avoidance",
        "instruction": (
            "If screen_sanctions returns status NOT_FOUND with valid ISO-2 and DOB arguments, what "
            "should the compliance agent do next?"
        ),
        "output": (
            "Do NOT repeat the identical screen_sanctions call. Proceed cleanly to remaining "
            "mandatory checks (screen_pep_adverse_media, check_fatf_jurisdiction_risk, or conditional "
            "tools) and issue the final JSON compliance verdict."
        ),
    },
]


class _HashedLocalEmbedder:
    """Deterministic offline hash-projection sentence embedder matching SentenceTransformer.encode."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        del batch_size, show_progress_bar, convert_to_numpy
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9_]+", (text or "").lower())
            if not tokens:
                mat[row_idx, 0] = 1.0
                continue
            for tok in tokens:
                digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                val = int.from_bytes(digest, "little")
                bucket = val % self.dim
                sign = 1.0 if ((val >> 8) & 1) == 0 else -1.0
                mat[row_idx, bucket] += sign
            compact = re.sub(r"\s+", " ", (text or "").lower()).strip()
            for idx in range(max(0, len(compact) - 2)):
                tri = compact[idx : idx + 3]
                digest = hashlib.blake2b(tri.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest, "little") % self.dim
                mat[row_idx, bucket] += 0.25
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return mat / norms


class OfflineSafeDiversityGate(DiversityGate):
    """CuratorKIT DiversityGate with offline cached-model or local-hash embedder fallback."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.94,
        text_field: str = "auto",
        coverage_field: str | None = None,
        batch_size: int = 64,
        device: str | None = None,
        allow_remote_download: bool = False,
    ) -> None:
        super().__init__(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            text_field=text_field,
            coverage_field=coverage_field,
            batch_size=batch_size,
            device=device,
        )
        self.allow_remote_download = allow_remote_download

    def _get_text(self, sample: DataSample) -> str:
        dedup = (sample.metadata or {}).get("dedup_text")
        if isinstance(dedup, str) and dedup.strip():
            return dedup.strip()
        return super()._get_text(sample)

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        """Deduplicate independently per task_type so SFT, DPO, and GRPO representations never cannibalize each other."""
        if not samples:
            return [], []
        by_task: dict[str, list[DataSample]] = {}
        for sample in samples:
            by_task.setdefault(sample.task_type or "default", []).append(sample)
        passed_all: list[DataSample] = []
        rejected_all: list[RejectedSample] = []
        for task_samples in by_task.values():
            p, r = super().run(task_samples)
            passed_all.extend(p)
            rejected_all.extend(r)
        return passed_all, rejected_all

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device,
                local_files_only=not self.allow_remote_download,
            )
            return self._model
        except Exception as exc:
            warnings.warn(
                f"DiversityGate using deterministic offline local embedder ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            self._model = _HashedLocalEmbedder(dim=512)
            return self._model


class _CachedSysPromptPIIPseudonymizer(PIIPseudonymizer):
    """Run real Presidio PII pseudonymization while skipping repeated constant tool schema headers."""

    def __init__(self, skip_prefix: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._skip_prefix = skip_prefix
        self._skip_json_fragment = json.dumps(skip_prefix, ensure_ascii=False)[1:-1] if skip_prefix else ""

    def _pseudonymize(self, text: str, entity_map: dict[str, str], analyzer: Any, faker: Any) -> tuple[str, dict[str, int]]:
        if self._skip_prefix and text == self._skip_prefix:
            return text, {}
        if self._skip_prefix and text.startswith(self._skip_prefix):
            tail = text[len(self._skip_prefix) :]
            masked_tail, counts = super()._pseudonymize(tail, entity_map, analyzer, faker)
            return self._skip_prefix + masked_tail, counts
        if self._skip_json_fragment and self._skip_json_fragment in text:
            token = "__SYS_PROMPT_PLACEHOLDER__"
            compact = text.replace(self._skip_json_fragment, token, 1)
            masked, counts = super()._pseudonymize(compact, entity_map, analyzer, faker)
            return masked.replace(token, self._skip_json_fragment, 1), counts
        return super()._pseudonymize(text, entity_map, analyzer, faker)


class _WatchlistSafeSecretsGate(SecretsGate):
    """Run real SecretsGate while masking verified public OFAC SDN wallet addresses during scan."""

    def __init__(
        self,
        allowed_public_wallets: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._allowed_wallets = sorted(
            {w.strip() for w in (allowed_public_wallets or set()) if isinstance(w, str) and len(w.strip()) >= 12},
            key=len,
            reverse=True,
        )

    def _mask_public_wallets(self, text: str, sample_wallets: list[str]) -> str:
        masked = text
        for w in sample_wallets:
            if w and w in masked:
                masked = masked.replace(w, "<PUBLIC_OFAC_WALLET>")
        return masked

    def _scan_sample(self, sample: DataSample) -> tuple[list[dict], list[str]]:
        meta = sample.metadata if isinstance(sample.metadata, dict) else {}
        per_sample = [
            w.strip()
            for w in (meta.get("allowed_public_wallets") or [])
            if isinstance(w, str) and len(w.strip()) >= 12
        ]
        # Order longest first so partial overlaps cannot leave high-entropy tail substrings
        wallets = sorted(set(per_sample) | set(self._allowed_wallets), key=len, reverse=True)
        if not wallets:
            return super()._scan_sample(sample)

        clone = sample.model_copy(deep=True)
        for field in ("instruction", "input", "output", "chosen", "rejected"):
            val = getattr(clone, field, None)
            if isinstance(val, str) and val:
                setattr(clone, field, self._mask_public_wallets(val, wallets))
        if isinstance(clone.responses, list):
            clone.responses = [
                self._mask_public_wallets(r, wallets) if isinstance(r, str) else r
                for r in clone.responses
            ]
        if isinstance(clone.metadata, dict):
            if isinstance(clone.metadata.get("system_prompt"), str):
                clone.metadata["system_prompt"] = self._mask_public_wallets(
                    clone.metadata["system_prompt"], wallets
                )
            if isinstance(clone.metadata.get("turns"), list):
                for turn in clone.metadata["turns"]:
                    if isinstance(turn, dict) and isinstance(turn.get("content"), str):
                        turn["content"] = self._mask_public_wallets(turn["content"], wallets)
        return super()._scan_sample(clone)


def _load_public_watchlist_wallets(db_path: Path | None) -> set[str]:
    """Read known public OFAC SDN crypto wallet addresses from watchlists.db if present."""
    if db_path is None or not db_path.exists():
        return set()
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT address FROM crypto_wallets").fetchall()
        return {str(r[0]).strip() for r in rows if r and r[0]}
    except sqlite3.Error:
        return set()
    finally:
        conn.close()


class _KYCSeedReader(BaseReader):
    """CuratorKIT BaseReader feeding curated compliance DataSamples into Pipeline."""

    _display_name = "KYCSeedReader"

    def __init__(self, samples: list[DataSample]) -> None:
        self._samples = samples

    def read(self) -> tuple[list[DataSample], list[RejectedSample]]:
        return list(self._samples), []


class _TaskFilteredExporter(BaseExporter):
    """Delegate export strictly to compatible task_types so multi-task pipelines stay warning-free."""

    def __init__(self, inner: BaseExporter, task_types: set[str], display_name: str) -> None:
        self.inner = inner
        self.task_types = task_types
        self.filename = getattr(inner, "filename", "export.jsonl")
        self._display_name = display_name

    def export(self, samples: list[DataSample], output_dir: Path) -> None:
        subset = [s for s in samples if s.task_type in self.task_types]
        if subset:
            self.inner.export(subset, output_dir)
        else:
            (output_dir / self.filename).write_text("", encoding="utf-8")


def _prompt_mode_system_text(tool_schemas: list[dict[str, Any]]) -> str:
    tools_desc = json.dumps(
        [
            {
                "name": s["function"]["name"],
                "description": s["function"].get("description", ""),
                "parameters": s["function"].get("parameters", {}),
            }
            for s in tool_schemas
        ],
        indent=2,
    )
    tools_header = DEFAULT_TOOLS_FALLBACK_PROMPT.format(tools=tools_desc).replace(
        "use it to answer the user's original question as plain text.",
        "use it to answer with the required FinCEN/FATF/OFAC JSON verdict.",
    )
    return (
        tools_header
        + "\n\nStrict FinCEN/FATF/OFAC KYC Rules:\n"
        + "1. Abstain from tool calls and answer immediately with "
        + '{"verdict": "REVIEW", "reason": "..."} if CIP fields are incomplete.\n'
        + "2. Country codes MUST be uppercase ISO-2 (^[A-Z]{2}$), never 'USA' or 'UK'.\n"
        + "3. Call screen_crypto_wallet ONLY when a crypto_wallet exists in the dossier.\n"
        + "4. Final answer must be JSON {\"verdict\": \"APPROVE\"|\"REVIEW\"|\"DENY\", \"reason\": \"...\"}."
    )


def _tool_args_for_case_curated(tool_name: str, case: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic SQLite tool parameters matching KYCAgentRunner conventions."""
    country = case.get("expected_country_iso2") or case.get("country_iso2") or "US"
    name = case.get("customer_name") or "Unknown"
    entity_type = case.get("entity_type") or "individual"
    dob = case.get("date_of_birth")

    if tool_name == "screen_sanctions":
        return {"name": name, "entity_type": entity_type, "country_iso2": country, "date_of_birth": dob}
    if tool_name == "screen_export_control":
        return {"entity_name": name, "country_iso2": country}
    if tool_name == "screen_pep_adverse_media":
        return {"name": name, "country_iso2": country, "date_of_birth": dob}
    if tool_name == "check_fatf_jurisdiction_risk":
        return {"country_iso2": country}
    if tool_name == "screen_crypto_wallet":
        return {
            "wallet_address": case.get("crypto_wallet") or "",
            "currency": case.get("crypto_currency") or "ETH",
        }
    if tool_name == "verify_source_of_wealth":
        return {
            "customer_name": name,
            "annual_income_usd": float(case.get("annual_income_usd") or 0.0),
            "declared_assets_usd": float(case.get("declared_assets_usd") or 0.0),
            "expected_monthly_tx_usd": float(case.get("expected_monthly_tx_usd") or 0.0),
            "source_of_wealth": case.get("source_of_wealth") or "salary",
            "offshore_jurisdiction": case.get("offshore_jurisdiction"),
        }
    return {"country_iso2": country}


def _optimal_first_action_for_case(case: dict[str, Any]) -> tuple[str, str]:
    """Return (optimal_cohere_action_text, plausible_bad_action_text) for case curation."""
    policy = compute_required_tools(case)
    country = case.get("expected_country_iso2") or "US"
    verdict = case.get("expected_verdict") or "REVIEW"
    name = case.get("customer_name") or "Unknown"
    entity_type = case.get("entity_type") or "individual"

    if policy["abstain"]:
        good = json.dumps(
            {
                "verdict": "REVIEW",
                "reason": "Missing/invalid required CIP fields; abstained from screening and escalated to REVIEW.",
            }
        )
        bad = json.dumps(
            [
                {
                    "tool_name": "screen_sanctions",
                    "parameters": {"name": name, "entity_type": entity_type, "country_iso2": country},
                }
            ]
        )
        return good, bad

    first_tool = policy["required_tools"][0]
    good_params = _tool_args_for_case_curated(first_tool, case)
    good = json.dumps([{"tool_name": first_tool, "parameters": good_params}])

    tag = case.get("scenario_tag", "")
    if tag == "iso2_format_challenge":
        bad = json.dumps(
            [
                {
                    "tool_name": "screen_sanctions",
                    "parameters": {
                        "name": name,
                        "entity_type": entity_type,
                        "country_iso2": "UK" if country == "GB" else "USA",
                    },
                }
            ]
        )
    elif tag == "fatf_jurisdiction_review" and not case.get("crypto_wallet"):
        bad = json.dumps(
            [
                {
                    "tool_name": "screen_crypto_wallet",
                    "parameters": {"wallet_address": "0x1111111111111111", "currency": "ETH"},
                }
            ]
        )
    elif verdict != "APPROVE":
        bad = json.dumps({"verdict": "APPROVE", "reason": "Skipped mandatory EDD screening."})
    else:
        bad = json.dumps({"verdict": "DENY", "reason": "Blocked without completing KYC checks."})

    return good, bad


def _format_dossier_text(case: dict[str, Any]) -> str:
    fields = {
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
    return f"Evaluate this customer onboarding dossier:\n{json.dumps(fields, ensure_ascii=False)}"


def build_kyc_curatorkit_samples(
    splits_dir: Path | None = None,
    seeds_dir: Path | None = None,
    db_path: Path | None = None,
    max_split_cases: int | None = None,
) -> list[DataSample]:
    """Assemble SFT, preference DPO, and GRPO DataSamples exclusively from non-C sources."""
    s_dir = splits_dir or PROJECT_ROOT / "data" / "splits"
    seed_root = seeds_dir or PROJECT_ROOT / "data" / "seeds"
    tool_schemas = get_openai_tool_schemas(db_path)
    sys_prompt = _prompt_mode_system_text(tool_schemas)

    samples: list[DataSample] = []

    # 1. Regulatory policy clauses (SFT + GRPO)
    for clause in REGULATORY_POLICY_SEEDS:
        samples.append(
            DataSample(
                source_uri=f"policy://fincen_fatf_ofac/{clause['clause_id']}",
                instruction=clause["instruction"],
                output=clause["output"],
                task_type="instruction_following",
                metadata={
                    "category": "regulatory_policy",
                    "clause_id": clause["clause_id"],
                    "system_prompt": sys_prompt,
                    "dedup_text": f"policy {clause['clause_id']} {clause['topic']} {clause['instruction']}",
                },
            )
        )

    # 2. Split A & Split B KYC dossiers (NEVER Split C — zero held-out leakage!)
    train_cases: list[dict[str, Any]] = []
    for split_name in ("split_a.jsonl", "split_b.jsonl"):
        p = s_dir / split_name
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    train_cases.append(json.loads(line))
    if max_split_cases is not None:
        train_cases = train_cases[:max_split_cases]

    resolved_db = Path(db_path or DEFAULT_WATCHLIST_DB)
    registry = (
        build_kyc_tool_registry(resolved_db, include_builtins=False)
        if resolved_db.exists()
        else None
    )

    for case in train_cases:
        assert case.get("split") != "split_c", "Split C must never enter CuratorKIT training pool"
        dossier_prompt = _format_dossier_text(case)
        good_action, bad_action = _optimal_first_action_for_case(case)
        policy = compute_required_tools(case)
        verdict = case.get("expected_verdict") or "REVIEW"
        case_sig = (
            f"{case['case_id']} | {case.get('customer_name')} | "
            f"{case.get('raw_country') or case.get('country_iso2')} | "
            f"{case.get('date_of_birth')} | {case.get('id_number')} | "
            f"{case.get('scenario_tag')} | {case.get('source_of_wealth')} | "
            f"{case.get('crypto_wallet')} | {verdict}"
        )

        # Multi-turn prompt-mode trajectory across ALL required SQLite tools for SFT
        extra_turns: list[dict[str, str]] = []
        if not policy["abstain"] and policy["required_tools"]:
            req_tools = list(policy["required_tools"])
            raw_turns: list[dict[str, str]] = []
            observations: list[dict[str, Any]] = []
            for idx, req_tool in enumerate(req_tools):
                tool_args = _tool_args_for_case_curated(req_tool, case)
                if idx > 0:
                    raw_turns.append(
                        {
                            "role": "assistant",
                            "content": json.dumps([{"tool_name": req_tool, "parameters": tool_args}]),
                        }
                    )
                if registry is not None:
                    exec_res = execute_kyc_tool_call(registry, req_tool, tool_args)
                    out_obj = exec_res.get("output")
                    out_str = (
                        json.dumps(out_obj, ensure_ascii=False)
                        if isinstance(out_obj, dict)
                        else str(out_obj or exec_res.get("error") or "{}")
                    )
                    observations.append(
                        {
                            "tool": req_tool,
                            "arguments": tool_args,
                            "success": bool(exec_res.get("success", False)),
                            "output": out_obj,
                        }
                    )
                else:
                    out_obj = {"status": "NOT_FOUND", "country_iso2": case.get("expected_country_iso2") or "US"}
                    out_str = json.dumps(out_obj)
                    observations.append(
                        {
                            "tool": req_tool,
                            "arguments": tool_args,
                            "success": True,
                            "output": out_obj,
                        }
                    )
                folded_obs = fold_tool_messages_into_user(
                    [{"role": "tool", "name": req_tool, "content": out_str}]
                )
                raw_turns.extend(
                    {"role": m["role"], "content": str(m["content"])} for m in folded_obs
                )

            oracle_out = evaluate_case_oracle(case, tool_observations=observations)
            final_verdict = str(oracle_out.get("verdict") or verdict)
            reason_codes = oracle_out.get("reason_codes") or [f"scenario={case.get('scenario_tag') or 'standard'}"]
            raw_turns.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "verdict": final_verdict,
                            "reason": (
                                f"Completed all required FinCEN/FATF/OFAC checks "
                                f"({', '.join(reason_codes)}) -> {final_verdict}."
                            ),
                        }
                    ),
                }
            )
            folded_clean = _merge_consecutive_same_role(raw_turns)
            extra_turns = [{"role": m["role"], "content": str(m["content"])} for m in folded_clean]

        allowed_wallets = [str(case["crypto_wallet"]).strip()] if case.get("crypto_wallet") else []

        # 2a. SFT Messages sample
        samples.append(
            DataSample(
                source_uri=f"split://{case.get('split', 'split_b')}/{case['case_id']}",
                instruction=dossier_prompt,
                output=good_action,
                task_type="instruction_following",
                metadata={
                    "case_id": case["case_id"],
                    "scenario_tag": case.get("scenario_tag"),
                    "system_prompt": sys_prompt,
                    "turns": extra_turns,
                    "dedup_text": f"[sft] {case_sig}",
                    **({"allowed_public_wallets": allowed_wallets} if allowed_wallets else {}),
                },
            )
        )

        # 2b. Conversational Preference (DPO) sample (Cohere prompt-mode turns)
        prompt_turns = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": dossier_prompt},
        ]
        chosen_turns = [{"role": "assistant", "content": good_action}]
        rejected_turns = [{"role": "assistant", "content": bad_action}]
        samples.append(
            DataSample(
                source_uri=f"split://{case.get('split', 'split_b')}/{case['case_id']}",
                instruction=json.dumps(prompt_turns, ensure_ascii=False),
                chosen=json.dumps(chosen_turns, ensure_ascii=False),
                rejected=json.dumps(rejected_turns, ensure_ascii=False),
                task_type="preference",
                metadata={
                    "case_id": case["case_id"],
                    "scenario_tag": case.get("scenario_tag"),
                    "expected_verdict": verdict,
                    "dedup_text": f"[dpo] {case_sig}",
                    **({"allowed_public_wallets": allowed_wallets} if allowed_wallets else {}),
                },
            )
        )

        # 2c. GRPO sample (clean dossier str prompt; AgentTune injects system/tool prompt once)
        samples.append(
            DataSample(
                source_uri=f"split://{case.get('split', 'split_b')}/{case['case_id']}",
                instruction=dossier_prompt,
                output=json.dumps({"verdict": verdict}),
                responses=[good_action, bad_action],
                reward_scores=[1.0, -1.0],
                task_type="grpo",
                metadata={
                    "case_id": case["case_id"],
                    "scenario_tag": case.get("scenario_tag"),
                    "expected_verdict": verdict,
                    "dedup_text": f"[grpo] {case_sig}",
                    **({"allowed_public_wallets": allowed_wallets} if allowed_wallets else {}),
                },
            )
        )

    # 3. Include train/validation slice from 316usman/sanctions-screening-match (prompt+chosen only)
    match_file = seed_root / "sanctions_match_prompts.jsonl"
    if match_file.exists():
        for idx, line in enumerate(match_file.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "test":
                continue
            p_txt = str(row.get("prompt") or "").strip()
            c_txt = str(row.get("chosen") or "").strip()
            if len(p_txt) < 12 or len(c_txt) < 8:
                continue
            samples.append(
                DataSample(
                    source_uri=f"hf://316usman/sanctions-screening-match/{row.get('split', 'train')}/{idx}",
                    instruction=p_txt,
                    output=c_txt,
                    task_type="instruction_following",
                    metadata={"source_dataset": "316usman/sanctions-screening-match"},
                )
            )
            if idx >= 40:
                break

    return samples


def run_kyc_curatorkit_pipeline(
    output_dir: str | Path | None = None,
    splits_dir: str | Path | None = None,
    seeds_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    spacy_model: str = "en_core_web_sm",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.96,
    min_tokens: int = 8,
    max_tokens: int = 4096,
    max_split_cases: int | None = None,
    extra_samples: list[DataSample] | None = None,
    live: bool = False,
    teacher_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    teacher_api_base: str = "https://router.huggingface.co/v1",
) -> dict[str, Any]:
    """Execute CuratorKIT hygiene, schema, diversity & optional live gates with full provenance."""
    load_project_env()
    out_dir = Path(output_dir or PROJECT_ROOT / "artifacts" / "curated")
    out_dir.mkdir(parents=True, exist_ok=True)
    s_dir = Path(splits_dir) if splits_dir else PROJECT_ROOT / "data" / "splits"
    seed_dir = Path(seeds_dir) if seeds_dir else PROJECT_ROOT / "data" / "seeds"
    resolved_db = Path(db_path) if db_path else DEFAULT_WATCHLIST_DB

    samples = build_kyc_curatorkit_samples(
        splits_dir=s_dir,
        seeds_dir=seed_dir,
        db_path=Path(db_path) if db_path else None,
        max_split_cases=max_split_cases,
    )
    if extra_samples:
        samples.extend(extra_samples)

    allowed_wallets = _load_public_watchlist_wallets(resolved_db)
    sys_prompt = _prompt_mode_system_text(get_openai_tool_schemas(Path(db_path) if db_path else None))
    steps: list[Any] = [
        _KYCSeedReader(samples),
        _WatchlistSafeSecretsGate(
            allowed_public_wallets=allowed_wallets,
            fields=["instruction", "input", "output", "chosen", "rejected"],
        ),
        _CachedSysPromptPIIPseudonymizer(
            skip_prefix=sys_prompt,
            entity_types=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"],
            fields=["instruction", "output"],
            faker_seed=42,
            spacy_model=spacy_model,
        ),
        SchemaGate(min_tokens=min_tokens, max_tokens=max_tokens, use_tiktoken=False),
        OfflineSafeDiversityGate(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            text_field="auto",
            coverage_field="task_type",
            allow_remote_download=live,
        ),
    ]

    if live:
        llm = LiteLLMBackend(
            model=f"openai/{teacher_model}" if not teacher_model.startswith("openai/") else teacher_model,
            api_base=teacher_api_base,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        steps.extend(
            [
                AdversarialPreferenceTask(llm=llm, num_questions=1, seed=42),
                HallucinationGate(llm=llm, threshold=0.65, on_judge_error="reject"),
                RewardGate(llm=llm, threshold=0.65, on_judge_error="reject"),
            ]
        )

    steps.extend(
        [
            _TaskFilteredExporter(
                MessagesExporter(),
                {"instruction_following", "conversational", "unpaired_preference"},
                "MessagesExporter",
            ),
            _TaskFilteredExporter(
                DPOExporter(),
                {"preference", "implicit_preference"},
                "DPOExporter",
            ),
            _TaskFilteredExporter(
                GRPOExporter(),
                {"grpo"},
                "GRPOExporter",
            ),
        ]
    )

    pipeline = Pipeline(steps=steps, output_dir=out_dir)
    result: PipelineResult = pipeline.run()

    cfg_hash = hashlib.sha256(
        json.dumps(
            {
                "spacy_model": spacy_model,
                "embedding_model": embedding_model,
                "similarity_threshold": similarity_threshold,
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
                "live": live,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]

    provenance = build_provenance(
        method="autoheal-kyc-curatorkit",
        inputs=[
            dataset_input(str(s_dir / "split_a.jsonl"), "split_a"),
            dataset_input(str(s_dir / "split_b.jsonl"), "split_b"),
        ],
        params={
            "pipeline_config_hash": cfg_hash,
            "live": live,
            "spacy_model": spacy_model,
        },
        base_model="CohereLabs/tiny-aya-global",
    )

    pm = ProvenanceManifest(
        result=result,
        pipeline_config_hash=cfg_hash,
        output_dir=out_dir,
        provenance=provenance,
    )
    manifest_path = pm.write()
    rejected_path = pm.write_rejected_sidecar()
    exported_files = [
        out_dir / "sft_messages.jsonl",
        out_dir / "dpo.jsonl",
        out_dir / "grpo.jsonl",
    ]
    export_file_counts = {
        p.name: sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        if p.exists()
        else 0
        for p in exported_files
    }
    checksums_path = pm.write_checksums(exported_files)
    manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dict["export_file_counts"] = export_file_counts
    manifest_path.write_text(json.dumps(manifest_dict, indent=2, default=str), encoding="utf-8")
    card_path = DatasetCardGenerator().generate(
        manifest=manifest_dict,
        output_dir=out_dir,
        pipeline_name="autoheal_kyc_curated",
        write_readme=True,
    )

    # AuditKIT lineage sidecar & manifest SourceEvidence bridge
    lineage_rows = result.passed + list(result.rejected)
    sample_dicts = [s.model_dump() for s in lineage_rows]
    sidecar = lineage_sidecar(sample_dicts)
    sidecar_path = out_dir / "auditkit_lineage_sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")

    manifest_evidence = curatorkit_manifest(manifest_dict, provenance=provenance)
    sample_evidence = (
        curatorkit_data_sample(sample_dicts[0], provenance=provenance) if sample_dicts else None
    )

    return {
        "result": result,
        "output_dir": out_dir,
        "manifest_path": manifest_path,
        "rejected_path": rejected_path,
        "checksums_path": checksums_path,
        "dataset_card_path": card_path,
        "lexsi_provenance_path": out_dir / "lexsi_provenance.json",
        "auditkit_lineage_sidecar_path": sidecar_path,
        "sft_messages_path": out_dir / "sft_messages.jsonl",
        "dpo_path": out_dir / "dpo.jsonl",
        "grpo_path": out_dir / "grpo.jsonl",
        "passed_count": len(result.passed),
        "rejected_count": len(result.rejected),
        "stage_counts": result.stage_counts,
        "export_file_counts": export_file_counts,
        "manifest_evidence": manifest_evidence,
        "sample_evidence": sample_evidence,
    }
