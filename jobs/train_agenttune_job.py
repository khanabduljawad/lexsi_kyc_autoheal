# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "agenttune==1.1.3",
#   "auditkit==1.1.6",
#   "torch>=2.4.0",
#   "transformers>=5.15,<6",
#   "trl==1.7.1",
#   "peft>=0.15",
#   "datasets>=3.0.0",
#   "accelerate>=1.0.0",
#   "huggingface_hub>=0.24.0",
#   "jsonschema>=4.20.0",
# ]
# ///
"""Self-contained Hugging Face Jobs GPU training script using AgentTune + Cohere Tiny Aya.

Supports 4 AgentTune training modes on cloud GPUs (L4/A100) via `hf jobs uv run`:
  - distill_sft         (create_distill_trainer over curated sft_messages.jsonl trajectories)
  - agentic_dpo_offline (TrlAgenticDPO over replay-validated dpo.jsonl prompt/chosen/rejected)
  - agentic_dpo_rollout (TrlAgenticDPO online-paired rollouts with KYC BaseTools + composite reward)
  - agentic_grpo        (TrlAgenticGrpo on grpo.jsonl prompts + combine_rewards + deterministic process reward)

Strictly self-contained: zero imports from `src.*` or `scripts.*`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

from agenttune.agentic.distill_trainer import (
    _merge_consecutive_same_role,
    create_distill_trainer,
)
from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.rewards.composite import combine_rewards
from agenttune.agentic.rollout_engines.rollout_factory import (
    DEFAULT_TOOLS_FALLBACK_PROMPT,
    fold_tool_messages_into_user,
)
from agenttune.agentic.rollout_engines.tool_call_parse import extract_tool_calls
from agenttune.agentic.tools.base import BaseTool, ToolResult
from agenttune.backends.trl.agentic.dpo.agentic_dpo import TrlAgenticDPO
from agenttune.backends.trl.agentic.grpo.agentic_grpo import TrlAgenticGrpo
from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator
from agenttune.utils.hf_publish import (
    brand_hf_repo,
    push_folder_to_hub,
    push_model_to_hf,
)
from agenttune.utils.provenance import write_provenance
from datasets import Dataset
from huggingface_hub import hf_hub_download
import jsonschema
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_COHERE_STUDENT = "CohereLabs/tiny-aya-global"
ISO2_RE = re.compile(r"^[A-Z]{2}$")
DOB_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

FATF_BLACK_LIST = {"KP", "IR", "MM"}
FATF_GREY_LIST = {
    "BG", "BF", "CM", "HR", "CD", "HT", "KE", "ML", "MZ", "NA",
    "NG", "PH", "SN", "ZA", "SS", "SY", "TZ", "VE", "VN", "YE",
}
OFAC_COMPREHENSIVE_EMBARGO = {"KP", "IR", "CU", "SY"}
SANCTIONS_SECTORAL_HIGH_RISK = {"RU", "BY", "AF", "SD", "LY", "SO"}
OFFSHORE_SECRECY_HUBS = {
    "KY", "VG", "PA", "BZ", "SC", "BM", "JE", "GG", "IM", "LI", "MC", "CY", "MT", "VU", "WS",
}
FATF_HIGH_RISK = (
    FATF_BLACK_LIST
    | FATF_GREY_LIST
    | OFAC_COMPREHENSIVE_EMBARGO
    | SANCTIONS_SECTORAL_HIGH_RISK
    | OFFSHORE_SECRECY_HUBS
)

COHERE_ROLLOUT_SYSTEM_RULES = (
    "Strict FinCEN/FATF/OFAC KYC Rules:\n"
    "1. Abstain from tool calls and answer immediately with "
    '{"verdict": "REVIEW", "reason": "..."} if CIP fields are incomplete.\n'
    "2. Country codes MUST be uppercase ISO-2 (^[A-Z]{2}$), never 'USA' or 'UK'.\n"
    "3. Call screen_crypto_wallet ONLY when a crypto_wallet exists in the dossier.\n"
    '4. Final answer must be JSON {"verdict": "APPROVE"|"REVIEW"|"DENY", "reason": "..."}.'
)

COHERE_TOOLS_FALLBACK_PROMPT = DEFAULT_TOOLS_FALLBACK_PROMPT.replace(
    "use it to answer the user's original question as plain text.",
    "use it to answer with the required FinCEN/FATF/OFAC JSON verdict.",
)

ALL_KYC_TOOL_NAMES = (
    "screen_sanctions",
    "screen_export_control",
    "screen_pep_adverse_media",
    "check_fatf_jurisdiction_risk",
    "screen_crypto_wallet",
    "verify_source_of_wealth",
)

_COUNTRY_ALIAS_TO_ISO2 = {
    "USA": "US",
    "U.S.": "US",
    "U.S.A.": "US",
    "UNITED STATES": "US",
    "UNITED STATES OF AMERICA": "US",
    "UK": "GB",
    "U.K.": "GB",
    "UNITED KINGDOM": "GB",
    "GREAT BRITAIN": "GB",
    "ENGLAND": "GB",
    "UAE": "AE",
    "UNITED ARAB EMIRATES": "AE",
    "RUS": "RU",
    "RUSSIA": "RU",
    "RUSSIAN FEDERATION": "RU",
    "IRN": "IR",
    "IRAN": "IR",
    "PRK": "KP",
    "DPRK": "KP",
    "NORTH KOREA": "KP",
    "KOR": "KR",
    "SOUTH KOREA": "KR",
    "MMR": "MM",
    "MYANMAR": "MM",
    "BURMA": "MM",
    "CHN": "CN",
    "CHINA": "CN",
    "BLR": "BY",
    "BELARUS": "BY",
    "SYR": "SY",
    "SYRIA": "SY",
    "CUB": "CU",
    "CUBA": "CU",
    "VEN": "VE",
    "VENEZUELA": "VE",
    "CAYMAN ISLANDS": "KY",
    "CYM": "KY",
    "BVI": "VG",
    "BRITISH VIRGIN ISLANDS": "VG",
    "PANAMA": "PA",
    "PAN": "PA",
    "CYPRUS": "CY",
    "CYP": "CY",
    "DEU": "DE",
    "GERMANY": "DE",
    "FRA": "FR",
    "FRANCE": "FR",
    "IND": "IN",
    "INDIA": "IN",
    "SGP": "SG",
    "SINGAPORE": "SG",
    "CHE": "CH",
    "SWITZERLAND": "CH",
    "CAN": "CA",
    "CANADA": "CA",
    "AUS": "AU",
    "AUSTRALIA": "AU",
    "JPN": "JP",
    "JAPAN": "JP",
    "NLD": "NL",
    "NETHERLANDS": "NL",
    "LUX": "LU",
    "LUXEMBOURG": "LU",
    "IRL": "IE",
    "IRELAND": "IE",
    "HKG": "HK",
    "HONG KONG": "HK",
    "TUR": "TR",
    "TURKEY": "TR",
    "NGA": "NG",
    "NIGERIA": "NG",
    "ZAF": "ZA",
    "SOUTH AFRICA": "ZA",
    "VNM": "VN",
    "VIETNAM": "VN",
    "PHL": "PH",
    "PHILIPPINES": "PH",
    "PAK": "PK",
    "PAKISTAN": "PK",
    "AFG": "AF",
    "AFGHANISTAN": "AF",
    "YEM": "YE",
    "YEMEN": "YE",
    "SDN": "SD",
    "SUDAN": "SD",
    "UKR": "UA",
    "UKRAINE": "UA",
    "BRA": "BR",
    "BRAZIL": "BR",
    "MEX": "MX",
    "MEXICO": "MX",
    "ESP": "ES",
    "SPAIN": "ES",
    "ITA": "IT",
    "ITALY": "IT",
}


def _resolve_file(path_or_repo: str, filename: str, token: str | None) -> Path:
    local = Path(path_or_repo)
    if local.exists():
        return local
    return Path(hf_hub_download(repo_id=path_or_repo, filename=filename, repo_type="dataset", token=token))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_cohere_messages(
    messages: Any,
    *,
    allow_assistant_start: bool = False,
) -> list[dict[str, str]]:
    """Enforce Cohere strict user/assistant alternation after folding any tool turns."""
    if not messages:
        return []
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}] if messages.strip() else []
    folded = fold_tool_messages_into_user(list(messages))
    cleaned: list[dict[str, str]] = []
    for msg in folded:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if role == "assistant" and msg.get("tool_calls") and not str(content or "").strip():
            payload = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                payload.append({"tool_name": fn.get("name"), "parameters": args})
            content = json.dumps(payload, ensure_ascii=False)
        cleaned.append({"role": role, "content": "" if content is None else str(content)})
    merged = _merge_consecutive_same_role(cleaned)
    if not allow_assistant_start:
        if merged and merged[0].get("role") == "assistant":
            merged = [{"role": "user", "content": "Evaluate the KYC compliance case."}] + merged
        if len(merged) >= 2 and merged[0].get("role") == "system" and merged[1].get("role") == "assistant":
            merged = [merged[0], {"role": "user", "content": "Evaluate the KYC compliance case."}] + merged[1:]
    return merged


def _extract_dossier_user_prompt(raw_prompt: Any) -> str:
    """Extract a hashable plain-text dossier prompt without pre-baked duplicate tool headers."""
    if isinstance(raw_prompt, list):
        user_parts = [
            str(m.get("content") or "").strip()
            for m in raw_prompt
            if isinstance(m, dict) and m.get("role") == "user" and str(m.get("content") or "").strip()
        ]
        if user_parts:
            return "\n\n".join(user_parts)
        non_sys = [
            str(m.get("content") or "").strip()
            for m in raw_prompt
            if isinstance(m, dict) and m.get("role") != "system" and str(m.get("content") or "").strip()
        ]
        return "\n\n".join(non_sys)
    text = str(raw_prompt or "").strip()
    marker = "Evaluate this customer onboarding dossier:"
    if marker in text:
        return text[text.index(marker) :].strip()
    if text.startswith("You can call these tools:") and "Strict FinCEN/FATF/OFAC" in text:
        parts = text.split("\n\n")
        if len(parts) > 1:
            return parts[-1].strip()
    return text


def _parse_verdict_from_ref(ref_val: Any) -> str | None:
    if isinstance(ref_val, dict):
        v = ref_val.get("verdict")
        if isinstance(v, str) and v.strip().upper() in {"APPROVE", "REVIEW", "DENY"}:
            return v.strip().upper()
    if isinstance(ref_val, str) and ref_val.strip():
        try:
            parsed = json.loads(ref_val)
            if isinstance(parsed, dict) and isinstance(parsed.get("verdict"), str):
                v = parsed["verdict"].strip().upper()
                if v in {"APPROVE", "REVIEW", "DENY"}:
                    return v
        except json.JSONDecodeError:
            pass
    return None


def _extract_expected_verdict(row: dict[str, Any]) -> str | None:
    for candidate in (
        row.get("expected_verdict"),
        (row.get("metadata") or {}).get("expected_verdict") if isinstance(row.get("metadata"), dict) else None,
        _parse_verdict_from_ref(row.get("reference")),
    ):
        if isinstance(candidate, str) and candidate.strip().upper() in {"APPROVE", "REVIEW", "DENY"}:
            return candidate.strip().upper()
    return None


def _normalize_entity_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower())).strip()


def _fuzzy_name_score(query: str, candidate: str) -> float:
    nq = _normalize_entity_text(query)
    nc = _normalize_entity_text(candidate)
    if not nq or not nc:
        return 0.0
    if nq == nc:
        return 1.0
    seq = SequenceMatcher(None, nq, nc).ratio()
    tq = " ".join(sorted(nq.split()))
    tc = " ".join(sorted(nc.split()))
    tok_sort = SequenceMatcher(None, tq, tc).ratio()
    q_set = set(nq.split())
    c_set = set(nc.split())
    jacc = len(q_set & c_set) / max(len(q_set | c_set), 1)
    return max(seq, tok_sort, 0.88 * jacc if len(q_set & c_set) >= 2 else 0.0)


def _search_sqlite_entities(
    db_path: Path,
    table: str,
    query_name: str,
    fuzzy_threshold: float = 0.80,
) -> list[dict[str, Any]]:
    norm = _normalize_entity_text(query_name)
    if not norm:
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(f"SELECT * FROM {table} WHERE norm_name = ? LIMIT 10", (norm,)).fetchall()
        if rows:
            matches = []
            for r in rows:
                d = dict(r)
                nt = str(d.get("name_type") or "").upper()
                d["match_score"] = 1.0
                d["match_strength"] = "AKA_WEAK" if nt == "AKA_WEAK" else "EXACT"
                matches.append(d)
            return matches

        alias_rows = conn.execute(
            f"SELECT * FROM {table} WHERE norm_aliases LIKE ? OR norm_aliases LIKE ? LIMIT 10",
            (f"%|{norm}|%", f"%{norm}%"),
        ).fetchall()
        exact_alias_hits: list[dict[str, Any]] = []
        for r in alias_rows:
            d = dict(r)
            aliases = [a.strip() for a in str(d.get("norm_aliases") or "").split("|") if a.strip()]
            if norm in aliases or any(norm == a.strip() for a in str(d.get("aliases") or "").lower().split(";")):
                nt = str(d.get("name_type") or "").upper()
                d["match_score"] = 0.96
                d["match_strength"] = "AKA_WEAK" if nt == "AKA_WEAK" else "ALIAS"
                exact_alias_hits.append(d)
        if exact_alias_hits:
            return exact_alias_hits

        all_rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        fuzzy_matches: list[dict[str, Any]] = []
        for r in all_rows:
            d = dict(r)
            best = _fuzzy_name_score(norm, d.get("norm_name") or d.get("name") or "")
            for al in [a.strip() for a in str(d.get("norm_aliases") or "").split("|") if a.strip()]:
                best = max(best, _fuzzy_name_score(norm, al))
            if best >= fuzzy_threshold:
                d["match_score"] = round(best, 4)
                d["match_strength"] = "FUZZY_REVIEW"
                fuzzy_matches.append(d)
        fuzzy_matches.sort(key=lambda x: x["match_score"], reverse=True)
        return fuzzy_matches[:5]
    finally:
        conn.close()


def _validate_dob(dob: str | None) -> str | None:
    if not dob:
        return None
    dob_s = str(dob).strip()
    if not DOB_RE.match(dob_s):
        return f"INVALID_DOB: {dob!r}"
    try:
        datetime.strptime(dob_s, "%Y-%m-%d")
    except ValueError:
        return f"INVALID_DOB_CALENDAR: {dob!r}"
    return None


def _build_sqlite_kyc_tools(db_path: Path | None) -> list[BaseTool]:
    """Self-contained SQLite compliance BaseTools matching src/kyc_tools.py for live rollouts."""

    class ScreenSanctionsTool(BaseTool):
        name = "screen_sanctions"
        description = "Screen an individual or organization against OFAC/UN/EU sanctions lists using uppercase ISO-2 country_iso2."

        def _parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "entity_type", "country_iso2"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "entity_type": {"type": "string", "enum": ["individual", "organization"]},
                    "country_iso2": {"type": "string", "pattern": "^[A-Z]{2}$"},
                    "date_of_birth": {"type": ["string", "null"]},
                },
            }

        def execute(self, name: str, entity_type: str, country_iso2: str, date_of_birth: str | None = None) -> ToolResult:
            if not ISO2_RE.match(str(country_iso2 or "")):
                return ToolResult(False, None, f"INVALID_ISO2: {country_iso2!r}")
            dob_err = _validate_dob(date_of_birth)
            if dob_err:
                return ToolResult(False, None, dob_err)
            if db_path is None or not db_path.exists():
                return ToolResult(
                    True,
                    {
                        "tool": self.name,
                        "matched": False,
                        "status": "NOT_FOUND",
                        "query": {"name": name, "entity_type": entity_type, "country_iso2": country_iso2, "date_of_birth": date_of_birth},
                        "matches": [],
                    },
                )
            matches = _search_sqlite_entities(db_path, "sanctions", name, fuzzy_threshold=0.80)
            return ToolResult(
                True,
                {
                    "tool": self.name,
                    "matched": len(matches) > 0,
                    "status": "MATCH_FOUND" if matches else "NOT_FOUND",
                    "query": {"name": name, "entity_type": entity_type, "country_iso2": country_iso2, "date_of_birth": date_of_birth},
                    "matches": matches,
                },
            )

    class ScreenPepTool(BaseTool):
        name = "screen_pep_adverse_media"
        description = "Screen an individual for PEP status & adverse media using uppercase ISO-2 country_iso2."

        def _parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "country_iso2"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "country_iso2": {"type": "string", "pattern": "^[A-Z]{2}$"},
                    "date_of_birth": {"type": ["string", "null"]},
                },
            }

        def execute(self, name: str, country_iso2: str, date_of_birth: str | None = None) -> ToolResult:
            if not ISO2_RE.match(str(country_iso2 or "")):
                return ToolResult(False, None, f"INVALID_ISO2: {country_iso2!r}")
            dob_err = _validate_dob(date_of_birth)
            if dob_err:
                return ToolResult(False, None, dob_err)
            if db_path is None or not db_path.exists():
                return ToolResult(
                    True,
                    {
                        "tool": self.name,
                        "matched": False,
                        "status": "NOT_FOUND",
                        "query": {"name": name, "country_iso2": country_iso2, "date_of_birth": date_of_birth},
                        "matches": [],
                    },
                )
            matches = _search_sqlite_entities(db_path, "pep_adverse_media", name, fuzzy_threshold=0.82)
            return ToolResult(
                True,
                {
                    "tool": self.name,
                    "matched": len(matches) > 0,
                    "status": "MATCH_FOUND" if matches else "NOT_FOUND",
                    "query": {"name": name, "country_iso2": country_iso2, "date_of_birth": date_of_birth},
                    "matches": matches,
                },
            )

    class CheckFatfTool(BaseTool):
        name = "check_fatf_jurisdiction_risk"
        description = "Check FATF jurisdiction risk & embargo status for an uppercase ISO-2 country_iso2."

        def _parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "additionalProperties": False,
                "required": ["country_iso2"],
                "properties": {"country_iso2": {"type": "string", "pattern": "^[A-Z]{2}$"}},
            }

        def execute(self, country_iso2: str) -> ToolResult:
            if not ISO2_RE.match(str(country_iso2 or "")):
                return ToolResult(False, None, f"INVALID_ISO2: {country_iso2!r}")
            c = str(country_iso2)
            edd = c in FATF_HIGH_RISK
            risk_level = "CRITICAL" if c in (FATF_BLACK_LIST | OFAC_COMPREHENSIVE_EMBARGO) else ("HIGH" if edd else "LOW")
            return ToolResult(True, {"tool": self.name, "country_iso2": c, "risk_level": risk_level, "edd_required": edd})

    class ScreenExportTool(BaseTool):
        name = "screen_export_control"
        description = "Screen an organization against US BIS Entity List & Denied Persons List."

        def _parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "additionalProperties": False,
                "required": ["entity_name", "country_iso2"],
                "properties": {
                    "entity_name": {"type": "string", "minLength": 1},
                    "country_iso2": {"type": "string", "pattern": "^[A-Z]{2}$"},
                },
            }

        def execute(self, entity_name: str, country_iso2: str) -> ToolResult:
            if not ISO2_RE.match(str(country_iso2 or "")):
                return ToolResult(False, None, f"INVALID_ISO2: {country_iso2!r}")
            if db_path is None or not db_path.exists():
                return ToolResult(
                    True,
                    {
                        "tool": self.name,
                        "matched": False,
                        "status": "NOT_FOUND",
                        "query": {"entity_name": entity_name, "country_iso2": country_iso2},
                        "matches": [],
                    },
                )
            matches = _search_sqlite_entities(db_path, "export_control", entity_name, fuzzy_threshold=0.82)
            return ToolResult(
                True,
                {
                    "tool": self.name,
                    "matched": len(matches) > 0,
                    "status": "MATCH_FOUND" if matches else "NOT_FOUND",
                    "query": {"entity_name": entity_name, "country_iso2": country_iso2},
                    "matches": matches,
                },
            )

    class ScreenWalletTool(BaseTool):
        name = "screen_crypto_wallet"
        description = "Screen a crypto wallet address against OFAC SDN digital currency addresses (only when present)."

        def _parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "additionalProperties": False,
                "required": ["wallet_address", "currency"],
                "properties": {
                    "wallet_address": {"type": "string", "minLength": 12},
                    "currency": {"type": "string", "minLength": 2},
                },
            }

        def execute(self, wallet_address: str, currency: str) -> ToolResult:
            addr = wallet_address.strip()
            curr = currency.strip().upper()
            if db_path is None or not db_path.exists():
                return ToolResult(
                    True,
                    {
                        "tool": self.name,
                        "sanctioned": False,
                        "status": "NOT_FOUND",
                        "wallet_address": addr,
                        "currency": curr,
                        "label": None,
                        "category": None,
                    },
                )
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    "SELECT * FROM crypto_wallets WHERE lower(address) = lower(?) LIMIT 1",
                    (addr,),
                ).fetchone()
                d = dict(row) if row else {}
                sanctioned = row is not None
                return ToolResult(
                    True,
                    {
                        "tool": self.name,
                        "sanctioned": sanctioned,
                        "status": "SANCTIONED_WALLET" if sanctioned else "NOT_FOUND",
                        "wallet_address": addr,
                        "currency": d.get("currency", curr) if row else curr,
                        "label": d.get("label") if row else None,
                        "category": d.get("category") if row else None,
                    },
                )
            finally:
                conn.close()

    class VerifySowTool(BaseTool):
        name = "verify_source_of_wealth"
        description = "Verify FinCEN Source-of-Wealth ratio consistency."

        def _parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "customer_name",
                    "annual_income_usd",
                    "declared_assets_usd",
                    "expected_monthly_tx_usd",
                    "source_of_wealth",
                ],
                "properties": {
                    "customer_name": {"type": "string", "minLength": 1},
                    "annual_income_usd": {"type": "number", "minimum": 0},
                    "declared_assets_usd": {"type": "number", "minimum": 0},
                    "expected_monthly_tx_usd": {"type": "number", "minimum": 0},
                    "source_of_wealth": {"type": "string", "minLength": 1},
                    "offshore_jurisdiction": {"type": ["string", "null"]},
                },
            }

        def execute(
            self,
            customer_name: str,
            annual_income_usd: float,
            declared_assets_usd: float,
            expected_monthly_tx_usd: float,
            source_of_wealth: str,
            offshore_jurisdiction: str | None = None,
        ) -> ToolResult:
            del customer_name
            flags: list[str] = []
            income = float(annual_income_usd)
            assets = float(declared_assets_usd)
            monthly_tx = float(expected_monthly_tx_usd)
            offshore = offshore_jurisdiction.strip().upper() if isinstance(offshore_jurisdiction, str) and offshore_jurisdiction.strip() else None
            if offshore and not ISO2_RE.match(offshore):
                return ToolResult(False, None, f"INVALID_OFFSHORE_ISO2: {offshore_jurisdiction!r}")
            if assets > max(income, 1.0) * 15.0 and assets >= 250_000.0:
                flags.append("HIGH_ASSET_TO_INCOME_RATIO")
            if monthly_tx > max(income / 12.0, 1.0) * 6.0 and monthly_tx >= 25_000.0:
                flags.append("EXCESSIVE_MONTHLY_VELOCITY")
            if offshore and offshore in OFFSHORE_SECRECY_HUBS:
                flags.append("OFFSHORE_SECRECY_JURISDICTION")
            sow_l = (source_of_wealth or "").lower()
            if any(tok in sow_l for tok in ("offshore", "shell", "bearer", "unexplained", "cash intensive", "mixer")):
                flags.append("OPAQUE_SOURCE_OF_WEALTH")
            return ToolResult(
                True,
                {
                    "tool": self.name,
                    "consistent": len(flags) == 0,
                    "edd_required": len(flags) > 0,
                    "flags": flags,
                },
            )

    return [
        ScreenSanctionsTool(),
        ScreenExportTool(),
        ScreenPepTool(),
        CheckFatfTool(),
        ScreenWalletTool(),
        VerifySowTool(),
    ]


_KYC_TOOL_SCHEMAS = {t.name: t._parameters() for t in _build_sqlite_kyc_tools(None)}
_PROCESS_EVAL = TrajectoryEvaluator(model_name="none", tool_schemas=_KYC_TOOL_SCHEMAS)
_TOOL_RESULT_USER_RE = re.compile(
    r"^(?:Tool Response(?:\s*\(([^)]+)\))?|Tool result(?:\s+for\s+([^\s:]+))?):\s*(.*)$",
    re.DOTALL | re.IGNORECASE,
)


def _parse_dossier_dict(raw_prompt: Any) -> dict[str, Any]:
    text = _extract_dossier_user_prompt(raw_prompt)
    s_idx, e_idx = text.find("{"), text.rfind("}")
    if s_idx != -1 and e_idx > s_idx:
        try:
            obj = json.loads(text[s_idx : e_idx + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


def _normalize_country_standalone(raw_country: Any) -> str | None:
    if raw_country is None:
        return None
    s = str(raw_country).strip()
    if not s:
        return None
    u = s.upper()
    if ISO2_RE.match(u):
        return u
    return _COUNTRY_ALIAS_TO_ISO2.get(u)


def _compute_tool_policy_standalone(case: dict[str, Any]) -> dict[str, Any]:
    """Self-contained FinCEN CIP + required/forbidden tool policy matching src/kyc_oracle.py."""
    if not case:
        return {"abstain": False, "required_tools": [], "forbidden_tools": []}

    entity_type = str(case.get("entity_type") or "individual").strip().lower()
    name = str(case.get("customer_name") or "").strip()
    norm_country = _normalize_country_standalone(
        case.get("country_iso2") or case.get("country") or case.get("raw_country")
    )
    dob = case.get("date_of_birth")
    id_number = str(case.get("id_number") or "").strip()
    missing = list(case.get("missing_cip_fields") or [])
    if not name:
        missing.append("customer_name")
    if not norm_country:
        missing.append("country_iso2")
    if entity_type == "individual" and (not dob or _validate_dob(str(dob)) is not None):
        missing.append("date_of_birth")
    if not id_number:
        missing.append("id_number")

    if missing:
        return {
            "abstain": True,
            "required_tools": [],
            "forbidden_tools": list(ALL_KYC_TOOL_NAMES),
        }

    wallet = str(case.get("crypto_wallet") or "").strip()
    assets = float(case.get("declared_assets_usd") or 0.0)
    monthly_tx = float(case.get("expected_monthly_tx_usd") or 0.0)
    offshore = _normalize_country_standalone(case.get("offshore_jurisdiction")) or ""

    required = ["screen_sanctions", "screen_pep_adverse_media", "check_fatf_jurisdiction_risk"]
    forbidden: list[str] = []
    if entity_type == "organization" or bool(case.get("check_export_control")):
        required.append("screen_export_control")
    if wallet:
        required.append("screen_crypto_wallet")
    else:
        forbidden.append("screen_crypto_wallet")
    if assets >= 250_000.0 or monthly_tx >= 50_000.0 or offshore in OFFSHORE_SECRECY_HUBS or bool(case.get("trigger_sow_check")):
        required.append("verify_source_of_wealth")

    return {"abstain": False, "required_tools": required, "forbidden_tools": forbidden}


def _extract_trajectory_from_completion(comp: Any) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Extract concatenated assistant text, deduplicated tool_calls, and tool/env outputs."""
    calls: list[dict[str, Any]] = []
    outputs: list[str] = []
    text_parts: list[str] = []

    if isinstance(comp, list):
        for m in comp:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "")
            content = str(m.get("content") or "")
            if role == "assistant":
                if content:
                    text_parts.append(content)
                msg_calls = m.get("tool_calls") or []
                if msg_calls:
                    for tc in msg_calls:
                        fn = tc.get("function") if isinstance(tc, dict) and isinstance(tc.get("function"), dict) else tc
                        if isinstance(fn, dict) and fn.get("name"):
                            calls.append({"name": fn["name"], "arguments": fn.get("arguments", {})})
                else:
                    for ext in extract_tool_calls(None, content) or []:
                        fn = ext.get("function") or ext
                        if isinstance(fn, dict) and fn.get("name"):
                            calls.append({"name": fn["name"], "arguments": fn.get("arguments", {})})
            elif role == "tool" or (role == "user" and _TOOL_RESULT_USER_RE.match(content.strip())):
                outputs.append(content)
    else:
        raw_text = str(comp or "")
        text_parts.append(raw_text)
        for ext in extract_tool_calls(None, raw_text) or []:
            fn = ext.get("function") or ext
            if isinstance(fn, dict) and fn.get("name"):
                calls.append({"name": fn["name"], "arguments": fn.get("arguments", {})})

    return "\n".join(text_parts), calls, outputs


def _validate_tool_call_standalone(tool_name: str, args: Any) -> str | None:
    schema = _KYC_TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        return f"UNKNOWN_TOOL:{tool_name}"
    if not isinstance(args, dict):
        return f"ARGS_NOT_DICT:{tool_name}"
    try:
        jsonschema.validate(instance=args, schema=schema)
    except jsonschema.ValidationError as exc:
        return f"SCHEMA_ERROR:{exc.message}"
    if "country_iso2" in args and not ISO2_RE.match(str(args["country_iso2"])):
        return f"ISO2_ERROR:{args['country_iso2']!r}"
    if args.get("offshore_jurisdiction") and not ISO2_RE.match(str(args["offshore_jurisdiction"]).strip().upper()):
        return f"OFFSHORE_ISO2_ERROR:{args['offshore_jurisdiction']!r}"
    if args.get("date_of_birth"):
        dob_err = _validate_dob(str(args["date_of_birth"]))
        if dob_err:
            return dob_err
    return None


def kyc_compliance_reward(
    completions: list[Any],
    prompts: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Anti-reward-hacking KYC/CIP/ISO-2/tool-coverage/loop/verdict reward for GRPO & rollout DPO."""
    del prompts
    expected_verdicts = list(kwargs.get("expected_verdict") or [])
    references = list(kwargs.get("reference") or [])
    req_lists = list(kwargs.get("required_tools") or [])
    forb_lists = list(kwargs.get("forbidden_tools") or [])
    abstain_flags = list(kwargs.get("abstain") or [])

    schema = {
        "type": "object",
        "required": ["verdict", "reason"],
        "properties": {
            "verdict": {"type": "string", "enum": ["APPROVE", "REVIEW", "DENY"]},
            "reason": {"type": "string", "minLength": 1},
        },
    }
    rewards: list[float] = []
    for idx, comp in enumerate(completions):
        text, calls, outputs = _extract_trajectory_from_completion(comp)
        is_abstain = bool(abstain_flags[idx]) if idx < len(abstain_flags) and abstain_flags[idx] is not None else False
        req_set = set(req_lists[idx] or []) if idx < len(req_lists) else set()
        forb_set = set(forb_lists[idx] or []) if idx < len(forb_lists) else set()

        score = 0.0
        called_valid: set[str] = set()
        seen_signatures: dict[str, int] = {}
        unique_valid_credit = 0

        for c in calls:
            t_name = str(c.get("name") or "")
            args_raw = c.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            sig = f"{t_name}::{json.dumps(args, sort_keys=True, default=str)}"
            seen_signatures[sig] = seen_signatures.get(sig, 0) + 1

            err = _validate_tool_call_standalone(t_name, args)
            if err is not None:
                score -= 0.75
            else:
                called_valid.add(t_name)
                if seen_signatures[sig] == 1 and (not req_set or t_name in req_set):
                    unique_valid_credit += 1

            if t_name in forb_set:
                score -= 0.75

        if is_abstain:
            score += 0.75 if len(calls) == 0 else (-1.0 * len(calls))
        else:
            if len(calls) == 0:
                score -= 1.0
            if req_set:
                covered = len(called_valid & req_set)
                missing_n = len(req_set - called_valid)
                score += 0.80 * (covered / len(req_set)) - 0.65 * missing_n
            cap_n = len(req_set) if req_set else 3
            score += 0.25 * (min(unique_valid_credit, cap_n) / max(cap_n, 1))

        max_rep = max(seen_signatures.values()) if seen_signatures else 0
        if max_rep >= 3:
            score -= 1.0
        elif max_rep == 2:
            score -= 0.35

        budget = max(len(req_set) + 1, 4) if not is_abstain else 0
        if not is_abstain and len(calls) > budget:
            score -= 0.15 * (len(calls) - budget)

        for out_txt in outputs:
            low = out_txt.lower()
            if '"success": false' in low or "'success': false" in low:
                score -= 0.30

        exp_v: str | None = None
        if idx < len(expected_verdicts) and expected_verdicts[idx]:
            exp_v = str(expected_verdicts[idx]).upper()
        elif idx < len(references):
            exp_v = _parse_verdict_from_ref(references[idx])

        called_forbidden = any(str(c.get("name") or "") in forb_set for c in calls)
        tools_satisfied = (
            ((len(calls) == 0) if is_abstain else (len(calls) > 0 and (not req_set or req_set.issubset(called_valid))))
            and (not called_forbidden)
        )
        m_start = text.rfind("{")
        m_end = text.rfind("}")
        if m_start != -1 and m_end > m_start:
            try:
                obj = json.loads(text[m_start : m_end + 1])
                jsonschema.validate(instance=obj, schema=schema)
                score += 0.40
                if exp_v:
                    if str(obj.get("verdict")).upper() == exp_v:
                        score += 1.0 if tools_satisfied else 0.25
                    else:
                        score -= 0.75
            except Exception:
                score -= 0.50
        else:
            score -= 0.50
        rewards.append(round(score, 4))
    return rewards


def kyc_process_reward(
    completions: list[Any],
    prompts: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Deterministic trajectory quality reward (TAC, ARR, LCF, SCSR; zero-tool loophole closed)."""
    del prompts
    abstain_flags = list(kwargs.get("abstain") or [])
    scores: list[float] = []
    for idx, comp in enumerate(completions):
        text, calls, outputs = _extract_trajectory_from_completion(comp)
        is_abstain = bool(abstain_flags[idx]) if idx < len(abstain_flags) and abstain_flags[idx] is not None else False
        if is_abstain and calls:
            scores.append(-1.0)
            continue
        if not calls and not outputs and not is_abstain:
            scores.append(-0.5)
            continue
        traj = {"tool_calls": calls, "tool_outputs": outputs}
        tac = float(_PROCESS_EVAL._calculate_tac(traj))
        arr = float(_PROCESS_EVAL._calculate_arr(traj))
        scsr = float(_PROCESS_EVAL._calculate_scsr(traj))
        lcf = int(_PROCESS_EVAL._calculate_lcf(traj))
        val = 0.5 * tac + 0.3 * scsr - 0.5 * arr - 0.75 * float(lcf)
        if '"verdict"' in text and ("APPROVE" in text or "REVIEW" in text or "DENY" in text):
            val += 0.2
        scores.append(round(val, 4))
    return scores


class _KYCEventLog(EventLog):
    """EventLog subclass preserving Cohere system role, tool call JSON, and alternating turns."""

    def as_dataset_rows(self, fmt: str = "sft") -> list[dict]:
        self._require_full("as_dataset_rows()")
        if fmt != "sft":
            raise ValueError(f"unknown fmt {fmt!r}; supported: 'sft'")

        raw_messages: list[dict[str, Any]] = []
        raw_weights: list[float] = []
        for e in self.events:
            if e.kind in (EventKind.REASONING, EventKind.TEXT):
                raw_messages.append({"role": "assistant", "content": str(e.payload.get("text", ""))})
                raw_weights.append(1.0)
            elif e.kind is EventKind.TOOL_CALL:
                bare_calls = self._unwrap_tool_calls(e.payload.get("action") or {})
                if not bare_calls:
                    act = e.payload.get("action") or {}
                    bare_calls = [act] if isinstance(act, dict) and act.get("name") else []
                if bare_calls:
                    cohere_payload = [
                        {"tool_name": c["name"], "parameters": c.get("arguments", {})}
                        for c in bare_calls
                    ]
                    raw_messages.append(
                        {"role": "assistant", "content": json.dumps(cohere_payload, ensure_ascii=False)}
                    )
                    raw_weights.append(2.0)
            elif e.kind is EventKind.TOOL_RESULT:
                tool_label = e.payload.get("tool")
                folded = fold_tool_messages_into_user(
                    [
                        {
                            "role": "tool",
                            **({"name": tool_label} if tool_label else {}),
                            "content": str(e.payload.get("output", "")),
                        }
                    ]
                )
                for fm in folded:
                    raw_messages.append({"role": "user", "content": str(fm.get("content") or "")})
                    raw_weights.append(0.0)
            elif e.kind is EventKind.OBSERVATION:
                obs_role = str(e.payload.get("role") or "user")
                if obs_role not in {"system", "user", "assistant"}:
                    obs_role = "user"
                raw_messages.append({"role": obs_role, "content": str(e.payload.get("text", ""))})
                raw_weights.append(0.0)

        merged_messages: list[dict[str, str]] = []
        merged_weights: list[float] = []
        for m, w in zip(raw_messages, raw_weights):
            if merged_messages and merged_messages[-1]["role"] == m["role"]:
                prev = merged_messages[-1]["content"]
                curr = m["content"]
                merged_messages[-1]["content"] = f"{prev}\n{curr}".strip() if prev and curr else (prev or curr)
                merged_weights[-1] = max(merged_weights[-1], w)
            else:
                merged_messages.append({"role": m["role"], "content": m["content"]})
                merged_weights.append(w)

        return [
            {
                "messages": merged_messages,
                "segment_weights": merged_weights,
                "loss_mask": [w == 0.0 for w in merged_weights],
            }
        ]


def _records_to_event_logs(rows: list[dict[str, Any]]) -> list[EventLog]:
    """Convert SFT message rows into full-tier EventLogs preserving system, tool calls, and results."""
    logs: list[EventLog] = []
    suffix = "\nUse this result to answer the original question."
    for row in rows:
        msgs = normalize_cohere_messages(list(row.get("messages") or []))
        if not msgs:
            continue
        log = _KYCEventLog(tier="full")
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            if role == "assistant":
                extracted = extract_tool_calls(None, content) or []
                if extracted:
                    for tc in extracted:
                        fn = tc.get("function") or {}
                        args_raw = fn.get("arguments", {})
                        try:
                            args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw or {})
                        except Exception:
                            args = {}
                        log.append(Event(EventKind.TOOL_CALL, {"action": {"name": fn.get("name"), "arguments": args}}))
                else:
                    log.append(Event(EventKind.TEXT, {"text": content}))
            elif role == "user":
                m_tool = _TOOL_RESULT_USER_RE.match(content.strip())
                if m_tool:
                    t_label = (m_tool.group(1) or m_tool.group(2) or "").strip()
                    t_out = m_tool.group(3)
                    if t_out.endswith(suffix):
                        t_out = t_out[: -len(suffix)].rstrip()
                    log.append(
                        Event(
                            EventKind.TOOL_RESULT,
                            {"output": t_out, **({"tool": t_label} if t_label else {})},
                        )
                    )
                else:
                    log.append(Event(EventKind.OBSERVATION, {"text": content, "role": "user"}))
            else:
                log.append(Event(EventKind.OBSERVATION, {"text": content, "role": role}))
        logs.append(log)
    return logs


def build_offline_dpo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare conversational Cohere DPO rows for TrlAgenticDPO(use_rollouts=False)."""
    return [
        {
            "prompt": normalize_cohere_messages(r["prompt"], allow_assistant_start=False),
            "chosen": normalize_cohere_messages(r["chosen"], allow_assistant_start=True),
            "rejected": normalize_cohere_messages(r["rejected"], allow_assistant_start=True),
        }
        for r in rows
        if r.get("prompt") and r.get("chosen") and r.get("rejected") and r.get("chosen") != r.get("rejected")
    ]


def build_rollout_prompt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare hashable str dossier prompts, verdicts, and tool policies for TrlAgenticDPO/Grpo."""
    prepared: list[dict[str, Any]] = []
    for r in rows:
        if not r.get("prompt"):
            continue
        user_prompt = _extract_dossier_user_prompt(r["prompt"])
        dossier = _parse_dossier_dict(user_prompt)
        policy = _compute_tool_policy_standalone(dossier)
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        abstain = r.get("abstain")
        if abstain is None:
            abstain = meta.get("abstain", policy["abstain"])
        req_tools = r.get("required_tools")
        if req_tools is None:
            req_tools = meta.get("required_tools", policy["required_tools"])
        forb_tools = r.get("forbidden_tools")
        if forb_tools is None:
            forb_tools = meta.get("forbidden_tools", policy["forbidden_tools"])
        prepared.append(
            {
                "prompt": user_prompt,
                "expected_verdict": _extract_expected_verdict(r),
                "required_tools": list(req_tools or []),
                "forbidden_tools": list(forb_tools or []),
                "abstain": bool(abstain),
            }
        )
    return prepared


def _cohere_peft_dict(lora_r: int = 16, lora_alpha: int = 32) -> dict[str, Any]:
    return {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": "all-linear",
        "task_type": "CAUSAL_LM",
    }


def _push_and_brand_if_requested(
    output_dir: Path,
    base_model: str,
    mode: str,
    push_repo: str | None,
    merge_before_push: bool,
    token: str | None,
    private: bool = False,
) -> str | None:
    if not push_repo:
        return None
    if merge_before_push:
        base_m = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            token=token,
        )
        peft_m = PeftModel.from_pretrained(base_m, str(output_dir))
        merged = peft_m.merge_and_unload()
        tok = AutoTokenizer.from_pretrained(str(output_dir), token=token)
        return push_model_to_hf(
            model=merged,
            tokenizer=tok,
            repo_id=push_repo,
            base_model=base_model,
            algorithm=mode,
            backend="agenttune-trl",
            private=private,
            token=token,
        )
    url = push_folder_to_hub(str(output_dir), repo_id=push_repo, private=private, token=token)
    brand_hf_repo(
        push_repo,
        kind="adapter",
        base_model=base_model,
        algorithm=mode,
        backend="agenttune-trl",
        private=private,
        token=token,
    )
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-contained AgentTune GPU training script for Cohere Tiny Aya.")
    parser.add_argument(
        "--mode",
        choices=["distill_sft", "agentic_dpo_offline", "agentic_dpo_rollout", "agentic_grpo"],
        default="agentic_dpo_offline",
    )
    parser.add_argument("--base-model", type=str, default=DEFAULT_COHERE_STUDENT)
    parser.add_argument("--dataset", type=str, required=True, help="Local JSONL path or HF Hub dataset repo ID.")
    parser.add_argument("--watchlist-db", type=str, default=None, help="Optional SQLite watchlist DB path/repo for rollouts.")
    parser.add_argument("--output-dir", type=str, default="autoheal_kyc_out")
    parser.add_argument("--push-to-hub", type=str, default=None, help="Optional HF repo ID to push & brand.")
    parser.add_argument("--merge-and-push", action="store_true", help="Merge LoRA weights before pushing.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_fname = {
        "distill_sft": "sft_messages.jsonl",
        "agentic_dpo_offline": "dpo.jsonl",
        "agentic_dpo_rollout": "grpo.jsonl",
        "agentic_grpo": "grpo.jsonl",
    }[args.mode]
    data_path = _resolve_file(args.dataset, default_fname, token)
    rows = _load_jsonl(data_path)
    db_path = (
        _resolve_file(args.watchlist_db, "watchlists.db", token)
        if args.watchlist_db
        else None
    )

    tools = _build_sqlite_kyc_tools(db_path)
    peft_cfg = _cohere_peft_dict(lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    if args.mode == "distill_sft":
        logs = _records_to_event_logs(rows)
        trainer = create_distill_trainer(
            student=args.base_model,
            trajectories=logs,
            output_dir=str(out_dir),
            use_peft=True,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_targets="all-linear",
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
            device_map="auto" if torch.cuda.is_available() else None,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            bf16=torch.cuda.is_available(),
            report_to="none",
        )
        trainer.train()

    elif args.mode == "agentic_dpo_offline":
        ds = Dataset.from_list(build_offline_dpo_rows(rows))
        trainer = TrlAgenticDPO(
            model=args.base_model,
            train_dataset=ds,
            output_dir=str(out_dir),
            beta=args.beta,
            peft_config=peft_cfg,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_length=2048,
            bf16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            report_to="none",
        )
        trainer.train()

    elif args.mode == "agentic_dpo_rollout":
        ds = Dataset.from_list(build_rollout_prompt_rows(rows))
        reward_fn = combine_rewards([kyc_compliance_reward, kyc_process_reward], weights=[0.6, 0.4])
        trainer = TrlAgenticDPO(
            model=args.base_model,
            train_dataset=ds,
            tools=tools,
            reward_funcs=reward_fn,
            use_rollouts=True,
            num_generations=2,
            max_steps_per_turn=6,
            system_prompt=COHERE_ROLLOUT_SYSTEM_RULES,
            output_dir=str(out_dir),
            beta=args.beta,
            peft_config=peft_cfg,
            learning_rate=args.lr,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            bf16=torch.cuda.is_available(),
            report_to="none",
        )
        trainer.train()

    else:  # agentic_grpo
        ds = Dataset.from_list(build_rollout_prompt_rows(rows))
        reward_fn = combine_rewards([kyc_compliance_reward, kyc_process_reward], weights=[0.6, 0.4])
        trainer = TrlAgenticGrpo(
            model=args.base_model,
            reward_funcs=reward_fn,
            tools=tools,
            system_prompt=COHERE_ROLLOUT_SYSTEM_RULES,
            tools_fallback_prompt=COHERE_TOOLS_FALLBACK_PROMPT,
            train_dataset=ds,
            output_dir=str(out_dir),
            peft_config=peft_cfg,
            learning_rate=args.lr,
            max_steps=args.max_steps,
            max_steps_per_turn=6,
            num_generations=4,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_completion_length=512,
            bf16=torch.cuda.is_available(),
            report_to="none",
        )
        trainer.train()

    prov = write_provenance(
        out_dir=out_dir,
        method=args.mode,
        base_model=args.base_model,
        dataset=str(args.dataset),
        params={
            "lora_targets": "all-linear",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "enable_thinking": False,
            "rows": len(rows),
        },
    )
    hub_url = _push_and_brand_if_requested(
        output_dir=out_dir,
        base_model=args.base_model,
        mode=args.mode,
        push_repo=args.push_to_hub,
        merge_before_push=args.merge_and_push,
        token=token,
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "base_model": args.base_model,
                "output_dir": str(out_dir),
                "provenance_schema": prov.get("schema"),
                "hub_url": hub_url,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
