# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "agenttune==1.1.3",
#   "auditkit==1.1.6",
#   "torch>=2.4.0",
#   "transformers>=5.15,<6",
#   "peft>=0.15",
#   "accelerate>=1.0.0",
#   "datasets>=3.0.0",
#   "huggingface_hub>=0.24.0",
#   "jsonschema>=4.20.0",
# ]
# ///
"""Self-contained Hugging Face Jobs GPU evaluation script for Cohere Tiny Aya on Split C + BFCL.

Uses Cohere prompt-mode tool rendering (DEFAULT_TOOLS_FALLBACK_PROMPT, fold_tool_messages_into_user,
_merge_consecutive_same_role, extract_tool_calls), EventLog recording, multi-trial AuditKIT
reliability (pass@k / pass^k), CustomPredicate invariants, LatencyStats/Throughput at
concurrency=1, BFCL simple+multiple evaluation, and AuditKIT push_rows_to_hub.

Strictly self-contained: zero imports from `src.*` or `scripts.*`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
from datetime import datetime
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from agenttune.agentic.distill_trainer import _merge_consecutive_same_role
from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.rollout_engines.rollout_factory import (
    DEFAULT_TOOLS_FALLBACK_PROMPT,
    fold_tool_messages_into_user,
)
from agenttune.agentic.rollout_engines.tool_call_parse import extract_tool_calls
from auditkit import evaluate as ak_evaluate
from auditkit import load_bfcl
from auditkit.adapter import ToolCallAdapter
from auditkit.agent_eval import (
    FAILURE,
    OBSERVED,
    SUCCESS,
    AgentCase,
    AgentEvalRunner,
    AgentEvalSpec,
    CustomPredicate,
    FinalStateAssertion,
    episode_from_eventlog,
    reliability,
)
from auditkit.bfcl import bfcl_arg_match
from auditkit.hf_publish import push_rows_to_hub
from auditkit.metrics.agent import ToolCallF1
from auditkit.metrics.perf import LatencyStats, Throughput
from huggingface_hub import hf_hub_download, snapshot_download
import jsonschema
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_COHERE_STUDENT = "CohereLabs/tiny-aya-global"
ISO2_RE = re.compile(r"^[A-Z]{2}$")
DOB_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VERDICT_KV_RE = re.compile(r'\bverdict["\']?\s*[:=]\s*["\']?(APPROVE|REVIEW|DENY)\b', re.IGNORECASE)

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

COHERE_TOOLS_FALLBACK_PROMPT = DEFAULT_TOOLS_FALLBACK_PROMPT.replace(
    "use it to answer the user's original question as plain text.",
    "use it to answer with the required FinCEN/FATF/OFAC JSON verdict.",
)

_COUNTRY_ALIAS_TO_ISO2 = {
    "USA": "US", "U.S.": "US", "U.S.A.": "US", "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US",
    "UK": "GB", "U.K.": "GB", "UNITED KINGDOM": "GB", "GREAT BRITAIN": "GB", "ENGLAND": "GB",
    "UAE": "AE", "UNITED ARAB EMIRATES": "AE", "RUS": "RU", "RUSSIA": "RU", "RUSSIAN FEDERATION": "RU",
    "IRN": "IR", "IRAN": "IR", "PRK": "KP", "DPRK": "KP", "NORTH KOREA": "KP",
    "KOR": "KR", "SOUTH KOREA": "KR", "MMR": "MM", "MYANMAR": "MM", "BURMA": "MM",
    "CHN": "CN", "CHINA": "CN", "BLR": "BY", "BELARUS": "BY", "SYR": "SY", "SYRIA": "SY",
    "CUB": "CU", "CUBA": "CU", "VEN": "VE", "VENEZUELA": "VE", "CAYMAN ISLANDS": "KY", "CYM": "KY",
    "BVI": "VG", "BRITISH VIRGIN ISLANDS": "VG", "PANAMA": "PA", "PAN": "PA", "CYPRUS": "CY", "CYP": "CY",
    "DEU": "DE", "GERMANY": "DE", "FRA": "FR", "FRANCE": "FR", "IND": "IN", "INDIA": "IN",
    "SGP": "SG", "SINGAPORE": "SG", "CHE": "CH", "SWITZERLAND": "CH", "CAN": "CA", "CANADA": "CA",
    "AUS": "AU", "AUSTRALIA": "AU", "JPN": "JP", "JAPAN": "JP", "NLD": "NL", "NETHERLANDS": "NL",
    "LUX": "LU", "LUXEMBOURG": "LU", "IRL": "IE", "IRELAND": "IE", "HKG": "HK", "HONG KONG": "HK",
    "TUR": "TR", "TURKEY": "TR", "NGA": "NG", "NIGERIA": "NG", "ZAF": "ZA", "SOUTH AFRICA": "ZA",
    "VNM": "VN", "VIETNAM": "VN", "PHL": "PH", "PHILIPPINES": "PH", "PAK": "PK", "PAKISTAN": "PK",
    "AFG": "AF", "AFGHANISTAN": "AF", "YEM": "YE", "YEMEN": "YE", "SDN": "SD", "SUDAN": "SD",
    "UKR": "UA", "UKRAINE": "UA", "BRA": "BR", "BRAZIL": "BR", "MEX": "MX", "MEXICO": "MX",
    "ESP": "ES", "SPAIN": "ES", "ITA": "IT", "ITALY": "IT",
}


def _evaluate_cip_and_policy(case: dict[str, Any]) -> dict[str, Any]:
    """Self-contained FinCEN CIP & tool policy matching src/kyc_oracle.py."""
    def _norm_c(raw: Any) -> str | None:
        if raw is None:
            return None
        s = str(raw).strip().upper()
        if not s:
            return None
        return s if ISO2_RE.match(s) else _COUNTRY_ALIAS_TO_ISO2.get(s)

    entity_type = str(case.get("entity_type") or "individual").strip().lower()
    name = str(case.get("customer_name") or "").strip()
    norm_country = _norm_c(case.get("country_iso2") or case.get("country") or case.get("raw_country"))
    dob = case.get("date_of_birth")
    id_number = str(case.get("id_number") or "").strip()
    missing = list(case.get("missing_cip_fields") or [])
    if not name:
        missing.append("customer_name")
    if not norm_country:
        missing.append("country_iso2")
    if entity_type == "individual":
        dob_ok = bool(dob and DOB_RE.match(str(dob)))
        if dob_ok:
            try:
                datetime.strptime(str(dob), "%Y-%m-%d")
            except ValueError:
                dob_ok = False
        if not dob_ok:
            missing.append("date_of_birth")
    if not id_number:
        missing.append("id_number")

    if missing:
        return {"abstain": True, "required_tools": [], "forbidden_tools": list(TOOL_SCHEMAS.keys())}

    wallet = str(case.get("crypto_wallet") or "").strip()
    assets = float(case.get("declared_assets_usd") or 0.0)
    monthly_tx = float(case.get("expected_monthly_tx_usd") or 0.0)
    offshore = _norm_c(case.get("offshore_jurisdiction")) or ""
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

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "screen_sanctions": {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "entity_type", "country_iso2"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "entity_type": {"type": "string", "enum": ["individual", "organization"]},
            "country_iso2": {"type": "string", "pattern": "^[A-Z]{2}$"},
            "date_of_birth": {"type": ["string", "null"]},
        },
    },
    "screen_export_control": {
        "type": "object",
        "additionalProperties": False,
        "required": ["entity_name", "country_iso2"],
        "properties": {
            "entity_name": {"type": "string", "minLength": 1},
            "country_iso2": {"type": "string", "pattern": "^[A-Z]{2}$"},
        },
    },
    "screen_pep_adverse_media": {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "country_iso2"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "country_iso2": {"type": "string", "pattern": "^[A-Z]{2}$"},
            "date_of_birth": {"type": ["string", "null"]},
        },
    },
    "check_fatf_jurisdiction_risk": {
        "type": "object",
        "additionalProperties": False,
        "required": ["country_iso2"],
        "properties": {"country_iso2": {"type": "string", "pattern": "^[A-Z]{2}$"}},
    },
    "screen_crypto_wallet": {
        "type": "object",
        "additionalProperties": False,
        "required": ["wallet_address", "currency"],
        "properties": {
            "wallet_address": {"type": "string", "minLength": 12},
            "currency": {"type": "string", "minLength": 2},
        },
    },
    "verify_source_of_wealth": {
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
    },
}


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


def _format_dossier_prompt(case: dict[str, Any]) -> str:
    """Format customer onboarding dossier identically to training/curator prompts with zero oracle leaks."""
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


def _parse_verdict_from_reply(reply: str) -> str | None:
    """Strictly parse final compliance verdict; return None on unparseable outputs."""
    if not isinstance(reply, str) or not reply.strip():
        return None
    s_i, e_i = reply.rfind("{"), reply.rfind("}")
    if s_i != -1 and e_i > s_i:
        try:
            obj = json.loads(reply[s_i : e_i + 1])
            if isinstance(obj, dict) and isinstance(obj.get("verdict"), str):
                v = obj["verdict"].strip().upper()
                if v in {"APPROVE", "REVIEW", "DENY"}:
                    return v
        except Exception:
            pass
    m = VERDICT_KV_RE.search(reply)
    if m:
        return m.group(1).upper()
    return None


def validate_tool_args_standalone(tool_name: str, args: dict[str, Any]) -> str | None:
    """Self-contained schema and ISO-2/DOB validation."""
    schema = TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        return f"UNKNOWN_TOOL: {tool_name}"
    try:
        jsonschema.validate(instance=args, schema=schema)
    except jsonschema.ValidationError as exc:
        return f"SCHEMA_VALIDATION_ERROR({tool_name}): {exc.message}"
    if "country_iso2" in args and not ISO2_RE.match(str(args["country_iso2"])):
        return f"ISO2_ERROR: {args['country_iso2']!r}"
    if args.get("offshore_jurisdiction") and not ISO2_RE.match(str(args["offshore_jurisdiction"]).strip().upper()):
        return f"OFFSHORE_ISO2_ERROR: {args['offshore_jurisdiction']!r}"
    if args.get("date_of_birth"):
        dob = str(args["date_of_birth"])
        if not DOB_RE.match(dob):
            return f"DOB_FORMAT_ERROR: {dob!r}"
        try:
            datetime.strptime(dob, "%Y-%m-%d")
        except ValueError:
            return f"DOB_CALENDAR_ERROR: {dob!r}"
    return None


def query_sqlite_tool(db_path: Path, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute validated compliance query against local SQLite watchlists."""
    err = validate_tool_args_standalone(tool_name, args)
    if err is not None:
        return {"success": False, "error": err}

    if tool_name == "check_fatf_jurisdiction_risk":
        c = str(args["country_iso2"])
        edd = c in FATF_HIGH_RISK
        risk_level = (
            "CRITICAL"
            if c in (FATF_BLACK_LIST | OFAC_COMPREHENSIVE_EMBARGO)
            else ("HIGH" if edd else "LOW")
        )
        return {"success": True, "tool": tool_name, "edd_required": edd, "risk_level": risk_level, "country_iso2": c}

    if tool_name == "verify_source_of_wealth":
        assets = float(args["declared_assets_usd"])
        income = float(args["annual_income_usd"])
        monthly_tx = float(args["expected_monthly_tx_usd"])
        offshore_raw = args.get("offshore_jurisdiction")
        offshore = offshore_raw.strip().upper() if isinstance(offshore_raw, str) and offshore_raw.strip() else None
        sow_l = str(args.get("source_of_wealth") or "").lower()
        flags: list[str] = []
        if assets > max(income, 1.0) * 15.0 and assets >= 250_000.0:
            flags.append("HIGH_ASSET_TO_INCOME_RATIO")
        if monthly_tx > max(income / 12.0, 1.0) * 6.0 and monthly_tx >= 25_000.0:
            flags.append("EXCESSIVE_MONTHLY_VELOCITY")
        if offshore and offshore in OFFSHORE_SECRECY_HUBS:
            flags.append("OFFSHORE_SECRECY_JURISDICTION")
        if any(tok in sow_l for tok in ("offshore", "shell", "bearer", "unexplained", "cash intensive", "mixer")):
            flags.append("OPAQUE_SOURCE_OF_WEALTH")
        return {"success": True, "tool": tool_name, "consistent": len(flags) == 0, "edd_required": len(flags) > 0, "flags": flags}

    if tool_name == "screen_crypto_wallet":
        addr = str(args["wallet_address"]).strip()
        curr = str(args["currency"]).strip().upper()
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM crypto_wallets WHERE lower(address) = lower(?) LIMIT 1",
                (addr,),
            ).fetchone()
            d = dict(row) if row else {}
            sanctioned = row is not None
            return {
                "success": True,
                "tool": tool_name,
                "sanctioned": sanctioned,
                "status": "SANCTIONED_WALLET" if sanctioned else "NOT_FOUND",
                "wallet_address": addr,
                "currency": d.get("currency", curr) if row else curr,
                "label": d.get("label") if row else None,
                "category": d.get("category") if row else None,
            }
        finally:
            conn.close()

    table_and_thresh = {
        "screen_sanctions": ("sanctions", 0.80),
        "screen_export_control": ("export_control", 0.82),
        "screen_pep_adverse_media": ("pep_adverse_media", 0.82),
    }[tool_name]
    table, thresh = table_and_thresh
    name_val = str(args.get("name") or args.get("entity_name") or "")
    matches = _search_sqlite_entities(db_path, table, name_val, fuzzy_threshold=thresh)
    return {
        "success": True,
        "tool": tool_name,
        "matched": len(matches) > 0,
        "status": "MATCH_FOUND" if matches else "NOT_FOUND",
        "query": dict(args),
        "matches": matches,
    }


def format_messages_for_cohere(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Fold role='tool' into user turns and merge same-role adjacent turns for Cohere templates."""
    folded = fold_tool_messages_into_user(messages)
    cleaned = [
        {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")}
        for m in folded
    ]
    merged = _merge_consecutive_same_role(cleaned)
    if merged and merged[0].get("role") == "assistant":
        merged = [{"role": "user", "content": "Evaluate the KYC dossier."}] + merged
    if len(merged) >= 2 and merged[0].get("role") == "system" and merged[1].get("role") == "assistant":
        merged = [merged[0], {"role": "user", "content": "Evaluate the KYC dossier."}] + merged[1:]
    return merged


def render_and_generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int = 384,
    temperature: float = 0.0,
) -> tuple[str, int]:
    """Render Cohere prompt-mode conversation cleanly and generate response + token count."""
    prepared = format_messages_for_cohere(messages)
    try:
        text = tokenizer.apply_chat_template(
            prepared,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(prepared, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature
    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)
    gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip(), int(len(gen_ids))


def resolve_input_file(path_or_repo: str, filename: str | None, token: str | None) -> Path:
    local = Path(path_or_repo)
    if local.exists():
        return local
    assert filename is not None
    return Path(hf_hub_download(repo_id=path_or_repo, filename=filename, repo_type="dataset", token=token))


def _evaluate_bfcl(
    model: Any,
    tokenizer: Any,
    bfcl_repo: str,
    bfcl_revision: str,
    limit: int = 20,
    token: str | None = None,
) -> dict[str, Any]:
    """Run AuditKIT ToolCallAdapter(mode='prompt') + ToolCallF1(arg_match=bfcl_arg_match) on BFCL."""
    snap = Path(
        snapshot_download(
            bfcl_repo,
            repo_type="dataset",
            revision=bfcl_revision,
            allow_patterns=[
                "BFCL_v3_simple.json",
                "possible_answer/BFCL_v3_simple.json",
                "BFCL_v3_multiple.json",
                "possible_answer/BFCL_v3_multiple.json",
            ],
            token=token,
        )
    )
    samples = []
    for cat in ("simple", "multiple"):
        qf = snap / f"BFCL_v3_{cat}.json"
        af = snap / "possible_answer" / f"BFCL_v3_{cat}.json"
        if qf.exists() and af.exists():
            samples.extend(load_bfcl(str(qf), answers=str(af), revision=bfcl_revision, limit=max(1, limit // 2)))

    def _model_fn(prompts: list[Any]) -> list[str]:
        outputs: list[str] = []
        for p in prompts:
            msgs = p if isinstance(p, list) else [{"role": "user", "content": str(p)}]
            reply, _ = render_and_generate(model, tokenizer, msgs, max_new_tokens=256, temperature=0.0)
            outputs.append(reply)
        return outputs

    res = ak_evaluate(
        samples[:limit],
        model=_model_fn,
        adapter=ToolCallAdapter(mode="prompt"),
        scorers=[ToolCallF1(arg_match=bfcl_arg_match)],
    )
    f1_key = next(
        (k for k in res.headline if k == "tool_call_f1" or k.startswith("tool_call_f1_")),
        None,
    )
    return {
        "num_samples": len(samples[:limit]),
        "f1_metric_key": f1_key or "tool_call_f1",
        "tool_call_f1": round(float(res.headline.get(f1_key, 0.0) if f1_key else 0.0), 4),
        "stats": res.stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-contained L4 evaluation job for Cohere AutoHeal-KYC.")
    parser.add_argument("--base-model", type=str, default=DEFAULT_COHERE_STUDENT)
    parser.add_argument("--adapter-repo", type=str, default=None, help="Optional LoRA adapter path or HF repo.")
    parser.add_argument("--split-c", type=str, required=True, help="Local split_c.jsonl or HF repo ID.")
    parser.add_argument("--watchlist-db", type=str, required=True, help="Local watchlists.db or HF repo ID.")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials for pass@k / pass^k.")
    parser.add_argument("--concurrency", type=int, default=1, help="Fixed request concurrency (default 1).")
    parser.add_argument("--bfcl-repo", type=str, default="gorilla-llm/Berkeley-Function-Calling-Leaderboard")
    parser.add_argument("--bfcl-revision", type=str, default="61fc0608cfd831fcfbbaa676ebdfef0ed963eeda")
    parser.add_argument("--bfcl-limit", type=int, default=20)
    parser.add_argument("--skip-bfcl", action="store_true")
    parser.add_argument("--push-eval-repo", type=str, default=None, help="Optional HF dataset repo to publish eval rows.")
    parser.add_argument("--output-json", type=str, default="eval_results.json")
    parser.add_argument("--max-cases", type=int, default=60)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    split_path = resolve_input_file(args.split_c, "split_c.jsonl", token)
    db_path = resolve_input_file(args.watchlist_db, "watchlists.db", token)

    cases = [
        json.loads(line)
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.max_cases]

    tok = AutoTokenizer.from_pretrained(args.base_model, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        token=token,
    )
    if args.adapter_repo:
        model = PeftModel.from_pretrained(model, args.adapter_repo, token=token)
    model.eval()

    system_tools_json = json.dumps(
        [{"name": name, "description": name, "parameters": schema} for name, schema in TOOL_SCHEMAS.items()],
        indent=2,
    )
    system_prompt = (
        COHERE_TOOLS_FALLBACK_PROMPT.format(tools=system_tools_json)
        + "\n\nStrict FinCEN/FATF/OFAC Rules: Use uppercase ISO-2 codes (^[A-Z]{2}$), YYYY-MM-DD DOBs, "
        + "abstain from screening on incomplete CIP, and conclude with JSON "
        + '{"verdict": "APPROVE"|"REVIEW"|"DENY", "reason": "..."}.'
    )

    lat_stats = LatencyStats()
    tp_stats = Throughput()
    total_gen_tokens = 0
    per_case_verdicts: dict[str, list[str]] = {}
    published_rows: list[dict[str, Any]] = []
    n_trials = max(1, int(args.trials))

    episodes = []
    agent_cases = []
    trial_1_correct = 0
    trial_1_first_try_ok = 0
    trial_1_loop = 0
    trial_1_parse_errors = 0
    trial_1_abstain_total = 0
    trial_1_abstain_ok = 0
    trial_1_req_cov_ok = 0

    for t_idx in range(n_trials):
        temp = 0.0 if t_idx == 0 else 0.2
        for case in cases:
            cid = str(case.get("case_id", ""))
            policy = _evaluate_cip_and_policy(case)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _format_dossier_prompt(case)},
            ]
            event_log = EventLog(tier="full")
            event_log.append(Event(EventKind.OBSERVATION, {"text": messages[1]["content"]}))
            repeat_counter: Counter[str] = Counter()
            executed_ok_tools: set[str] = set()
            executed_all_tools: list[str] = []
            had_tool_err = False
            is_loop = False
            verdict: str | None = None
            final_reply = ""
            t0 = time.perf_counter()

            for step_i in range(6):
                reply, n_tok = render_and_generate(model, tok, messages, temperature=temp)
                total_gen_tokens += n_tok
                final_reply = reply
                messages.append({"role": "assistant", "content": reply})

                calls = extract_tool_calls(None, reply) or []
                if not calls:
                    verdict = _parse_verdict_from_reply(reply)
                    event_log.append(Event(EventKind.TURN_COMPLETE, {"step": step_i + 1}))
                    break

                for idx_c, tc in enumerate(calls):
                    fn = tc.get("function") or {}
                    t_name = str(fn.get("name") or "")
                    executed_all_tools.append(t_name)
                    raw_args = fn.get("arguments", "{}")
                    try:
                        t_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                    except Exception:
                        t_args = {}
                    tc_id = f"call_{t_idx}_{step_i}_{idx_c}"
                    call_key = f"{t_name}::{json.dumps(t_args, sort_keys=True)}"
                    repeat_counter[call_key] += 1
                    if repeat_counter[call_key] >= 3:
                        is_loop = True
                    event_log.append(Event(EventKind.TOOL_CALL, {"action": {"name": t_name, "arguments": t_args, "id": tc_id}}))
                    res = query_sqlite_tool(db_path, t_name, t_args)
                    if res.get("success"):
                        executed_ok_tools.add(t_name)
                    else:
                        had_tool_err = True
                    event_log.append(
                        Event(
                            EventKind.TOOL_RESULT,
                            {"output": res, "tool": t_name, "call_id": tc_id, "success": res.get("success"), "error": res.get("error")},
                        )
                    )
                    messages.append({"role": "tool", "name": t_name, "content": json.dumps(res)})
                event_log.append(Event(EventKind.TURN_COMPLETE, {"step": step_i + 1}))
                if is_loop:
                    break

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            lat_stats.record(elapsed_ms)
            tp_stats.record(elapsed_ms, num_requests=1)

            abstain_ok = (len(executed_all_tools) == 0) if policy["abstain"] else True
            forbidden_called = any(t in policy["forbidden_tools"] for t in executed_all_tools)
            req_covered = set(policy["required_tools"]).issubset(executed_ok_tools)
            policy_ok = abstain_ok and (not forbidden_called) and req_covered

            predicted_verdict = verdict if verdict is not None else "PARSE_ERROR"
            event_log.append(Event(EventKind.TEXT, {"text": final_reply}))
            expected_v = str(case.get("expected_verdict") or "REVIEW")
            passed = (predicted_verdict == expected_v) and (not had_tool_err) and (not is_loop) and policy_ok
            per_case_verdicts.setdefault(cid, []).append(SUCCESS if passed else FAILURE)

            if t_idx == 0:
                if predicted_verdict == "PARSE_ERROR":
                    trial_1_parse_errors += 1
                if predicted_verdict == expected_v:
                    trial_1_correct += 1
                if (not had_tool_err) and abstain_ok and (not forbidden_called):
                    trial_1_first_try_ok += 1
                if is_loop:
                    trial_1_loop += 1
                if policy["abstain"]:
                    trial_1_abstain_total += 1
                    if abstain_ok and predicted_verdict == "REVIEW":
                        trial_1_abstain_ok += 1
                if req_covered and abstain_ok:
                    trial_1_req_cov_ok += 1
                ep = episode_from_eventlog(event_log, source_id=f"{cid}_t0")
                ep = dataclasses.replace(
                    ep,
                    case_id=cid,
                    final_state={"verdict": predicted_verdict, "loop_collapse": is_loop, "policy_ok": policy_ok},
                    coverage={**ep.coverage, "final_state": OBSERVED},
                )
                episodes.append(ep)
                agent_cases.append(
                    AgentCase(
                        id=cid,
                        task=f"Screen {case.get('customer_name')}",
                        outcome=FinalStateAssertion(key="verdict", equals=expected_v),
                    )
                )
                published_rows.append(
                    {
                        "case_id": cid,
                        "expected_verdict": expected_v,
                        "predicted_verdict": predicted_verdict,
                        "first_try_ok": (not had_tool_err) and abstain_ok and (not forbidden_called),
                        "policy_ok": policy_ok,
                        "loop_collapse": is_loop,
                        "latency_ms": round(elapsed_ms, 2),
                    }
                )

    loop_free_pred = CustomPredicate(
        lambda ep, c=None: not bool((getattr(ep, "final_state", None) or {}).get("loop_collapse")),
        version="kyc_v1",
        name="loop_free_invariant",
    )
    ak_res = AgentEvalRunner().run(
        AgentEvalSpec(
            cases=agent_cases,
            mode="recorded",
            episodes=episodes,
            scorers=["agent_loop_detection", "tool_call_validity"],
            name="hf-job-kyc-eval",
        )
    )
    rel_per_case = {
        cid: reliability(vs, k=n_trials, independent=True, n_requested=n_trials)
        for cid, vs in per_case_verdicts.items()
    }
    pass_at_k = [float(v["pass_at_k"]) for v in rel_per_case.values() if "pass_at_k" in v]
    pass_hat_k = [
        float(v["all_k"] if "all_k" in v else v["pass_hat_k"])
        for v in rel_per_case.values()
        if ("all_k" in v or "pass_hat_k" in v)
    ]

    bfcl_summary = None
    if not args.skip_bfcl:
        bfcl_summary = _evaluate_bfcl(
            model,
            tok,
            bfcl_repo=args.bfcl_repo,
            bfcl_revision=args.bfcl_revision,
            limit=args.bfcl_limit,
            token=token,
        )

    n = max(len(cases), 1)
    summary_attr = getattr(ak_res, "summary", None)
    auditkit_summary = summary_attr() if callable(summary_attr) else summary_attr
    report = {
        "model": args.adapter_repo or args.base_model,
        "base_model": args.base_model,
        "tool_mode": "prompt",
        "concurrency": args.concurrency,
        "trials": n_trials,
        "num_cases": len(cases),
        "verdict_accuracy": round(trial_1_correct / n, 4),
        "parse_error_rate": round(trial_1_parse_errors / n, 4),
        "first_try_tool_accuracy": round(trial_1_first_try_ok / n, 4),
        "abstain_accuracy": round(trial_1_abstain_ok / max(trial_1_abstain_total, 1), 4),
        "required_tool_coverage_rate": round(trial_1_req_cov_ok / n, 4),
        "loop_collapse_rate": round(trial_1_loop / n, 4),
        "loop_free_predicate_rate": round(
            sum(1 for ep, ac in zip(episodes, agent_cases) if loop_free_pred.evaluate(ep, ac).verdict == SUCCESS) / n,
            4,
        ),
        "reliability": {
            f"mean_pass_at_{n_trials}": round(sum(pass_at_k) / len(pass_at_k), 4) if pass_at_k else None,
            f"mean_pass_hat_{n_trials}": round(sum(pass_hat_k) / len(pass_hat_k), 4) if pass_hat_k else None,
        },
        "perf": {
            "concurrency": args.concurrency,
            "latency_ms": lat_stats.stats(),
            "throughput": tp_stats.stats(),
            "total_generated_tokens": total_gen_tokens,
        },
        "auditkit_summary": auditkit_summary,
        "bfcl_general_regression": bfcl_summary,
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.push_eval_repo:
        push_rows_to_hub(
            published_rows,
            repo_id=args.push_eval_repo,
            token=token,
            kind="run",
            method="agent-eval",
            model=args.adapter_repo or args.base_model,
            provenance=report,
        )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
