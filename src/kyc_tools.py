"""Schema-enforced KYC/AML compliance tools backed by local SQLite watchlists.

Implements 6 deterministic AgentTune BaseTool instances:
1. screen_sanctions            (US OFAC SDN/Consolidated, UN SC, EU FSF)
2. screen_export_control       (US BIS Entity List & Denied Persons List)
3. screen_pep_adverse_media    (Global PEP registry + adverse media + DOB disambiguation)
4. check_fatf_jurisdiction_risk (FATF black/grey lists, embargoes, offshore hubs)
5. screen_crypto_wallet        (OFAC SDN sanctioned crypto addresses)
6. verify_source_of_wealth     (FinCEN CDD/EDD SoW vs assets/velocity ratio checks)

All tools enforce strict jsonschema validation, uppercase ISO-2 country codes
(^[A-Z]{2}$), and calendar-valid YYYY-MM-DD DOB formats without silent coercion.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

import jsonschema

from agenttune.agentic.tools.base import BaseTool, ToolResult
from agenttune.agentic.tools.registry import ToolRegistry
from agenttune.decide.closed_loop.tool_isolation import IsolatedTool, isolate_tools

from src.kyc_oracle import (
    evaluate_jurisdiction_risk,
    evaluate_source_of_wealth_logic,
    is_valid_dob,
    is_valid_iso2,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATCHLIST_DB = PROJECT_ROOT / "data" / "watchlists.db"

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_COMMON_CORP_SUFFIX_RE = re.compile(
    r"\b(ltd|limited|llc|inc|incorporated|corp|corporation|co|gmbh|sa|ag|plc|pte|bv|nv|fzco|fze)\b"
)


def normalize_entity_text(value: str) -> str:
    """Normalize an entity or person name for deterministic & fuzzy watchlist lookup."""
    text = _NON_ALNUM_RE.sub(" ", (value or "").lower())
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def strip_corporate_suffixes(norm_text: str) -> str:
    """Remove trailing/common legal entity designators for entity resolution scoring."""
    stripped = _COMMON_CORP_SUFFIX_RE.sub(" ", norm_text)
    return _MULTI_SPACE_RE.sub(" ", stripped).strip()


def fuzzy_name_score(query: str, candidate: str) -> float:
    """Compute a bounded [0, 1] similarity combining full and suffix-stripped tokens."""
    q_norm = normalize_entity_text(query)
    c_norm = normalize_entity_text(candidate)
    if not q_norm or not c_norm:
        return 0.0
    if q_norm == c_norm:
        return 1.0
    raw_ratio = SequenceMatcher(None, q_norm, c_norm).ratio()
    q_strip = strip_corporate_suffixes(q_norm)
    c_strip = strip_corporate_suffixes(c_norm)
    if q_strip and q_strip == c_strip:
        return max(raw_ratio, 0.96)
    strip_ratio = SequenceMatcher(None, q_strip, c_strip).ratio() if (q_strip and c_strip) else 0.0
    q_tokens = set(q_strip.split())
    c_tokens = set(c_strip.split())
    token_overlap = (
        len(q_tokens & c_tokens) / max(len(q_tokens | c_tokens), 1)
        if (q_tokens and c_tokens)
        else 0.0
    )
    return max(raw_ratio, 0.7 * strip_ratio + 0.3 * token_overlap)


def _connect_db(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Watchlist SQLite DB not found at {path}. Run scripts/mock_watchlists.py first."
        )
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def _candidate_tokens(query: str) -> list[str]:
    norm = strip_corporate_suffixes(normalize_entity_text(query)) or normalize_entity_text(query)
    tokens = [tok for tok in norm.split() if len(tok) >= 3]
    tokens.sort(key=len, reverse=True)
    return tokens[:3]


def _parse_json_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                return [str(x) for x in loaded if x]
        except json.JSONDecodeError:
            return [part.strip() for part in raw.split(";") if part.strip()]
    return []


def validate_strict_tool_args(tool_name: str, schema: dict[str, Any], arguments: Any) -> str | None:
    """Validate arguments strictly against jsonschema plus ISO-2 and DOB invariants."""
    if not isinstance(arguments, dict):
        return f"SCHEMA_VALIDATION_ERROR({tool_name}): arguments must be a JSON object"
    try:
        jsonschema.validate(instance=arguments, schema=schema)
    except jsonschema.ValidationError as exc:
        return f"SCHEMA_VALIDATION_ERROR({tool_name}): {exc.message}"

    if "country_iso2" in arguments:
        c = arguments["country_iso2"]
        if not is_valid_iso2(c):
            return (
                f"ISO2_COUNTRY_ERROR({tool_name}): country_iso2={c!r} must be a valid "
                "uppercase 2-letter ISO-3166-1 alpha-2 code (e.g. 'US', 'GB'), never 'USA' or 'UK'."
            )

    offshore = arguments.get("offshore_jurisdiction")
    if offshore is not None and offshore != "":
        if not is_valid_iso2(offshore):
            return (
                f"ISO2_COUNTRY_ERROR({tool_name}): offshore_jurisdiction={offshore!r} "
                "must be a valid uppercase ISO-2 code or null."
            )

    dob = arguments.get("date_of_birth")
    if dob is not None:
        if not isinstance(dob, str) or not is_valid_dob(dob):
            return (
                f"DOB_FORMAT_ERROR({tool_name}): date_of_birth={dob!r} must be calendar-valid "
                "YYYY-MM-DD or null."
            )

    if tool_name == "screen_crypto_wallet":
        wallet = str(arguments.get("wallet_address") or "").strip()
        if len(wallet) < 12 or " " in wallet:
            return f"WALLET_FORMAT_ERROR({tool_name}): invalid wallet_address={wallet!r}."

    return None


def _search_sqlite_entities(
    db_path: str | Path,
    table: str,
    query_name: str,
    fuzzy_threshold: float = 0.80,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Query exact and token-shortlisted fuzzy matches from a SQLite entity watchlist table."""
    q_norm = normalize_entity_text(query_name)
    if not q_norm:
        return []

    tokens = _candidate_tokens(query_name)
    conn = _connect_db(db_path)
    try:
        rows_by_id: dict[int, sqlite3.Row] = {}
        exact_sql = f"SELECT * FROM {table} WHERE norm_name = ? LIMIT 25"
        for row in conn.execute(exact_sql, (q_norm,)):
            rows_by_id[int(row["id"])] = row

        for tok in tokens:
            like_pat = f"%{tok}%"
            cand_sql = (
                f"SELECT * FROM {table} "
                "WHERE norm_name LIKE ? OR norm_aliases LIKE ? "
                "LIMIT 80"
            )
            for row in conn.execute(cand_sql, (like_pat, like_pat)):
                rows_by_id[int(row["id"])] = row
                if len(rows_by_id) >= 160:
                    break
            if len(rows_by_id) >= 160:
                break
    finally:
        conn.close()

    scored: list[dict[str, Any]] = []
    for row in rows_by_id.values():
        row_dict = dict(row)
        cand_names = [row_dict.get("name") or ""] + _parse_json_list(row_dict.get("aliases_json"))
        best_score = 0.0
        matched_via = row_dict.get("name") or ""
        for cand in cand_names:
            sc = fuzzy_name_score(query_name, cand)
            if sc > best_score:
                best_score = sc
                matched_via = cand

        if best_score < fuzzy_threshold:
            continue

        raw_strength = str(row_dict.get("name_type") or "PRIMARY").upper()
        if raw_strength == "AKA_WEAK":
            match_strength = "AKA_WEAK"
        elif best_score >= 0.94:
            match_strength = raw_strength if raw_strength in {"PRIMARY", "AKA_STRONG"} else "STRONG"
        else:
            match_strength = "FUZZY_REVIEW"

        scored.append(
            {
                "id": row_dict.get("id"),
                "name": row_dict.get("name"),
                "matched_alias": matched_via,
                "score": round(best_score, 4),
                "match_strength": match_strength,
                "name_type": row_dict.get("name_type") or match_strength,
                "entity_type": row_dict.get("entity_type") or "unknown",
                "source": row_dict.get("source") or table.upper(),
                "programs": _parse_json_list(row_dict.get("programs_json")),
                "countries": _parse_json_list(row_dict.get("countries_json")),
                "birth_date": row_dict.get("birth_date"),
                "is_pep": bool(row_dict.get("is_pep", 0)),
                "pep_tier": row_dict.get("pep_tier"),
                "pep_body": row_dict.get("pep_body"),
                "adverse_media": bool(row_dict.get("adverse_media", 0)),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


class KYCBaseTool(BaseTool):
    """Shared BaseTool mixin adding strict JSONSchema + ISO-2 + DOB validation."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or DEFAULT_WATCHLIST_DB)
        super().__init__()

    def validate_args(self, kwargs: dict[str, Any]) -> str | None:
        return validate_strict_tool_args(self.name, self._parameters(), kwargs)

    def _run_validated(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    def execute(self, **kwargs: Any) -> ToolResult:
        err = self.validate_args(kwargs)
        if err is not None:
            return ToolResult(success=False, output="", error=err, metadata={"validation_failed": True})
        try:
            return self._run_validated(**kwargs)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return ToolResult(
                success=False,
                output="",
                error=f"TOOL_EXECUTION_ERROR({self.name}): {exc}",
                metadata={"exception_type": type(exc).__name__},
            )


class ScreenSanctionsTool(KYCBaseTool):
    """Screen a person or organization against US OFAC SDN, UN SC, and EU FSF watchlists."""

    @property
    def name(self) -> str:
        return "screen_sanctions"

    @property
    def description(self) -> str:
        return (
            "Screen an individual or organization against OFAC SDN/Consolidated, UN Security Council, "
            "and EU Financial Sanctions lists. Requires uppercase ISO-2 country_iso2 (e.g. 'US', 'GB')."
        )

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

    def _run_validated(
        self,
        name: str,
        entity_type: str,
        country_iso2: str,
        date_of_birth: str | None = None,
    ) -> ToolResult:
        matches = _search_sqlite_entities(self.db_path, "sanctions", name, fuzzy_threshold=0.80)
        payload = {
            "tool": self.name,
            "matched": len(matches) > 0,
            "status": "MATCH_FOUND" if matches else "NOT_FOUND",
            "query": {
                "name": name,
                "entity_type": entity_type,
                "country_iso2": country_iso2,
                "date_of_birth": date_of_birth,
            },
            "matches": matches,
        }
        return ToolResult(
            success=True,
            output=json.dumps(payload),
            metadata={"matched": len(matches) > 0, "match_count": len(matches)},
        )


class ScreenExportControlTool(KYCBaseTool):
    """Screen entities against US BIS Entity List and Denied Persons List."""

    @property
    def name(self) -> str:
        return "screen_export_control"

    @property
    def description(self) -> str:
        return (
            "Screen an organization or counterparty against US BIS Entity List and Denied Persons List. "
            "Requires uppercase ISO-2 country_iso2."
        )

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

    def _run_validated(self, entity_name: str, country_iso2: str) -> ToolResult:
        matches = _search_sqlite_entities(self.db_path, "export_control", entity_name, fuzzy_threshold=0.82)
        payload = {
            "tool": self.name,
            "matched": len(matches) > 0,
            "status": "MATCH_FOUND" if matches else "NOT_FOUND",
            "query": {"entity_name": entity_name, "country_iso2": country_iso2},
            "matches": matches,
        }
        return ToolResult(
            success=True,
            output=json.dumps(payload),
            metadata={"matched": len(matches) > 0, "match_count": len(matches)},
        )


class ScreenPepAdverseMediaTool(KYCBaseTool):
    """Screen individuals against global PEP registries and adverse-media flags."""

    @property
    def name(self) -> str:
        return "screen_pep_adverse_media"

    @property
    def description(self) -> str:
        return (
            "Screen an individual for Politically Exposed Person (PEP) status and adverse media "
            "using uppercase ISO-2 country_iso2 and optional YYYY-MM-DD date_of_birth for disambiguation."
        )

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

    def _run_validated(
        self,
        name: str,
        country_iso2: str,
        date_of_birth: str | None = None,
    ) -> ToolResult:
        matches = _search_sqlite_entities(self.db_path, "pep_adverse_media", name, fuzzy_threshold=0.82)
        payload = {
            "tool": self.name,
            "matched": len(matches) > 0,
            "status": "MATCH_FOUND" if matches else "NOT_FOUND",
            "query": {"name": name, "country_iso2": country_iso2, "date_of_birth": date_of_birth},
            "matches": matches,
        }
        return ToolResult(
            success=True,
            output=json.dumps(payload),
            metadata={"matched": len(matches) > 0, "match_count": len(matches)},
        )


class CheckFatfJurisdictionRiskTool(KYCBaseTool):
    """Evaluate FATF blacklist, greylist, embargo, and offshore secrecy jurisdiction risk."""

    @property
    def name(self) -> str:
        return "check_fatf_jurisdiction_risk"

    @property
    def description(self) -> str:
        return (
            "Check FATF High-Risk (Black List), Increased Monitoring (Grey List), OFAC embargo, "
            "and offshore secrecy risk for an uppercase ISO-2 country_iso2 code."
        )

    def _parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["country_iso2"],
            "properties": {
                "country_iso2": {"type": "string", "pattern": "^[A-Z]{2}$"},
            },
        }

    def _run_validated(self, country_iso2: str) -> ToolResult:
        result = evaluate_jurisdiction_risk(country_iso2)
        result["tool"] = self.name
        return ToolResult(
            success=True,
            output=json.dumps(result),
            metadata={"risk_level": result["risk_level"], "edd_required": result["edd_required"]},
        )


class ScreenCryptoWalletTool(KYCBaseTool):
    """Screen a cryptocurrency wallet address against OFAC SDN digital currency addresses."""

    @property
    def name(self) -> str:
        return "screen_crypto_wallet"

    @property
    def description(self) -> str:
        return (
            "Screen a cryptocurrency wallet address against OFAC SDN digital currency addresses. "
            "Invoke ONLY when the customer dossier contains a crypto_wallet address."
        )

    def _parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["wallet_address", "currency"],
            "properties": {
                "wallet_address": {"type": "string", "minLength": 12},
                "currency": {"type": "string", "minLength": 2, "maxLength": 12},
            },
        }

    def _run_validated(self, wallet_address: str, currency: str) -> ToolResult:
        addr_clean = wallet_address.strip()
        curr_clean = currency.strip().upper()
        conn = _connect_db(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM crypto_wallets WHERE lower(address) = lower(?) LIMIT 1",
                (addr_clean,),
            ).fetchone()
        finally:
            conn.close()

        sanctioned = row is not None
        payload = {
            "tool": self.name,
            "sanctioned": sanctioned,
            "status": "SANCTIONED_WALLET" if sanctioned else "NOT_FOUND",
            "wallet_address": addr_clean,
            "currency": dict(row).get("currency", curr_clean) if row else curr_clean,
            "label": dict(row).get("label") if row else None,
            "category": dict(row).get("category") if row else None,
        }
        return ToolResult(
            success=True,
            output=json.dumps(payload),
            metadata={"sanctioned": sanctioned},
        )


class VerifySourceOfWealthTool(KYCBaseTool):
    """Evaluate FinCEN CDD/EDD Source-of-Wealth consistency against assets & velocity."""

    @property
    def name(self) -> str:
        return "verify_source_of_wealth"

    @property
    def description(self) -> str:
        return (
            "Verify Source-of-Wealth consistency against declared income, assets, monthly "
            "transaction velocity, and optional offshore_jurisdiction ISO-2 code."
        )

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

    def _run_validated(
        self,
        customer_name: str,
        annual_income_usd: float,
        declared_assets_usd: float,
        expected_monthly_tx_usd: float,
        source_of_wealth: str,
        offshore_jurisdiction: str | None = None,
    ) -> ToolResult:
        res = evaluate_source_of_wealth_logic(
            annual_income_usd=annual_income_usd,
            declared_assets_usd=declared_assets_usd,
            expected_monthly_tx_usd=expected_monthly_tx_usd,
            source_of_wealth=source_of_wealth,
            offshore_jurisdiction=offshore_jurisdiction,
        )
        res["tool"] = self.name
        res["customer_name"] = customer_name
        return ToolResult(
            success=True,
            output=json.dumps(res),
            metadata={"consistent": res["consistent"], "edd_required": res["edd_required"]},
        )


class ScopedToolRegistry(ToolRegistry):
    """Per-instance ToolRegistry avoiding mutable class-level tool collisions."""

    def __init__(self, include_builtins: bool = False) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._builtins_registered: bool = not include_builtins

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def register_custom(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if not self._builtins_registered:
            ToolRegistry.auto_register_builtins()
            for builtin_name in ToolRegistry.list_all():
                if builtin_name not in self._tools:
                    self._tools[builtin_name] = ToolRegistry.get(builtin_name)
            self._builtins_registered = True
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {self.list_all()}")
        return self._tools[name]

    def get_many(self, names: list[str]) -> list[BaseTool]:
        return [self.get(n) for n in names]

    def list_all(self) -> list[str]:
        return list(self._tools.keys())

    def list_tools(self) -> list[str]:
        return self.list_all()


def create_kyc_tools(db_path: str | Path | None = None) -> list[KYCBaseTool]:
    """Instantiate all 6 KYC compliance BaseTool objects pointing at db_path."""
    resolved = Path(db_path or DEFAULT_WATCHLIST_DB)
    return [
        ScreenSanctionsTool(resolved),
        ScreenExportControlTool(resolved),
        ScreenPepAdverseMediaTool(resolved),
        CheckFatfJurisdictionRiskTool(resolved),
        ScreenCryptoWalletTool(resolved),
        VerifySourceOfWealthTool(resolved),
    ]


def wrap_isolated_kyc_tools(
    db_path: str | Path | None = None,
    timeout_s: float = 30.0,
    require_isolation: bool = False,
) -> list[IsolatedTool]:
    """Wrap all 6 SQLite KYC tools in AgentTune IsolatedTool subprocess sandboxes."""
    return list(
        isolate_tools(
            create_kyc_tools(db_path),
            timeout_s=timeout_s,
            require_isolation=require_isolation,
        )
    )


def build_kyc_tool_registry(
    db_path: str | Path | None = None,
    include_builtins: bool = False,
    isolate: bool = False,
    timeout_s: float = 30.0,
) -> ScopedToolRegistry:
    """Register the 6 KYC compliance tools cleanly on an isolated ToolRegistry."""
    registry = ScopedToolRegistry(include_builtins=include_builtins)
    tools: list[BaseTool] = (
        wrap_isolated_kyc_tools(db_path, timeout_s=timeout_s, require_isolation=False)
        if isolate
        else create_kyc_tools(db_path)
    )
    for tool in tools:
        registry.register_custom(tool)
    return registry


def get_openai_tool_schemas(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return full OpenAI function-calling tool schemas for all 6 KYC tools."""
    return [tool.to_schema() for tool in create_kyc_tools(db_path)]


def execute_kyc_tool_call(
    registry: ToolRegistry,
    tool_name: str,
    arguments: Any,
) -> dict[str, Any]:
    """Safely execute a named KYC tool on ToolRegistry and parse structured JSON output."""
    try:
        tool = registry.get(tool_name)
    except KeyError:
        tool = None
    if tool is None:
        return {
            "tool": tool_name,
            "arguments": arguments,
            "success": False,
            "error": f"UNKNOWN_TOOL: '{tool_name}' is not registered.",
            "output": None,
        }

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return {
                "tool": tool_name,
                "arguments": arguments,
                "success": False,
                "error": f"INVALID_JSON_ARGUMENTS({tool_name}): {exc}",
                "output": None,
            }

    if not isinstance(arguments, dict):
        return {
            "tool": tool_name,
            "arguments": arguments,
            "success": False,
            "error": f"SCHEMA_VALIDATION_ERROR({tool_name}): arguments must be a JSON object.",
            "output": None,
        }

    res = tool.execute(**arguments)
    parsed_output: Any = res.output
    if res.success and isinstance(res.output, str) and res.output.strip():
        try:
            parsed_output = json.loads(res.output)
        except json.JSONDecodeError:
            parsed_output = res.output

    return {
        "tool": tool_name,
        "arguments": arguments,
        "success": bool(res.success),
        "error": res.error,
        "output": parsed_output,
        "metadata": dict(res.metadata),
    }
