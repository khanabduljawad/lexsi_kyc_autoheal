"""Deterministic FinCEN/FATF/OFAC KYC & AML compliance oracle.

Provides the single source of truth for:
- Strict ISO-3166-1 alpha-2 country validation and normalization.
- Calendar-valid YYYY-MM-DD date-of-birth formatting and disambiguation.
- FinCEN 31 CFR 1020.220 CIP completeness gates.
- Conditional tool-execution policy (crypto wallet gating, abstain on missing CIP).
- FATF high-risk / increased-monitoring / offshore secrecy jurisdiction scoring.
- FinCEN CDD / EDD disposition synthesis (APPROVE / REVIEW / DENY).
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

FATF_LIST_VERSION = "2025-2026-static"

ISO2_COUNTRY_RE = re.compile(r"^[A-Z]{2}$")
DOB_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

OFFICIAL_ISO2_CODES: frozenset[str] = frozenset(
    {
        "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT",
        "AU", "AW", "AX", "AZ", "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI",
        "BJ", "BL", "BM", "BN", "BO", "BQ", "BR", "BS", "BT", "BV", "BW", "BY",
        "BZ", "CA", "CC", "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN",
        "CO", "CR", "CU", "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM",
        "DO", "DZ", "EC", "EE", "EG", "EH", "ER", "ES", "ET", "FI", "FJ", "FK",
        "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF", "GG", "GH", "GI", "GL",
        "GM", "GN", "GP", "GQ", "GR", "GS", "GT", "GU", "GW", "GY", "HK", "HM",
        "HN", "HR", "HT", "HU", "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR",
        "IS", "IT", "JE", "JM", "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN",
        "KP", "KR", "KW", "KY", "KZ", "LA", "LB", "LC", "LI", "LK", "LR", "LS",
        "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MH", "MK",
        "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW",
        "MX", "MY", "MZ", "NA", "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP",
        "NR", "NU", "NZ", "OM", "PA", "PE", "PF", "PG", "PH", "PK", "PL", "PM",
        "PN", "PR", "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS", "RU", "RW",
        "SA", "SB", "SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM",
        "SN", "SO", "SR", "SS", "ST", "SV", "SX", "SY", "SZ", "TC", "TD", "TF",
        "TG", "TH", "TJ", "TK", "TL", "TM", "TN", "TO", "TR", "TT", "TV", "TW",
        "TZ", "UA", "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI",
        "VN", "VU", "WF", "WS", "XK", "YE", "YT", "ZA", "ZM", "ZW",
    }
)

COUNTRY_ALIAS_TO_ISO2: dict[str, str] = {
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

FATF_BLACK_LIST: frozenset[str] = frozenset({"KP", "IR", "MM"})
FATF_GREY_LIST: frozenset[str] = frozenset(
    {
        "BG", "BF", "CM", "HR", "CD", "HT", "KE", "ML", "MZ", "NA",
        "NG", "PH", "SN", "ZA", "SS", "SY", "TZ", "VE", "VN", "YE",
    }
)
OFAC_COMPREHENSIVE_EMBARGO: frozenset[str] = frozenset({"KP", "IR", "CU", "SY"})
SANCTIONS_SECTORAL_HIGH_RISK: frozenset[str] = frozenset({"RU", "BY", "AF", "SD", "LY", "SO"})
OFFSHORE_SECRECY_HUBS: frozenset[str] = frozenset(
    {"KY", "VG", "PA", "BZ", "SC", "BM", "JE", "GG", "IM", "LI", "MC", "CY", "MT", "VU", "WS"}
)

ALL_TOOL_NAMES: tuple[str, ...] = (
    "screen_sanctions",
    "screen_export_control",
    "screen_pep_adverse_media",
    "check_fatf_jurisdiction_risk",
    "screen_crypto_wallet",
    "verify_source_of_wealth",
)

HIGH_RISK_SOW_SOURCES: frozenset[str] = frozenset(
    {
        "cash_business",
        "unexplained_crypto",
        "bearer_shares",
        "shell_company_dividends",
        "unverified_offshore_trust",
        "unknown",
    }
)


def is_valid_iso2(country_code: Any) -> bool:
    """Return True strictly for official uppercase ISO-3166-1 alpha-2 country codes."""
    if not isinstance(country_code, str):
        return False
    return bool(ISO2_COUNTRY_RE.match(country_code)) and country_code in OFFICIAL_ISO2_CODES


def normalize_country_iso2(raw_country: Any) -> str | None:
    """Normalize common country names/ISO-3 aliases to uppercase ISO-2 if unambiguous."""
    if not isinstance(raw_country, str):
        return None
    cleaned = raw_country.strip().upper()
    if not cleaned:
        return None
    if is_valid_iso2(cleaned):
        return cleaned
    return COUNTRY_ALIAS_TO_ISO2.get(cleaned)


def is_valid_dob(dob: Any) -> bool:
    """Return True only if dob matches YYYY-MM-DD and is a real calendar date."""
    if not isinstance(dob, str) or not DOB_ISO_RE.match(dob):
        return False
    try:
        parsed = datetime.strptime(dob, "%Y-%m-%d")
    except ValueError:
        return False
    return 1900 <= parsed.year <= datetime.now(timezone.utc).year


def extract_dob_parts(dob: Any) -> tuple[int, int | None, int | None] | None:
    """Extract (year, month, day) from ISO date or year-prefix watchlist strings."""
    if not isinstance(dob, str):
        return None
    raw = dob.strip()
    if not raw:
        return None
    m_full = re.match(r"^(\d{4})-(\d{2})-(\d{2})", raw)
    if m_full:
        try:
            dt = datetime(int(m_full.group(1)), int(m_full.group(2)), int(m_full.group(3)))
            return (dt.year, dt.month, dt.day)
        except ValueError:
            return None
    m_ym = re.match(r"^(\d{4})-(\d{2})$", raw)
    if m_ym:
        y, mo = int(m_ym.group(1)), int(m_ym.group(2))
        if 1900 <= y <= 2100 and 1 <= mo <= 12:
            return (y, mo, None)
    m_y = re.match(r"^(\d{4})$", raw)
    if m_y:
        y = int(m_y.group(1))
        if 1900 <= y <= 2100:
            return (y, None, None)
    return None


def dob_disambiguates(query_dob: Any, record_dob: Any) -> bool:
    """Return True if both DOBs carry valid date info and clearly refer to different people."""
    q = extract_dob_parts(query_dob)
    r = extract_dob_parts(record_dob)
    if q is None or r is None:
        return False
    if q[0] != r[0]:
        return True
    if q[1] is not None and r[1] is not None and q[1] != r[1]:
        return True
    if q[2] is not None and r[2] is not None and q[2] != r[2]:
        return True
    return False


def evaluate_jurisdiction_risk(country_iso2: str) -> dict[str, Any]:
    """Deterministically score FATF blacklist, greylist, embargo, and secrecy risk."""
    code = (country_iso2 or "").strip().upper()
    if not is_valid_iso2(code):
        raise ValueError(f"Invalid ISO-2 country code: {country_iso2!r}")

    tiers: list[str] = []
    if code in FATF_BLACK_LIST:
        tiers.append("FATF_BLACK_LIST")
    if code in OFAC_COMPREHENSIVE_EMBARGO:
        tiers.append("OFAC_EMBARGO")
    if code in FATF_GREY_LIST:
        tiers.append("FATF_GREY_LIST")
    if code in SANCTIONS_SECTORAL_HIGH_RISK:
        tiers.append("SANCTIONS_HIGH_RISK")
    if code in OFFSHORE_SECRECY_HUBS:
        tiers.append("OFFSHORE_SECRECY_HUB")

    if "FATF_BLACK_LIST" in tiers or "OFAC_EMBARGO" in tiers:
        risk_level = "CRITICAL"
        edd_required = True
    elif "FATF_GREY_LIST" in tiers or "SANCTIONS_HIGH_RISK" in tiers or "OFFSHORE_SECRECY_HUB" in tiers:
        risk_level = "HIGH"
        edd_required = True
    else:
        risk_level = "LOW"
        edd_required = False

    return {
        "country_iso2": code,
        "fatf_list_version": FATF_LIST_VERSION,
        "risk_level": risk_level,
        "edd_required": edd_required,
        "risk_tiers": tiers,
    }


def evaluate_source_of_wealth_logic(
    annual_income_usd: float,
    declared_assets_usd: float,
    expected_monthly_tx_usd: float,
    source_of_wealth: str,
    offshore_jurisdiction: str | None = None,
) -> dict[str, Any]:
    """Deterministic FinCEN CDD/EDD Source-of-Wealth consistency evaluation."""
    income = max(float(annual_income_usd or 0.0), 0.0)
    assets = max(float(declared_assets_usd or 0.0), 0.0)
    monthly_tx = max(float(expected_monthly_tx_usd or 0.0), 0.0)
    sow_raw = (source_of_wealth or "").strip().lower().replace(" ", "_")
    offshore = (offshore_jurisdiction or "").strip().upper()

    flags: list[str] = []
    if income <= 0.0 and (assets >= 100_000.0 or monthly_tx >= 15_000.0):
        flags.append("ZERO_INCOME_HIGH_ASSETS")
    elif income > 0.0:
        if assets > income * 15.0 and assets >= 250_000.0 and sow_raw not in {"inheritance", "business_exit", "real_estate_sale"}:
            flags.append("ASSETS_DISPROPORTIONATE_TO_INCOME")
        if (monthly_tx * 12.0) > income * 3.0 and monthly_tx >= 25_000.0:
            flags.append("TX_VELOCITY_DISPROPORTIONATE_TO_INCOME")

    if sow_raw in HIGH_RISK_SOW_SOURCES or not sow_raw:
        flags.append("UNVERIFIED_OR_HIGH_RISK_SOW")

    if offshore and offshore in OFFSHORE_SECRECY_HUBS:
        flags.append(f"OFFSHORE_SOW_NEXUS:{offshore}")

    inconsistent = len(flags) > 0
    return {
        "consistent": not inconsistent,
        "edd_required": inconsistent,
        "flags": flags,
        "annual_income_usd": income,
        "declared_assets_usd": assets,
        "expected_monthly_tx_usd": monthly_tx,
        "source_of_wealth": sow_raw or "unspecified",
        "offshore_jurisdiction": offshore or None,
    }


def evaluate_cip(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate FinCEN 31 CFR 1020.220 CIP completeness on a customer record."""
    explicit_missing = [str(f) for f in (case.get("missing_cip_fields") or []) if f]
    entity_type = str(case.get("entity_type") or "individual").strip().lower()
    name = str(case.get("customer_name") or "").strip()

    raw_country = case.get("country_iso2") or case.get("country") or case.get("raw_country")
    normalized_country = normalize_country_iso2(raw_country)
    dob = case.get("date_of_birth")
    id_number = str(case.get("id_number") or "").strip()

    missing: list[str] = list(dict.fromkeys(explicit_missing))
    if not name and "customer_name" not in missing:
        missing.append("customer_name")
    if not normalized_country and "country" not in missing and "country_iso2" not in missing:
        missing.append("country_iso2")
    if entity_type == "individual":
        if not dob and "date_of_birth" not in missing:
            missing.append("date_of_birth")
        elif dob and not is_valid_dob(dob) and "date_of_birth" not in missing:
            missing.append("date_of_birth")
    if not id_number and "id_number" not in missing:
        missing.append("id_number")

    strict_iso2 = is_valid_iso2(case.get("country_iso2"))
    dob_valid = True if entity_type != "individual" else is_valid_dob(dob)
    cip_complete = len(missing) == 0

    return {
        "cip_complete": cip_complete,
        "missing_fields": missing,
        "country_iso2": normalized_country,
        "country_iso2_valid": strict_iso2 or bool(normalized_country),
        "strict_iso2_provided": strict_iso2,
        "dob_format_valid": dob_valid,
        "abstain_screening": not cip_complete,
    }


def compute_required_tools(case: dict[str, Any]) -> dict[str, Any]:
    """Return required, optional, and forbidden tool names for a case."""
    cip = evaluate_cip(case)
    if not cip["cip_complete"]:
        return {
            "abstain": True,
            "required_tools": [],
            "optional_tools": [],
            "forbidden_tools": list(ALL_TOOL_NAMES),
        }

    entity_type = str(case.get("entity_type") or "individual").strip().lower()
    wallet = str(case.get("crypto_wallet") or "").strip()
    assets = float(case.get("declared_assets_usd") or 0.0)
    monthly_tx = float(case.get("expected_monthly_tx_usd") or 0.0)
    offshore = normalize_country_iso2(case.get("offshore_jurisdiction")) or ""

    required = [
        "screen_sanctions",
        "screen_pep_adverse_media",
        "check_fatf_jurisdiction_risk",
    ]
    optional: list[str] = []
    forbidden: list[str] = []

    if entity_type == "organization" or bool(case.get("check_export_control")):
        required.append("screen_export_control")
    else:
        optional.append("screen_export_control")

    if wallet:
        required.append("screen_crypto_wallet")
    else:
        forbidden.append("screen_crypto_wallet")

    if (
        assets >= 250_000.0
        or monthly_tx >= 50_000.0
        or offshore in OFFSHORE_SECRECY_HUBS
        or bool(case.get("trigger_sow_check"))
    ):
        required.append("verify_source_of_wealth")
    else:
        optional.append("verify_source_of_wealth")

    return {
        "abstain": False,
        "required_tools": required,
        "optional_tools": optional,
        "forbidden_tools": forbidden,
    }


def evaluate_case_oracle(
    case: dict[str, Any],
    tool_observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute the authoritative FinCEN/FATF/OFAC KYC verdict for a case.

    Priority order:
      1. CIP Incomplete -> REVIEW (abstain from watchlist screening)
      2. Confirmed Watchlist Match (PRIMARY/AKA_STRONG SDN, UN/EU, BIS, Sanctioned Wallet) -> DENY
      3. Weak/Ambiguous Watchlist Match (AKA_WEAK or fuzzy ambiguity) -> REVIEW
      4. EDD Triggers (Active PEP without DOB disambiguation, FATF High Risk, Offshore Hub,
         SoW inconsistency, unverified beneficial owner) -> REVIEW
      5. Verified Clean -> APPROVE
    """
    cip = evaluate_cip(case)
    policy = compute_required_tools(case)

    if not cip["cip_complete"]:
        return {
            "verdict": "REVIEW",
            "risk_level": "HIGH",
            "edd_required": True,
            "reason_codes": ["CIP_INCOMPLETE"],
            "cip": cip,
            "tool_policy": policy,
            "explanation": f"Missing/invalid CIP fields {cip['missing_fields']}; abstain from screening and escalate to CIP remediation REVIEW.",
        }

    observations = tool_observations or []
    deny_reasons: list[str] = []
    review_reasons: list[str] = []
    disambiguated_false_positives: list[str] = []

    query_dob = case.get("date_of_birth")

    for obs in observations:
        if obs.get("success") is False:
            continue
        tool_name = obs.get("tool") or obs.get("name")
        if obs.get("result") is not None:
            res = obs.get("result")
        elif obs.get("output") is not None:
            res = obs.get("output")
        else:
            res = obs

        if not isinstance(res, dict):
            continue

        if tool_name == "screen_sanctions":
            for match in res.get("matches", []):
                rec_dob = match.get("birth_date")
                if query_dob and rec_dob and dob_disambiguates(query_dob, rec_dob):
                    disambiguated_false_positives.append(
                        f"DOB_DISAMBIGUATED:{match.get('name', 'unknown')}"
                    )
                    continue
                strength = str(match.get("match_strength") or match.get("name_type") or "STRONG").upper()
                if strength in {"AKA_WEAK", "WEAK", "FUZZY_REVIEW"}:
                    review_reasons.append(f"WEAK_OR_FUZZY_SANCTIONS_MATCH:{match.get('source', 'SANCTIONS')}")
                else:
                    deny_reasons.append(f"SANCTIONS_MATCH:{match.get('source', 'OFAC_SDN')}")

        elif tool_name == "screen_export_control":
            for match in res.get("matches", []):
                strength = str(match.get("match_strength") or "STRONG").upper()
                if strength in {"WEAK", "FUZZY_REVIEW"}:
                    review_reasons.append(f"FUZZY_EXPORT_CONTROL_MATCH:{match.get('source', 'BIS')}")
                else:
                    deny_reasons.append(f"EXPORT_CONTROL_MATCH:{match.get('source', 'BIS')}")

        elif tool_name == "screen_crypto_wallet":
            if res.get("sanctioned"):
                deny_reasons.append(f"SANCTIONED_CRYPTO_WALLET:{res.get('currency', 'CRYPTO')}")

        elif tool_name == "screen_pep_adverse_media":
            for match in res.get("matches", []):
                rec_dob = match.get("birth_date")
                if query_dob and rec_dob and dob_disambiguates(query_dob, rec_dob):
                    disambiguated_false_positives.append(
                        f"PEP_DOB_DISAMBIGUATED:{match.get('name', 'unknown')}"
                    )
                    continue
                if match.get("is_pep") or match.get("adverse_media"):
                    review_reasons.append(f"EDD_PEP_OR_ADVERSE_MEDIA:{match.get('pep_tier', 'PEP')}")

        elif tool_name == "check_fatf_jurisdiction_risk":
            if res.get("edd_required"):
                tiers = res.get("risk_tiers") or []
                tier_label = "+".join(tiers) if tiers else res.get("risk_level", "HIGH")
                review_reasons.append(f"JURISDICTION_EDD:{tier_label}:{res.get('country_iso2', '')}")

        elif tool_name == "verify_source_of_wealth":
            if not res.get("consistent", True):
                for flag in res.get("flags", ["SOW_INCONSISTENCY"]):
                    review_reasons.append(f"SOW:{flag}")

    if case.get("entity_type") == "organization" and case.get("beneficial_ownership_verified") is False:
        review_reasons.append("BO_VERIFICATION_REQUIRED")

    # Deduplicate preserving order
    deny_reasons = list(dict.fromkeys(deny_reasons))
    review_reasons = list(dict.fromkeys(review_reasons))

    if deny_reasons:
        return {
            "verdict": "DENY",
            "risk_level": "CRITICAL",
            "edd_required": True,
            "reason_codes": deny_reasons + review_reasons,
            "disambiguated_false_positives": disambiguated_false_positives,
            "cip": cip,
            "tool_policy": policy,
            "explanation": f"Confirmed sanctions/export/wallet prohibition: {', '.join(deny_reasons)}.",
        }

    if review_reasons:
        return {
            "verdict": "REVIEW",
            "risk_level": "HIGH",
            "edd_required": True,
            "reason_codes": review_reasons,
            "disambiguated_false_positives": disambiguated_false_positives,
            "cip": cip,
            "tool_policy": policy,
            "explanation": f"FinCEN CDD/EDD escalation required: {', '.join(review_reasons)}.",
        }

    reason_codes = ["CLEAN_VERIFIED"]
    if disambiguated_false_positives:
        reason_codes.append("DOB_FALSE_POSITIVE_CLEARED")
    return {
        "verdict": "APPROVE",
        "risk_level": "LOW",
        "edd_required": False,
        "reason_codes": reason_codes,
        "disambiguated_false_positives": disambiguated_false_positives,
        "cip": cip,
        "tool_policy": policy,
        "explanation": "All required FinCEN/OFAC/FATF checks passed with no unresolved matches.",
    }
