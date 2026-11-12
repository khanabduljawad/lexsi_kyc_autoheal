"""Build deterministic, zero-leakage KYC/AML Split A (40), Split B (120), Split C (60).

Guarantees:
1. Ground-truth expected_verdict for every non-abstain case is computed from real
   SQLite watchlist screening + src/kyc_oracle.py (never guessed from noisy fields).
2. Zero overlapping entity names (normalized) or crypto wallets across A, B, and C.
3. Full scenario diversity across all FinCEN/FATF/OFAC/tool-calling failure shapes:
   - sanctions_primary_deny (OFAC/UN/EU/PRIMARY SDN -> DENY)
   - export_control_deny (BIS Entity List / Denied Persons -> DENY)
   - sanctioned_wallet_deny (OFAC crypto address -> DENY)
   - weak_alias_review (sahith002 AKA_WEAK -> REVIEW)
   - pep_edd_review (Senior PEP -> REVIEW)
   - dob_disambiguation_clean (Same PEP name + conflicting DOB -> APPROVE)
   - fatf_jurisdiction_review (FATF blacklist/greylist/offshore -> REVIEW)
   - offshore_sow_review (Sovereign-Forger/EDD disproportionate SoW -> REVIEW)
   - abstain_missing_cip (Incomplete CIP -> abstain + REVIEW)
   - iso2_format_challenge (Natural-language 'USA'/'United Kingdom' input -> normalize to ISO-2 -> APPROVE/DENY)
   - clean_benign_entity (synthetic Faker + CuratorKIT sanitized commercial orgs & individuals -> APPROVE)
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kyc_oracle import (
    FATF_BLACK_LIST,
    FATF_GREY_LIST,
    OFFSHORE_SECRECY_HUBS,
    compute_required_tools,
    evaluate_case_oracle,
    is_valid_dob,
    normalize_country_iso2,
)
from src.kyc_tools import (
    DEFAULT_WATCHLIST_DB,
    build_kyc_tool_registry,
    execute_kyc_tool_call,
    normalize_entity_text,
)

SEEDS_DIR = PROJECT_ROOT / "data" / "seeds"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

SPLIT_SIZES = {"split_a": 40, "split_b": 120, "split_c": 60}

RAW_COUNTRY_PHRASES: dict[str, list[str]] = {
    "US": ["USA", "United States", "U.S.A."],
    "GB": ["UK", "United Kingdom", "Great Britain"],
    "DE": ["Germany", "DEU"],
    "FR": ["France", "FRA"],
    "CA": ["Canada", "CAN"],
    "AU": ["Australia", "AUS"],
    "SG": ["Singapore", "SGP"],
    "JP": ["Japan", "JPN"],
    "NL": ["Netherlands", "NLD"],
    "CH": ["Switzerland", "CHE"],
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _verify_oracle_for_case(
    case: dict[str, Any],
    registry: Any,
) -> dict[str, Any]:
    """Run all required SQLite tools deterministically on a case and verify oracle output."""
    policy = compute_required_tools(case)
    if policy["abstain"]:
        return evaluate_case_oracle(case, tool_observations=[])

    country = case["expected_country_iso2"]
    observations: list[dict[str, Any]] = []
    for t_name in policy["required_tools"]:
        if t_name == "screen_sanctions":
            args = {
                "name": case["customer_name"],
                "entity_type": case["entity_type"],
                "country_iso2": country,
                "date_of_birth": case.get("date_of_birth"),
            }
        elif t_name == "screen_export_control":
            args = {"entity_name": case["customer_name"], "country_iso2": country}
        elif t_name == "screen_pep_adverse_media":
            args = {
                "name": case["customer_name"],
                "country_iso2": country,
                "date_of_birth": case.get("date_of_birth"),
            }
        elif t_name == "check_fatf_jurisdiction_risk":
            args = {"country_iso2": country}
        elif t_name == "screen_crypto_wallet":
            args = {
                "wallet_address": case["crypto_wallet"],
                "currency": case.get("crypto_currency") or "ETH",
            }
        elif t_name == "verify_source_of_wealth":
            args = {
                "customer_name": case["customer_name"],
                "annual_income_usd": float(case.get("annual_income_usd") or 0.0),
                "declared_assets_usd": float(case.get("declared_assets_usd") or 0.0),
                "expected_monthly_tx_usd": float(case.get("expected_monthly_tx_usd") or 0.0),
                "source_of_wealth": case.get("source_of_wealth") or "salary",
                "offshore_jurisdiction": case.get("offshore_jurisdiction"),
            }
        else:
            continue
        res = execute_kyc_tool_call(registry, t_name, args)
        if not res["success"]:
            raise RuntimeError(f"Tool execution failed building case {case['customer_name']}: {res['error']}")
        observations.append({"tool": t_name, "arguments": args, "result": res["output"]})

    return evaluate_case_oracle(case, tool_observations=observations)


def _shift_dob_far(dob: str) -> str:
    """Create a guaranteed >15 year shifted DOB for clean PEP false-positive disambiguation."""
    if dob and re.match(r"^\d{4}-\d{2}-\d{2}$", dob):
        year = int(dob[:4])
        new_year = year - 18 if year >= 1955 else year + 22
        return f"{new_year:04d}-06-15"
    if dob and re.match(r"^\d{4}$", dob):
        year = int(dob)
        new_year = year - 18 if year >= 1955 else year + 22
        return f"{new_year:04d}-06-15"
    return "1991-09-14"


def _make_base_case(
    idx: int,
    scenario_tag: str,
    customer_name: str,
    entity_type: str,
    country_iso2: str,
    raw_country: str | None = None,
    dob: str | None = None,
    id_number: str | None = None,
    annual_income_usd: float = 120_000.0,
    declared_assets_usd: float = 150_000.0,
    expected_monthly_tx_usd: float = 4_000.0,
    source_of_wealth: str = "salary",
    offshore_jurisdiction: str | None = None,
    crypto_wallet: str | None = None,
    crypto_currency: str | None = None,
    missing_cip_fields: list[str] | None = None,
    source_dataset: str = "curated_hf",
    extra_narrative: str = "",
) -> dict[str, Any]:
    raw_c = raw_country or country_iso2
    wallet_note = (
        f" Declared crypto wallet {crypto_wallet} ({crypto_currency or 'ETH'})."
        if crypto_wallet
        else ""
    )
    offshore_note = (
        f" Associated offshore jurisdiction: {offshore_jurisdiction}."
        if offshore_jurisdiction
        else ""
    )
    dob_note = f" DOB: {dob}." if dob else " DOB: not provided."
    id_note = f" ID: {id_number}." if id_number else " ID: not provided."
    narrative = (
        f"Onboarding request for {entity_type} '{customer_name}' domiciled in {raw_c}.{dob_note}{id_note} "
        f"Declared annual income ${annual_income_usd:,.0f}, total assets ${declared_assets_usd:,.0f}, "
        f"expected monthly volume ${expected_monthly_tx_usd:,.0f}, source of wealth: {source_of_wealth}."
        f"{offshore_note}{wallet_note} {extra_narrative}"
    ).strip()

    return {
        "case_id": f"kyc_{scenario_tag}_{idx:04d}",
        "scenario_tag": scenario_tag,
        "source_dataset": source_dataset,
        "customer_name": customer_name,
        "entity_type": entity_type,
        "country_iso2": country_iso2 if not missing_cip_fields or "country_iso2" not in missing_cip_fields else None,
        "expected_country_iso2": country_iso2,
        "raw_country": raw_c,
        "date_of_birth": dob,
        "id_number": id_number,
        "occupation_or_industry": "Technology Consulting" if entity_type == "organization" else "Senior Engineer",
        "annual_income_usd": annual_income_usd,
        "declared_assets_usd": declared_assets_usd,
        "expected_monthly_tx_usd": expected_monthly_tx_usd,
        "source_of_wealth": source_of_wealth,
        "offshore_jurisdiction": offshore_jurisdiction,
        "crypto_wallet": crypto_wallet,
        "crypto_currency": crypto_currency,
        "beneficial_ownership_verified": True,
        "missing_cip_fields": missing_cip_fields or [],
        "narrative": narrative,
    }


def _collect_disjoint_pools(seeds_dir: Path, db_path: Path, seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    registry = build_kyc_tool_registry(db_path, include_builtins=False)

    sanctions = _read_jsonl(seeds_dir / "sanctions_records.jsonl")
    export_ctrl = _read_jsonl(seeds_dir / "export_control_records.jsonl")
    peps = _read_jsonl(seeds_dir / "pep_records.jsonl")
    wallets = _read_jsonl(seeds_dir / "crypto_wallets.jsonl")
    synthetic_benign = _read_jsonl(seeds_dir / "synthetic_benign_entities.jsonl")
    sf_rows = _read_jsonl(seeds_dir / "sovereign_forger_kyc.jsonl")

    # Track unique entity names globally to enforce zero-collision pools across categories
    used_names: set[str] = set()
    used_wallets: set[str] = set()

    def _claim_name(name: str) -> bool:
        norm = normalize_entity_text(name)
        if not norm or norm in used_names:
            return False
        used_names.add(norm)
        return True

    # Partition clean benign commercial orgs first — verify ZERO watchlist collisions against SQLite!
    benign_orgs: list[dict[str, Any]] = []
    iso2_challenge_benign: list[dict[str, Any]] = []
    rng.shuffle(synthetic_benign)
    for idx, row in enumerate(synthetic_benign):
        name = row["name"]
        country = row.get("country_iso2") or "US"
        if country in FATF_BLACK_LIST or country in FATF_GREY_LIST or country in OFFSHORE_SECRECY_HUBS:
            continue
        if not _claim_name(name):
            continue
        tag = "iso2_format_challenge" if idx % 3 == 0 and country in RAW_COUNTRY_PHRASES else "clean_benign_organization"
        raw_c = RAW_COUNTRY_PHRASES[country][0] if tag == "iso2_format_challenge" else country
        cand = _make_base_case(
            len(benign_orgs) + len(iso2_challenge_benign),
            tag,
            customer_name=name,
            entity_type="organization",
            country_iso2=country,
            raw_country=raw_c,
            dob=None,
            id_number=row.get("entity_id") or f"REG-{abs(hash(name)) % 900000 + 100000}",
            source_dataset="synthetic/curatorkit-faker-benign",
        )
        oracle = _verify_oracle_for_case(cand, registry)
        if oracle["verdict"] == "APPROVE":
            cand["expected_verdict"] = "APPROVE"
            cand["oracle_reason_codes"] = oracle["reason_codes"]
            if tag == "iso2_format_challenge":
                iso2_challenge_benign.append(cand)
            else:
                benign_orgs.append(cand)
        if len(benign_orgs) >= 45 and len(iso2_challenge_benign) >= 35:
            break

    # Clean benign individuals + sovereign-forger cases + offshore SoW + FATF jurisdiction cases
    first_names = ["claire", "david", "hannah", "marcus", "priya", "oliver", "sofia", "daniel", "maya", "lucas"]
    last_names = ["thorne", "kessler", "lindqvist", "okafor", "bennett", "fischer", "patel", "nakamura"]
    clean_ind_pool: list[dict[str, Any]] = []
    fatf_pool: list[dict[str, Any]] = []
    sow_pool: list[dict[str, Any]] = []
    abstain_pool: list[dict[str, Any]] = []

    synth_idx = 0
    for fn in first_names:
        for ln in last_names:
            full_name = f"{fn.title()} {ln.title()}"
            if not _claim_name(full_name):
                continue
            synth_idx += 1
            mode = synth_idx % 4
            if mode == 0:
                cand = _make_base_case(
                    synth_idx,
                    "clean_benign_individual",
                    customer_name=full_name,
                    entity_type="individual",
                    country_iso2="US" if synth_idx % 2 == 0 else "CA",
                    raw_country="United States" if synth_idx % 2 == 0 else "CA",
                    dob=f"198{synth_idx % 9}-0{(synth_idx % 8) + 1}-1{(synth_idx % 8) + 1}",
                    id_number=f"PASS-{800000 + synth_idx}",
                    source_dataset="curated_clean_individual",
                )
            elif mode == 1:
                jur = sorted(FATF_GREY_LIST | FATF_BLACK_LIST)[synth_idx % len(FATF_GREY_LIST | FATF_BLACK_LIST)]
                cand = _make_base_case(
                    synth_idx,
                    "fatf_jurisdiction_review",
                    customer_name=full_name,
                    entity_type="individual",
                    country_iso2=jur,
                    dob=f"198{synth_idx % 9}-05-12",
                    id_number=f"NAT-{700000 + synth_idx}",
                    source_dataset="curated_fatf_jurisdiction",
                )
            elif mode == 2:
                hub = sorted(OFFSHORE_SECRECY_HUBS)[synth_idx % len(OFFSHORE_SECRECY_HUBS)]
                cand = _make_base_case(
                    synth_idx,
                    "offshore_sow_review",
                    customer_name=full_name,
                    entity_type="individual",
                    country_iso2="US",
                    dob=f"197{synth_idx % 9}-03-21",
                    id_number=f"ID-{600000 + synth_idx}",
                    annual_income_usd=65_000.0,
                    declared_assets_usd=3_500_000.0,
                    expected_monthly_tx_usd=95_000.0,
                    source_of_wealth="shell_company_dividends",
                    offshore_jurisdiction=hub,
                    source_dataset="curated_sow_edd",
                )
            else:
                cand = _make_base_case(
                    synth_idx,
                    "abstain_missing_cip",
                    customer_name=full_name,
                    entity_type="individual",
                    country_iso2="US",
                    dob=None,
                    id_number=None,
                    missing_cip_fields=["date_of_birth", "id_number"],
                    source_dataset="curated_missing_cip",
                )

            oracle = _verify_oracle_for_case(cand, registry)
            cand["expected_verdict"] = oracle["verdict"]
            cand["oracle_reason_codes"] = oracle["reason_codes"]
            if cand["scenario_tag"] == "clean_benign_individual" and oracle["verdict"] == "APPROVE":
                clean_ind_pool.append(cand)
            elif cand["scenario_tag"] == "fatf_jurisdiction_review" and oracle["verdict"] == "REVIEW":
                fatf_pool.append(cand)
            elif cand["scenario_tag"] == "offshore_sow_review" and oracle["verdict"] == "REVIEW":
                sow_pool.append(cand)
            elif cand["scenario_tag"] == "abstain_missing_cip" and oracle["verdict"] == "REVIEW":
                abstain_pool.append(cand)

    # Enrich offshore_sow_review with real sovereign-forger dossiers
    for i, sf in enumerate(sf_rows):
        sf_name = str(sf.get("full_name") or "").strip()
        if not sf_name or not _claim_name(sf_name):
            continue
        assets = float(str(sf.get("total_assets") or "5000000").replace(",", "").replace("$", ""))
        cand = _make_base_case(
            900 + i,
            "offshore_sow_review",
            customer_name=sf_name,
            entity_type="individual",
            country_iso2="US",
            dob=f"197{i % 9}-08-14",
            id_number=f"SF-{1000 + i}",
            annual_income_usd=180_000.0,
            declared_assets_usd=assets,
            expected_monthly_tx_usd=250_000.0,
            source_of_wealth="unverified_offshore_trust",
            offshore_jurisdiction="KY",
            source_dataset="sovereign-forger/kyc-aml-sample-data",
            extra_narrative=f"Offshore vehicle: {sf.get('offshore_vehicle', '')}; {sf.get('profession', '')}.",
        )
        oracle = _verify_oracle_for_case(cand, registry)
        cand["expected_verdict"] = oracle["verdict"]
        cand["oracle_reason_codes"] = oracle["reason_codes"]
        if oracle["verdict"] == "REVIEW":
            sow_pool.append(cand)

    # Watchlist Sanctions PRIMARY/AKA_STRONG -> DENY & sahith002 AKA_WEAK -> REVIEW
    sanctions_deny_pool: list[dict[str, Any]] = []
    weak_alias_pool: list[dict[str, Any]] = []
    rng.shuffle(sanctions)
    for row in sanctions:
        nt = str(row.get("name_type") or "PRIMARY").upper()
        name = row["name"]
        if nt == "AKA_WEAK" and len(weak_alias_pool) < 28:
            if not _claim_name(name):
                continue
            cand = _make_base_case(
                len(weak_alias_pool),
                "weak_alias_review",
                customer_name=name,
                entity_type=row.get("entity_type", "organization"),
                country_iso2=(row.get("countries") or ["US"])[0],
                dob=row.get("birth_date") or ("1980-02-14" if row.get("entity_type") == "individual" else None),
                id_number=f"ID-WK-{len(weak_alias_pool):04d}",
                source_dataset=row.get("source", "SAHITH_OFAC"),
            )
            oracle = _verify_oracle_for_case(cand, registry)
            if oracle["verdict"] == "REVIEW":
                cand["expected_verdict"] = "REVIEW"
                cand["oracle_reason_codes"] = oracle["reason_codes"]
                weak_alias_pool.append(cand)
        elif nt in {"PRIMARY", "AKA_STRONG"} and len(sanctions_deny_pool) < 35:
            if not _claim_name(name):
                continue
            c_iso = (row.get("countries") or ["IR" if "SDN" in row.get("source", "") else "RU"])[0]
            cand = _make_base_case(
                len(sanctions_deny_pool),
                "sanctions_primary_deny",
                customer_name=name,
                entity_type=row.get("entity_type", "individual"),
                country_iso2=c_iso,
                dob=row.get("birth_date") or ("1978-11-22" if row.get("entity_type") == "individual" else None),
                id_number=f"ID-SDN-{len(sanctions_deny_pool):04d}",
                source_dataset=row.get("source", "US_OFAC_SDN"),
            )
            oracle = _verify_oracle_for_case(cand, registry)
            if oracle["verdict"] == "DENY":
                cand["expected_verdict"] = "DENY"
                cand["oracle_reason_codes"] = oracle["reason_codes"]
                sanctions_deny_pool.append(cand)
        if len(weak_alias_pool) >= 28 and len(sanctions_deny_pool) >= 35:
            break

    # Export Control BIS -> DENY
    export_deny_pool: list[dict[str, Any]] = []
    rng.shuffle(export_ctrl)
    for row in export_ctrl:
        name = row["name"]
        if not _claim_name(name):
            continue
        c_iso = (row.get("countries") or ["CN"])[0]
        cand = _make_base_case(
            len(export_deny_pool),
            "export_control_deny",
            customer_name=name,
            entity_type="organization",
            country_iso2=c_iso,
            id_number=f"BIS-{len(export_deny_pool):04d}",
            source_dataset=row.get("source", "US_BIS_ENTITY_LIST"),
        )
        oracle = _verify_oracle_for_case(cand, registry)
        if oracle["verdict"] == "DENY":
            cand["expected_verdict"] = "DENY"
            cand["oracle_reason_codes"] = oracle["reason_codes"]
            export_deny_pool.append(cand)
        if len(export_deny_pool) >= 25:
            break

    # PEP EDD (matching/unknown DOB -> REVIEW) & DOB Disambiguation Clean (different DOB -> APPROVE)
    pep_edd_pool: list[dict[str, Any]] = []
    dob_clean_pool: list[dict[str, Any]] = []
    rng.shuffle(peps)
    for row in peps:
        if not row.get("birth_date"):
            continue
        name = row["name"]
        if not _claim_name(name):
            continue
        country = (row.get("countries") or ["GB"])[0]
        if country in FATF_BLACK_LIST or country in FATF_GREY_LIST or country in OFFSHORE_SECRECY_HUBS:
            continue
        rec_dob = row["birth_date"]
        use_clean_dob = len(dob_clean_pool) < len(pep_edd_pool) and len(dob_clean_pool) < 25
        if use_clean_dob:
            shifted_dob = _shift_dob_far(rec_dob)
            cand = _make_base_case(
                len(dob_clean_pool),
                "dob_disambiguation_clean",
                customer_name=name,
                entity_type="individual",
                country_iso2=country,
                dob=shifted_dob,
                id_number=f"PASS-FP-{len(dob_clean_pool):04d}",
                source_dataset="proofaml/sanctions-snapshot",
            )
            oracle = _verify_oracle_for_case(cand, registry)
            if oracle["verdict"] == "APPROVE":
                cand["expected_verdict"] = "APPROVE"
                cand["oracle_reason_codes"] = oracle["reason_codes"]
                dob_clean_pool.append(cand)
        elif len(pep_edd_pool) < 28:
            exact_dob = rec_dob if is_valid_dob(rec_dob) else f"{rec_dob}-01-01"
            if not is_valid_dob(exact_dob):
                continue
            cand = _make_base_case(
                len(pep_edd_pool),
                "pep_edd_review",
                customer_name=name,
                entity_type="individual",
                country_iso2=country,
                dob=exact_dob,
                id_number=f"PASS-PEP-{len(pep_edd_pool):04d}",
                source_dataset="proofaml/sanctions-snapshot",
            )
            oracle = _verify_oracle_for_case(cand, registry)
            if oracle["verdict"] == "REVIEW":
                cand["expected_verdict"] = "REVIEW"
                cand["oracle_reason_codes"] = oracle["reason_codes"]
                pep_edd_pool.append(cand)
        if len(pep_edd_pool) >= 28 and len(dob_clean_pool) >= 25:
            break

    # Sanctioned Crypto Wallets -> DENY
    wallet_deny_pool: list[dict[str, Any]] = []
    rng.shuffle(wallets)
    for idx, w in enumerate(wallets):
        addr = w["address"]
        if addr.lower() in used_wallets or len(addr) < 12:
            continue
        cust_name = f"Digital Treasury Client {idx:04d}"
        if not _claim_name(cust_name):
            continue
        used_wallets.add(addr.lower())
        cand = _make_base_case(
            len(wallet_deny_pool),
            "sanctioned_wallet_deny",
            customer_name=cust_name,
            entity_type="individual",
            country_iso2="US",
            raw_country="USA",
            dob="1986-07-19",
            id_number=f"WLT-{idx:04d}",
            crypto_wallet=addr,
            crypto_currency=w.get("currency", "ETH"),
            source_dataset="ai-decisions/ofac-sdn-crypto-addresses",
        )
        oracle = _verify_oracle_for_case(cand, registry)
        if oracle["verdict"] == "DENY":
            cand["expected_verdict"] = "DENY"
            cand["oracle_reason_codes"] = oracle["reason_codes"]
            wallet_deny_pool.append(cand)
        if len(wallet_deny_pool) >= 25:
            break

    return {
        "sanctions_primary_deny": sanctions_deny_pool,
        "export_control_deny": export_deny_pool,
        "sanctioned_wallet_deny": wallet_deny_pool,
        "weak_alias_review": weak_alias_pool,
        "pep_edd_review": pep_edd_pool,
        "fatf_jurisdiction_review": fatf_pool,
        "offshore_sow_review": sow_pool,
        "abstain_missing_cip": abstain_pool,
        "dob_disambiguation_clean": dob_clean_pool,
        "iso2_format_challenge": iso2_challenge_benign,
        "clean_benign_organization": benign_orgs,
        "clean_benign_individual": clean_ind_pool,
    }


def _allocate_disjoint_splits(pools: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Allocate exactly 40 (A), 120 (B), 60 (C) cases across stratified scenarios."""
    # Per-split target quotas per scenario summing cleanly to 40, 120, 60
    quotas: dict[str, tuple[int, int, int]] = {
        "sanctions_primary_deny": (5, 16, 8),
        "export_control_deny": (3, 10, 5),
        "sanctioned_wallet_deny": (4, 12, 6),
        "weak_alias_review": (3, 10, 5),
        "pep_edd_review": (4, 12, 6),
        "fatf_jurisdiction_review": (3, 10, 4),
        "offshore_sow_review": (3, 10, 5),
        "abstain_missing_cip": (3, 8, 4),
        "dob_disambiguation_clean": (3, 10, 5),
        "iso2_format_challenge": (3, 8, 4),
        "clean_benign_organization": (3, 8, 4),
        "clean_benign_individual": (3, 6, 4),
    }

    split_a: list[dict[str, Any]] = []
    split_b: list[dict[str, Any]] = []
    split_c: list[dict[str, Any]] = []

    for scenario, (qa, qb, qc) in quotas.items():
        items = pools.get(scenario, [])
        needed = qa + qb + qc
        if len(items) < needed:
            raise RuntimeError(
                f"Pool '{scenario}' has only {len(items)} verified items, needs {needed}."
            )
        split_a.extend(items[:qa])
        split_b.extend(items[qa : qa + qb])
        split_c.extend(items[qa + qb : needed])

    # Deterministically shuffle each split individually so prefix slices (e.g. --limit 36)
    # stay stratified across all scenarios without changing split membership or leakage bounds.
    split_rng = random.Random(42)
    for split_label, split_rows in (("split_a", split_a), ("split_b", split_b), ("split_c", split_c)):
        split_rng.shuffle(split_rows)
        for i, row in enumerate(split_rows):
            row["split"] = split_label
            row["case_id"] = f"{split_label}_{row['scenario_tag']}_{i:03d}"

    return {"split_a": split_a, "split_b": split_b, "split_c": split_c}


def verify_zero_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Strictly verify zero entity/wallet/id overlap across Split A, Split B, Split C."""
    assert len(splits["split_a"]) == SPLIT_SIZES["split_a"]
    assert len(splits["split_b"]) == SPLIT_SIZES["split_b"]
    assert len(splits["split_c"]) == SPLIT_SIZES["split_c"]

    def _keys(rows: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
        ids = {r["case_id"] for r in rows}
        names = {normalize_entity_text(r["customer_name"]) for r in rows if r.get("customer_name")}
        wallets = {str(r["crypto_wallet"]).lower() for r in rows if r.get("crypto_wallet")}
        return ids, names, wallets

    a_ids, a_names, a_w = _keys(splits["split_a"])
    b_ids, b_names, b_w = _keys(splits["split_b"])
    c_ids, c_names, c_w = _keys(splits["split_c"])

    for label, s1, s2 in (
        ("case_id A&B", a_ids, b_ids),
        ("case_id A&C", a_ids, c_ids),
        ("case_id B&C", b_ids, c_ids),
        ("name A&B", a_names, b_names),
        ("name A&C", a_names, c_names),
        ("name B&C", b_names, c_names),
        ("wallet A&B", a_w, b_w),
        ("wallet A&C", a_w, c_w),
        ("wallet B&C", b_w, c_w),
    ):
        overlap = s1 & s2
        if overlap:
            raise AssertionError(f"Zero-leakage violation on {label}: {sorted(overlap)[:5]}")


def build_all_splits(
    seeds_dir: Path | None = None,
    splits_dir: Path | None = None,
    db_path: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Build Split A (40), Split B (120), Split C (60), verify zero leakage, and write JSONLs."""
    s_dir = seeds_dir or SEEDS_DIR
    out_dir = splits_dir or SPLITS_DIR
    resolved_db = Path(db_path or DEFAULT_WATCHLIST_DB)
    out_dir.mkdir(parents=True, exist_ok=True)

    pools = _collect_disjoint_pools(s_dir, resolved_db, seed=seed)
    splits = _allocate_disjoint_splits(pools)
    verify_zero_leakage(splits)

    summary: dict[str, Any] = {}
    for name, rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        verdict_counts: dict[str, int] = {}
        for row in rows:
            v = row["expected_verdict"]
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        summary[name] = {"count": len(rows), "verdicts": verdict_counts, "path": str(path)}

    return summary


if __name__ == "__main__":
    res = build_all_splits()
    print(json.dumps(res, indent=2))
