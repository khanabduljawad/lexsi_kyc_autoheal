"""Download and normalize public Hugging Face compliance datasets into data/seeds/.

Sources used (all public, ungated):
1. proofaml/global-sanctions-regulatory-enforcement (OFAC SDN, UN SC, EU FSF)
2. proofaml/global-export-control-denied-party (US BIS Entity List & Denied Persons)
3. proofaml/sanctions-snapshot (PEPs + birth dates)
4. sahith002/ofac-screening-names-sample (PRIMARY / AKA_STRONG / AKA_WEAK labels)
5. ai-decisions/ofac-sdn-crypto-addresses (OFAC sanctioned digital currency addresses)
6. 316usman/sanctions-screening-match (prompt + chosen ONLY; degenerate rejected omitted)
7. sovereign-forger/kyc-aml-sample-data (30-field KYC/AML wealth & offshore dossiers)
8. Synthetic benign commercial controls sanitized via Faker + CuratorKIT PIIPseudonymizer
9. gorilla-llm/Berkeley-Function-Calling-Leaderboard (BFCL_v3_simple & BFCL_v3_multiple via AuditKIT)
"""

from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path
import re
import sys
from typing import Any

from curatorkit.hygiene.pii import PIIPseudonymizer
from curatorkit.schema import DataSample
from datasets import load_dataset
from faker import Faker
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit_eval import load_bfcl_compliance_subset
from src.kyc_oracle import is_valid_iso2, normalize_country_iso2

SEEDS_DIR = PROJECT_ROOT / "data" / "seeds"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

DOB_CLEAN_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
YEAR_ONLY_RE = re.compile(r"^(\d{4})$")


def _clean_birth_date(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Reject uncertain upper-bound dates like '≤ 2008-03-09' for clean disambiguation
    if text.startswith("≤") or text.startswith("<="):
        return None
    m = DOB_CLEAN_RE.search(text)
    if m:
        return m.group(1)
    if YEAR_ONLY_RE.match(text):
        return text
    return None


def _normalize_iso2_list(raw_countries: Any) -> list[str]:
    if isinstance(raw_countries, str):
        try:
            raw_countries = json.loads(raw_countries)
        except json.JSONDecodeError:
            raw_countries = [raw_countries]
    out: list[str] = []
    for item in raw_countries or []:
        code = normalize_country_iso2(str(item))
        if code and is_valid_iso2(code) and code not in out:
            out.append(code)
    return out


def _parse_python_or_json_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    if not isinstance(raw, str) or not raw.strip():
        return []
    text = raw.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x]
        except Exception:
            continue
    return [text]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_proofaml_snapshot_gz(
    repo_id: str,
    filename: str,
    source_tag: str,
    limit: int,
) -> list[dict[str, Any]]:
    local_file = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    records: list[dict[str, Any]] = []
    with gzip.open(local_file, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("l") is False:
                continue
            name = str(raw.get("c") or "").strip()
            if len(name) < 3:
                continue
            schema = str(raw.get("s") or "").lower()
            entity_type = "individual" if "person" in schema else "organization"
            records.append(
                {
                    "name": name,
                    "aliases": [str(a).strip() for a in (raw.get("a") or []) if str(a).strip()],
                    "entity_type": entity_type,
                    "source": source_tag,
                    "programs": [str(p) for p in (raw.get("p") or []) if p],
                    "countries": _normalize_iso2_list(raw.get("cc") or []),
                    "birth_date": _clean_birth_date(raw.get("bd")),
                    "name_type": "PRIMARY",
                    "risk_type": str(raw.get("rt") or "sanctioned"),
                }
            )
            if len(records) >= limit:
                break
    return records


def build_synthetic_benign_controls(count: int = 250, seed: int = 42) -> list[dict[str, Any]]:
    """Generate deterministic Faker benign commercial controls sanitized via CuratorKIT PIIPseudonymizer."""
    Faker.seed(seed)
    fake = Faker("en_US")
    fake.seed_instance(seed)

    countries = ["US", "GB", "DE", "FR", "CA", "AU", "SG", "JP", "NL", "CH"]
    legal_suffixes = ["Inc.", "LLC", "GmbH", "SAS", "Corp.", "Pty Ltd", "Pte. Ltd.", "K.K.", "B.V.", "AG"]
    industries = [
        "Industrial Robotics",
        "Precision Optics",
        "Cloud Logistics",
        "Biomedical Instrumentation",
        "Renewable Grid Storage",
        "Maritime Telemetry",
        "Agricultural Sensors",
        "Acoustic Engineering",
        "Enterprise Payments",
        "Semiconductor Metrology",
    ]
    pseudonymizer = PIIPseudonymizer(
        entity_types=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"],
        fields=["instruction"],
        faker_seed=seed,
        spacy_model="en_core_web_sm",
    )

    rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    samples_to_clean: list[DataSample] = []
    raw_records: list[dict[str, Any]] = []

    idx = 0
    while len(raw_records) < count:
        idx += 1
        country = countries[(idx - 1) % len(countries)]
        suffix = legal_suffixes[(idx - 1) % len(legal_suffixes)]
        sector = industries[(idx * 3) % len(industries)]
        base_label = f"{fake.unique.company()} {sector}"
        clean_base = re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9\s\-&]", "", base_label)).strip()
        if len(clean_base) < 6 or clean_base.lower() in seen_names:
            continue
        seen_names.add(clean_base.lower())
        legal_name = f"{clean_base} {suffix}"
        city_addr = fake.address().replace("\n", ", ")
        contact_note = f"Headquarters: {city_addr}. Primary desk contact: compliance@{clean_base.lower().replace(' ', '')[:18]}.example.com."
        samples_to_clean.append(
            DataSample(
                source_uri="synthetic://faker-benign-controls",
                instruction=contact_note,
                output=legal_name,
                task_type="instruction_following",
                metadata={"entity_idx": idx, "country_iso2": country},
            )
        )
        raw_records.append(
            {
                "entity_id": f"SYN-BENIGN-{idx:04d}",
                "name": clean_base,
                "name_legal": legal_name,
                "country_iso2": country,
                "sector": sector,
                "source": "synthetic_faker_curatorkit",
                "license": "synthetic-mit-curated",
            }
        )

    cleaned_samples = pseudonymizer.run(samples_to_clean)
    for rec, sample in zip(raw_records, cleaned_samples, strict=True):
        rec["address"] = sample.instruction
        rows.append(rec)
    return rows


def load_all_seeds(seeds_dir: Path | None = None, splits_dir: Path | None = None) -> dict[str, Any]:
    """Download, normalize, and persist all HF datasets into compact JSONL seed files."""
    target_seeds = seeds_dir or SEEDS_DIR
    target_splits = splits_dir or SPLITS_DIR
    target_seeds.mkdir(parents=True, exist_ok=True)
    target_splits.mkdir(parents=True, exist_ok=True)

    # 1. Sanctions: OFAC SDN + UN SC + EU FSF + sahith002
    sanctions_rows: list[dict[str, Any]] = []
    sanctions_rows.extend(
        _load_proofaml_snapshot_gz(
            "proofaml/global-sanctions-regulatory-enforcement",
            "data/us_ofac_sdn.snapshot.jsonl.gz",
            "US_OFAC_SDN",
            limit=2500,
        )
    )
    sanctions_rows.extend(
        _load_proofaml_snapshot_gz(
            "proofaml/global-sanctions-regulatory-enforcement",
            "data/un_sc_sanctions.snapshot.jsonl.gz",
            "UN_SC_SANCTIONS",
            limit=1000,
        )
    )
    sanctions_rows.extend(
        _load_proofaml_snapshot_gz(
            "proofaml/global-sanctions-regulatory-enforcement",
            "data/eu_fsf.snapshot.jsonl.gz",
            "EU_FSF",
            limit=1500,
        )
    )

    sahith_path = hf_hub_download(
        repo_id="sahith002/ofac-screening-names-sample",
        filename="screening_names_sample.jsonl",
        repo_type="dataset",
    )
    sahith_count = 0
    with Path(sahith_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            name = str(raw.get("name") or "").strip()
            if len(name) < 3:
                continue
            ent = str(raw.get("entity_type") or "").upper()
            sanctions_rows.append(
                {
                    "name": name,
                    "aliases": [],
                    "entity_type": "individual" if ent == "INDIVIDUAL" else "organization",
                    "source": f"SAHITH_OFAC_{raw.get('source_list', 'SDN')}",
                    "programs": _parse_python_or_json_list(raw.get("programs")),
                    "countries": [],
                    "birth_date": None,
                    "name_type": str(raw.get("name_type") or "PRIMARY").upper(),
                    "risk_type": "sanctioned",
                    "entity_uid": str(raw.get("entity_uid") or ""),
                }
            )
            sahith_count += 1
    _write_jsonl(target_seeds / "sanctions_records.jsonl", sanctions_rows)

    # 2. Export Control: BIS Entity List + BIS Denied Persons
    export_rows: list[dict[str, Any]] = []
    export_rows.extend(
        _load_proofaml_snapshot_gz(
            "proofaml/global-export-control-denied-party",
            "data/us_bis_entity_list.snapshot.jsonl.gz",
            "US_BIS_ENTITY_LIST",
            limit=1500,
        )
    )
    export_rows.extend(
        _load_proofaml_snapshot_gz(
            "proofaml/global-export-control-denied-party",
            "data/us_bis_denied.snapshot.jsonl.gz",
            "US_BIS_DENIED",
            limit=1000,
        )
    )
    _write_jsonl(target_seeds / "export_control_records.jsonl", export_rows)

    # 3. PEP & Adverse Media: targeted parliament/world-leader snapshots from proofaml/sanctions-snapshot
    pep_rows: list[dict[str, Any]] = []
    pep_files = [
        ("data/us_congress.snapshot.jsonl.gz", 400),
        ("data/uk_parliament.snapshot.jsonl.gz", 400),
        ("data/de_parliament.snapshot.jsonl.gz", 400),
        ("data/eu_meps.snapshot.jsonl.gz", 300),
        ("data/us_cia_world_leaders.snapshot.jsonl.gz", 300),
    ]
    for pep_file, cap in pep_files:
        local_gz = hf_hub_download(
            repo_id="proofaml/sanctions-snapshot",
            filename=pep_file,
            repo_type="dataset",
        )
        added = 0
        with gzip.open(local_gz, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                raw = json.loads(line)
                name = str(raw.get("c") or "").strip()
                if len(name) < 4:
                    continue
                bd = _clean_birth_date(raw.get("bd"))
                pep_rows.append(
                    {
                        "name": name,
                        "aliases": [str(a).strip() for a in (raw.get("a") or []) if str(a).strip()],
                        "entity_type": "individual"
                        if "person" in str(raw.get("s") or "Person").lower()
                        else "organization",
                        "source": f"PEP_{(raw.get('d') or [pep_file])[0]}",
                        "programs": [str(p) for p in (raw.get("d") or []) if p],
                        "countries": _normalize_iso2_list(raw.get("cc") or []),
                        "birth_date": bd,
                        "is_pep": True,
                        "pep_tier": str(raw.get("pt") or "level_1"),
                        "pep_body": str(raw.get("pb") or ""),
                        "adverse_media": str(raw.get("rt") or "").lower() in {"crime", "sanction", "pep"},
                    }
                )
                added += 1
                if added >= cap:
                    break
    _write_jsonl(target_seeds / "pep_records.jsonl", pep_rows)

    # 4. Sanctioned Crypto Wallets: ai-decisions/ofac-sdn-crypto-addresses
    wallet_ds = load_dataset("ai-decisions/ofac-sdn-crypto-addresses", split="train")
    wallet_rows: list[dict[str, Any]] = []
    for raw in wallet_ds:
        addr = str(raw.get("address") or "").strip()
        if not addr:
            continue
        wallet_rows.append(
            {
                "address": addr,
                "currency": str(raw.get("currency") or "ETH").strip().upper(),
                "label": str(raw.get("label") or "").strip(),
                "category": str(raw.get("category") or "sanction").strip(),
                "source": "OFAC_SDN_CRYPTO",
            }
        )
    _write_jsonl(target_seeds / "crypto_wallets.jsonl", wallet_rows)

    # 5. Fuzzy Sanctions Screening Match (316usman) — ONLY prompt and chosen! Never rejected!
    match_ds = load_dataset("316usman/sanctions-screening-match")
    match_rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        if split_name not in match_ds:
            continue
        ds_split = match_ds[split_name]
        cap = 80 if split_name == "train" else 30
        for idx, raw in enumerate(ds_split):
            if idx >= cap:
                break
            match_rows.append(
                {
                    "split": split_name,
                    "prompt": str(raw.get("prompt") or "").strip(),
                    "chosen": str(raw.get("chosen") or "").strip(),
                    "source": str(raw.get("source") or ""),
                }
            )
    _write_jsonl(target_seeds / "sanctions_match_prompts.jsonl", match_rows)

    # 6. Sovereign Forger 30-field KYC dossiers
    sf_path = hf_hub_download(
        repo_id="sovereign-forger/kyc-aml-sample-data",
        filename="sovereign-forger-kyc-preview-5records.jsonl",
        repo_type="dataset",
    )
    sf_rows: list[dict[str, Any]] = []
    with Path(sf_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                sf_rows.append(json.loads(line))
    _write_jsonl(target_seeds / "sovereign_forger_kyc.jsonl", sf_rows)

    # 7. Clean benign negative controls (Faker + CuratorKIT PIIPseudonymizer; zero unlicensed corpora)
    legacy_vc940 = target_seeds / "vc940_benign_entities.jsonl"
    if legacy_vc940.exists():
        legacy_vc940.unlink()
    benign_rows = build_synthetic_benign_controls(count=250, seed=42)
    _write_jsonl(target_seeds / "synthetic_benign_entities.jsonl", benign_rows)

    # 8. BFCL pinned subset via AuditKIT load_bfcl (explicit simple + multiple + possible_answer)
    bfcl_rows = load_bfcl_compliance_subset(
        output_path=target_splits / "bfcl_subset.json",
        limit=25,
    )

    manifest = {
        "sanctions_records": len(sanctions_rows),
        "sahith_ofac_included": sahith_count,
        "export_control_records": len(export_rows),
        "pep_records": len(pep_rows),
        "crypto_wallets": len(wallet_rows),
        "sanctions_match_prompts": len(match_rows),
        "sovereign_forger_kyc": len(sf_rows),
        "synthetic_benign_entities": len(benign_rows),
        "bfcl_subset": len(bfcl_rows),
    }
    (target_seeds / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    summary = load_all_seeds()
    print(json.dumps(summary, indent=2))
