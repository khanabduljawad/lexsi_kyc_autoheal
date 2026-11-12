"""Populate indexed SQLite watchlists (data/watchlists.db) from downloaded HF seeds."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kyc_tools import DEFAULT_WATCHLIST_DB, normalize_entity_text

SEEDS_DIR = PROJECT_ROOT / "data" / "seeds"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Seed file missing: {path}. Run scripts/load_hf_seeds.py first.")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_watchlists_db(
    seeds_dir: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, int]:
    """Create and populate indexed sanctions, export_control, pep, and crypto tables."""
    s_dir = seeds_dir or SEEDS_DIR
    target_db = Path(db_path or DEFAULT_WATCHLIST_DB)
    target_db.parent.mkdir(parents=True, exist_ok=True)
    if target_db.exists():
        target_db.unlink()

    sanctions = _read_jsonl(s_dir / "sanctions_records.jsonl")
    export_ctrl = _read_jsonl(s_dir / "export_control_records.jsonl")
    peps = _read_jsonl(s_dir / "pep_records.jsonl")
    wallets = _read_jsonl(s_dir / "crypto_wallets.jsonl")

    conn = sqlite3.connect(str(target_db))
    try:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE sanctions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                norm_name TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                norm_aliases TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                source TEXT NOT NULL,
                programs_json TEXT NOT NULL,
                countries_json TEXT NOT NULL,
                birth_date TEXT,
                name_type TEXT NOT NULL,
                risk_type TEXT NOT NULL
            );
            CREATE INDEX idx_sanctions_norm_name ON sanctions(norm_name);

            CREATE TABLE export_control (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                norm_name TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                norm_aliases TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                source TEXT NOT NULL,
                programs_json TEXT NOT NULL,
                countries_json TEXT NOT NULL,
                birth_date TEXT,
                name_type TEXT NOT NULL,
                risk_type TEXT NOT NULL
            );
            CREATE INDEX idx_export_norm_name ON export_control(norm_name);

            CREATE TABLE pep_adverse_media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                norm_name TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                norm_aliases TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                source TEXT NOT NULL,
                programs_json TEXT NOT NULL,
                countries_json TEXT NOT NULL,
                birth_date TEXT,
                is_pep INTEGER NOT NULL,
                pep_tier TEXT,
                pep_body TEXT,
                adverse_media INTEGER NOT NULL
            );
            CREATE INDEX idx_pep_norm_name ON pep_adverse_media(norm_name);

            CREATE TABLE crypto_wallets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                address TEXT NOT NULL,
                currency TEXT NOT NULL,
                label TEXT,
                category TEXT,
                source TEXT NOT NULL
            );
            CREATE INDEX idx_crypto_addr_lower ON crypto_wallets(lower(address));
            """
        )

        def _norm_aliases_str(aliases: list[str]) -> str:
            return " | ".join(normalize_entity_text(a) for a in aliases if normalize_entity_text(a))

        conn.executemany(
            """
            INSERT INTO sanctions (
                name, norm_name, aliases_json, norm_aliases, entity_type,
                source, programs_json, countries_json, birth_date, name_type, risk_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["name"],
                    normalize_entity_text(r["name"]),
                    json.dumps(r.get("aliases", [])),
                    _norm_aliases_str(r.get("aliases", [])),
                    r.get("entity_type", "organization"),
                    r.get("source", "US_OFAC_SDN"),
                    json.dumps(r.get("programs", [])),
                    json.dumps(r.get("countries", [])),
                    r.get("birth_date"),
                    r.get("name_type", "PRIMARY"),
                    r.get("risk_type", "sanctioned"),
                )
                for r in sanctions
                if normalize_entity_text(r.get("name", ""))
            ],
        )

        conn.executemany(
            """
            INSERT INTO export_control (
                name, norm_name, aliases_json, norm_aliases, entity_type,
                source, programs_json, countries_json, birth_date, name_type, risk_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["name"],
                    normalize_entity_text(r["name"]),
                    json.dumps(r.get("aliases", [])),
                    _norm_aliases_str(r.get("aliases", [])),
                    r.get("entity_type", "organization"),
                    r.get("source", "US_BIS_ENTITY_LIST"),
                    json.dumps(r.get("programs", [])),
                    json.dumps(r.get("countries", [])),
                    r.get("birth_date"),
                    r.get("name_type", "PRIMARY"),
                    r.get("risk_type", "sanctioned"),
                )
                for r in export_ctrl
                if normalize_entity_text(r.get("name", ""))
            ],
        )

        conn.executemany(
            """
            INSERT INTO pep_adverse_media (
                name, norm_name, aliases_json, norm_aliases, entity_type,
                source, programs_json, countries_json, birth_date,
                is_pep, pep_tier, pep_body, adverse_media
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["name"],
                    normalize_entity_text(r["name"]),
                    json.dumps(r.get("aliases", [])),
                    _norm_aliases_str(r.get("aliases", [])),
                    r.get("entity_type", "individual"),
                    r.get("source", "PEP_SNAPSHOT"),
                    json.dumps(r.get("programs", [])),
                    json.dumps(r.get("countries", [])),
                    r.get("birth_date"),
                    1 if r.get("is_pep", True) else 0,
                    r.get("pep_tier", "level_1"),
                    r.get("pep_body", ""),
                    1 if r.get("adverse_media", False) else 0,
                )
                for r in peps
                if normalize_entity_text(r.get("name", ""))
            ],
        )

        conn.executemany(
            """
            INSERT INTO crypto_wallets (address, currency, label, category, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    r["address"],
                    r.get("currency", "ETH"),
                    r.get("label", ""),
                    r.get("category", "sanction"),
                    r.get("source", "OFAC_SDN_CRYPTO"),
                )
                for r in wallets
            ],
        )

        conn.commit()

        counts = {
            "sanctions": conn.execute("SELECT COUNT(*) FROM sanctions").fetchone()[0],
            "export_control": conn.execute("SELECT COUNT(*) FROM export_control").fetchone()[0],
            "pep_adverse_media": conn.execute("SELECT COUNT(*) FROM pep_adverse_media").fetchone()[0],
            "crypto_wallets": conn.execute("SELECT COUNT(*) FROM crypto_wallets").fetchone()[0],
        }
    finally:
        conn.close()

    return counts


if __name__ == "__main__":
    table_counts = build_watchlists_db()
    print(json.dumps(table_counts, indent=2))
