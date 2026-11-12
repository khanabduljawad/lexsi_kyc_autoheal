"""Phase 1/2 CLI: Curate FinCEN/FATF/OFAC SFT, DPO, and GRPO artifacts via CuratorKIT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.curator_pipeline import run_kyc_curatorkit_pipeline
from src.env_utils import load_project_env
from src.kyc_agent_runner import load_decide_pipeline_config


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description="Run CuratorKIT KYC data hygiene and export pipeline.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--live-curator", action="store_true", help="Enable live AdversarialPreferenceTask/HallucinationGate/RewardGate.")
    parser.add_argument("--max-split-cases", type=int, default=None)
    args = parser.parse_args()

    cfg = load_decide_pipeline_config()
    ck = cfg.get("curatorkit", {})
    out = run_kyc_curatorkit_pipeline(
        output_dir=args.out_dir or (PROJECT_ROOT / ck.get("output_dir", "artifacts/curated")),
        spacy_model=ck.get("spacy_model", "en_core_web_sm"),
        embedding_model=ck.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        similarity_threshold=float(ck.get("similarity_threshold", 0.96)),
        min_tokens=int(ck.get("min_tokens", 8)),
        max_tokens=int(ck.get("max_tokens", 4096)),
        max_split_cases=args.max_split_cases,
        live=args.live_curator,
        teacher_model=ck.get("teacher_model", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
        teacher_api_base=ck.get("teacher_api_base", "https://router.huggingface.co/v1"),
    )
    print(
        json.dumps(
            {
                "output_dir": str(out["output_dir"]),
                "passed_count": out["passed_count"],
                "rejected_count": out["rejected_count"],
                "export_file_counts": out["export_file_counts"],
                "stage_counts": out["stage_counts"],
                "manifest_path": str(out["manifest_path"]),
                "lexsi_provenance_path": str(out["lexsi_provenance_path"]),
                "auditkit_lineage_sidecar_path": str(out["auditkit_lineage_sidecar_path"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
