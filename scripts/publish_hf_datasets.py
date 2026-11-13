"""Upload curated KYC datasets and manifest to Hugging Face Hub under obluda420."""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import HfApi, upload_file, upload_folder
from src.env_utils import load_project_env

DATASET_REPO = "obluda420/autoheal-kyc-curated-dataset"

DATASET_README = """---
license: cc-by-nc-4.0
task_categories:
  - text-generation
language:
  - en
tags:
  - agentic-reasoning
  - self-healing
  - kyc-compliance
  - fincen-cip
  - ofac-sanctions
  - fatf-aml
  - agenttune
  - curatorkit
  - auditkit
  - cohere-aya
size_categories:
  - n<1K
---

# AutoHeal-KYC: Curated FinCEN/FATF/OFAC Compliance & Self-Healing Dataset

This dataset provides the curated training, validation, and held-out evaluation splits for **AutoHeal-KYC**, an autonomous self-healing agent submitted to **Track 3: Agentic Reasoning** (*Business workflow & decision agents* and *Self-healing agents*).

## Overview

AutoHeal-KYC enforces strict deterministic compliance standards:
- **FinCEN CIP Gate (31 CFR 1020.220):** Customer name, valid ISO-3166-1 alpha-2 country, valid YYYY-MM-DD date of birth, and tax/national ID. Abstains from tool execution when mandatory CIP elements are missing.
- **OFAC & International Sanctions Screening:** Full-name and fuzzy-alias screening against 6,000 global sanctioned entities.
- **FATF High-Risk & Tax Secrecy Jurisdictions:** Enhanced Due Diligence (EDD) escalation for FATF blacklist (KP, IR, MM), grey-list, sectoral, and offshore secrecy havens.
- **PEP & Adverse Media Screening:** Identifies PEP tier 1/2 political exposure and adverse financial crime media.
- **OFAC Sanctioned Crypto Wallets:** Validates Bitcoin, Ethereum, Tron, and EVM addresses against OFAC Specially Designated Nationals (SDN) digital currency addresses.
- **Source of Wealth (SoW) Verification:** Verifies asset origins for declared assets $\\ge \\$250,000$ or monthly transactions $\\ge \\$50,000$.

## Zero-Leakage Disjoint Splits

All customer entities and synthetic personas are strictly disjoint across splits:
- **Split A (`split_a.jsonl`, 40 cases):** Seed trajectories for initial bootstrapping.
- **Split B (`split_b.jsonl`, 120 cases):** Multi-turn agent rollouts used for baseline failure elicitation, teacher repair synthesis, and DPO training.
- **Split C (`split_c.jsonl`, 60 cases):** Held-out evaluation split never seen during training or distillation (18 clean approvals, 15 sanctioned, 11 PEP, 6 sanctioned crypto wallets, 10 incomplete CIP cases).

## Curated Trajectory Files

Curated using **CuratorKIT** with `SchemaGate`, `SecretsGate`, `PIIPseudonymizer`, and `DiversityGate`:
- **`sft_messages.jsonl` (187 trajectories):** Curated multi-turn reasoning and tool-calling dialogues for teacher-to-student SFT distillation (`create_distill_trainer`).
- **`dpo.jsonl` & `dpo_combined.jsonl` (189 pairs):** Replay-validated prompt/chosen/rejected preference pairs repairing tool loops, missing required checks, premature approvals, and hallucinated verdicts.
- **`grpo.jsonl` (160 cases):** Verifiable policy prompts with required/forbidden tool constraints and oracle ground-truth targets for online reinforcement learning (`TrlAgenticGrpo`).
- **`manifest.json`:** SHA-256 provenance manifest with gate rejection audit and pipeline configuration.

## Licensing

Licensed under **CC-BY-NC-4.0**, compatible with the Cohere Aya open-weight model license terms.
"""


def main() -> None:
    load_project_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True, private=False)

    temp_readme = PROJECT_ROOT / "artifacts" / "DATASET_README.md"
    temp_readme.write_text(DATASET_README, encoding="utf-8")

    files_to_upload = [
        (temp_readme, "README.md"),
        (PROJECT_ROOT / "data" / "splits" / "split_a.jsonl", "splits/split_a.jsonl"),
        (PROJECT_ROOT / "data" / "splits" / "split_b.jsonl", "splits/split_b.jsonl"),
        (PROJECT_ROOT / "data" / "splits" / "split_c.jsonl", "splits/split_c.jsonl"),
        (PROJECT_ROOT / "artifacts" / "curated" / "sft_messages.jsonl", "curated/sft_messages.jsonl"),
        (PROJECT_ROOT / "artifacts" / "curated" / "dpo.jsonl", "curated/dpo.jsonl"),
        (PROJECT_ROOT / "artifacts" / "curated" / "grpo.jsonl", "curated/grpo.jsonl"),
        (PROJECT_ROOT / "artifacts" / "dpo_pairs" / "dpo_combined.jsonl", "dpo_pairs/dpo_combined.jsonl"),
        (PROJECT_ROOT / "artifacts" / "curated" / "manifest.json", "curated/manifest.json"),
    ]

    for local_p, hub_p in files_to_upload:
        if local_p.exists():
            print(f"Uploading {local_p.name} -> {hub_p}...")
            api.upload_file(
                path_or_fileobj=str(local_p),
                path_in_repo=hub_p,
                repo_id=DATASET_REPO,
                repo_type="dataset",
                commit_message=f"feat(data): add {hub_p}",
            )
        else:
            print(f"Warning: {local_p} does not exist, skipping.")

    print(f"Dataset successfully published to: https://huggingface.co/datasets/{DATASET_REPO}")


if __name__ == "__main__":
    main()
