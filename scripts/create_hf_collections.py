"""Create and populate official Hugging Face Collections for AutoHeal-KYC Track 3 submission."""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import HfApi
from src.env_utils import load_project_env


def main() -> None:
    load_project_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    # 1. Models Collection
    print("Creating/updating Models Collection...")
    try:
        model_col = api.create_collection(
            title="AutoHeal-KYC Models (Track 3)",
            namespace="obluda420",
            description="Trained & self-healed Cohere Tiny Aya models for KYC compliance (Track 3: Agentic Reasoning).",
            private=False,
            exists_ok=True,
        )
    except Exception as e:
        print(f"Collection creation notice: {e}")
        cols = api.list_collections(owner="obluda420")
        model_col = next((c for c in cols if "model" in c.slug.lower() or "autoheal" in c.slug.lower()), None)

    print(f"Model Collection Slug: {model_col.slug}")
    print(f"Model Collection URL: {model_col.url}")

    # Add SFT and DPO models
    models_to_add = [
        ("obluda420/autoheal-kyc-tiny-aya-distill-sft", "Student SFT distillation model trained with AgentTune create_distill_trainer on 187 curated KYC trajectories."),
        ("obluda420/autoheal-kyc-tiny-aya-dpo", "Self-healed closed-loop model trained with AgentTune TrlAgenticDPO on 189 replay-validated preference pairs."),
    ]
    for m_id, note in models_to_add:
        try:
            api.add_collection_item(
                collection_slug=model_col.slug,
                item_id=m_id,
                item_type="model",
                note=note,
                exists_ok=True,
            )
            print(f"Added model {m_id} to collection.")
        except Exception as e:
            print(f"Notice adding model {m_id}: {e}")

    # 2. Datasets Collection
    print("\nCreating/updating Datasets Collection...")
    try:
        ds_col = api.create_collection(
            title="AutoHeal-KYC Datasets (Track 3)",
            namespace="obluda420",
            description="Curated zero-leakage KYC splits and CuratorKIT trajectories (SFT, DPO, GRPO) with oracle labels.",
            private=False,
            exists_ok=True,
        )
    except Exception as e:
        print(f"Collection creation notice: {e}")
        cols = api.list_collections(owner="obluda420")
        ds_col = next((c for c in cols if "dataset" in c.slug.lower()), None)

    print(f"Dataset Collection Slug: {ds_col.slug}")
    print(f"Dataset Collection URL: {ds_col.url}")

    try:
        api.add_collection_item(
            collection_slug=ds_col.slug,
            item_id="obluda420/autoheal-kyc-curated-dataset",
            item_type="dataset",
            note="Full curated zero-leakage splits (A/B/C), SFT messages, DPO pairs, GRPO prompts, and CuratorKIT manifest.",
            exists_ok=True,
        )
        print("Added dataset obluda420/autoheal-kyc-curated-dataset to collection.")
    except Exception as e:
        print(f"Notice adding dataset: {e}")

    print("\n--- Summary of Official Collection URLs ---")
    print(f"Model Collection:   {model_col.url}")
    print(f"Dataset Collection: {ds_col.url}")


if __name__ == "__main__":
    main()
