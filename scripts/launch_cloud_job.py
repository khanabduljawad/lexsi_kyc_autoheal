"""Safe launcher for Hugging Face Jobs GPU training via AgentTune and hf CLI.

Auto-loads HF_TOKEN from .env, validates local datasets and volume mounts,
and submits the UV script to Hugging Face cloud infrastructure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env_utils import load_project_env


def main() -> None:
    load_project_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN is not set in .env or environment.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Launch AutoHeal-KYC training on Hugging Face Jobs.")
    parser.add_argument(
        "--mode",
        choices=["distill_sft", "agentic_dpo_offline", "agentic_dpo_rollout", "agentic_grpo"],
        default="distill_sft",
        help="Training mode.",
    )
    parser.add_argument("--flavor", default="l4x1", help="Hardware flavor (e.g. l4x1, a10g-small, a100-large).")
    parser.add_argument("--base-model", default="CohereLabs/tiny-aya-global", help="Base model repo ID.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Local dataset path (relative to PROJECT_ROOT) or HF Hub dataset repo ID.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta.")
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length.")
    parser.add_argument("--push-to-hub", type=str, default=None, help="Optional HF repo ID to push adapter.")
    parser.add_argument("--timeout", type=str, default="2h", help="Job max duration timeout (default: 2h).")
    parser.add_argument("--detach", action="store_true", help="Submit job and return immediately without streaming.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without submitting.")
    args = parser.parse_args()

    # Determine default dataset path if not specified
    default_datasets = {
        "distill_sft": "artifacts/curated/sft_messages.jsonl",
        "agentic_dpo_offline": "artifacts/dpo_pairs/dpo_train.jsonl",
        "agentic_dpo_rollout": "artifacts/curated/grpo.jsonl",
        "agentic_grpo": "artifacts/curated/grpo.jsonl",
    }
    dataset_rel = args.dataset or default_datasets[args.mode]
    local_dataset_path = PROJECT_ROOT / dataset_rel
    if not local_dataset_path.exists() and not ("/" in dataset_rel and not dataset_rel.endswith(".jsonl")):
        print(f"ERROR: Dataset not found locally at {local_dataset_path}", file=sys.stderr)
        sys.exit(1)

    # In container, volume mounts map ./artifacts -> /artifacts and ./data -> /kyc_data
    rel_clean = dataset_rel.replace("\\", "/")
    if local_dataset_path.exists():
        if rel_clean.startswith("data/"):
            container_dataset = f"/kyc_data/{rel_clean[5:]}"
        else:
            container_dataset = f"/{rel_clean}"
    else:
        container_dataset = dataset_rel

    output_dir = f"autoheal_kyc_{args.mode}_adapter"
    push_repo = args.push_to_hub or f"obluda420/autoheal-kyc-tiny-aya-{args.mode.replace('_', '-')}"

    hf_exe = sys.executable.replace("python.exe", "hf.exe")
    if not Path(hf_exe).exists():
        hf_exe = "hf"

    cmd = [
        hf_exe,
        "jobs",
        "uv",
        "run",
        "--flavor",
        args.flavor,
        "-e",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "--timeout",
        args.timeout,
        "--secrets",
        "HF_TOKEN",
        "-v",
        "./artifacts:/artifacts",
        "-v",
        "./data:/kyc_data",
    ]
    if args.detach:
        cmd.append("--detach")
    if args.dry_run:
        cmd.append("--dry-run")

    cmd.extend(
        [
            "jobs/train_agenttune_job.py",
            "--mode",
            args.mode,
            "--base-model",
            args.base_model,
            "--dataset",
            container_dataset,
            "--output-dir",
            output_dir,
            "--push-to-hub",
            push_repo,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum",
            str(args.grad_accum),
            "--lr",
            str(args.lr),
            "--beta",
            str(args.beta),
            "--max-length",
            str(args.max_length),
        ]
    )

    print(f"Launching {args.mode} on HF Jobs (hardware: {args.flavor})...")
    print(f"Base model: {args.base_model}")
    print(f"Container dataset: {container_dataset}")
    print(f"Target Hub Repo: {push_repo}")
    print(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=os.environ)
        sys.exit(res.returncode)

    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=os.environ)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
