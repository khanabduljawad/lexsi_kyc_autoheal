"""Safe launcher for Hugging Face Jobs GPU evaluation via AuditKIT and hf CLI.

Evaluates baseline or fine-tuned adapters on held-out Split C (60 cases)
with full multi-trial reliability (pass@k), CustomPredicates, and performance telemetry.
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

    parser = argparse.ArgumentParser(description="Launch AutoHeal-KYC evaluation on Hugging Face Jobs.")
    parser.add_argument("--adapter-repo", type=str, default="obluda420/autoheal-kyc-tiny-aya-dpo", help="Adapter HF repo ID or local path (omit for baseline).")
    parser.add_argument("--base-model", type=str, default="CohereLabs/tiny-aya-global", help="Base model repo ID.")
    parser.add_argument("--flavor", default="l4x1", help="Hardware flavor (default: l4x1).")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials for pass@k.")
    parser.add_argument("--max-cases", type=int, default=60, help="Max test cases from Split C (default: 60).")
    parser.add_argument("--skip-bfcl", action="store_true", default=True, help="Skip BFCL benchmark.")
    parser.add_argument("--output-json", type=str, default="eval_results.json", help="Output results file name.")
    parser.add_argument("--push-eval-repo", type=str, default=None, help="Optional HF dataset repo to publish eval rows.")
    parser.add_argument("--detach", action="store_true", help="Submit job and return immediately.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without submitting.")
    args = parser.parse_args()

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
        "--secrets",
        "HF_TOKEN",
        "-v",
        "./data:/kyc_data",
    ]
    if args.detach:
        cmd.append("--detach")
    if args.dry_run:
        cmd.append("--dry-run")

    eval_args = [
        "jobs/eval_hf_job.py",
        "--base-model",
        args.base_model,
        "--split-c",
        "/kyc_data/splits/split_c.jsonl",
        "--watchlist-db",
        "/kyc_data/watchlists.db",
        "--trials",
        str(args.trials),
        "--max-cases",
        str(args.max_cases),
        "--output-json",
        args.output_json,
    ]
    if args.adapter_repo:
        eval_args.extend(["--adapter-repo", args.adapter_repo])
    if args.skip_bfcl:
        eval_args.append("--skip-bfcl")
    if args.push_eval_repo:
        eval_args.extend(["--push-eval-repo", args.push_eval_repo])

    cmd.extend(eval_args)

    label_target = args.adapter_repo or f"{args.base_model}-baseline"
    print(f"Launching AuditKIT evaluation on HF Jobs ({args.flavor})...")
    print(f"Target: {label_target}")
    print(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=os.environ)
        sys.exit(res.returncode)

    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=os.environ)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
