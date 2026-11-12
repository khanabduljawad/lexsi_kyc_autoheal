---
tags:
- curatorkit
configs:
- config_name: sft_messages
  default: true
  description: curatorkit sft_messages export, sha256 dbe6885f0b73dc3799c75c3238e0ea82d42fb1038d24dd6041f6c20e096fa8be
  data_files:
  - split: train
    path: sft_messages.jsonl
- config_name: dpo
  description: curatorkit dpo export, sha256 051dd49782f9cea5396a8fd45133ead04fa69e6e2d258fec97cee7ec49021e26
  data_files:
  - split: train
    path: dpo.jsonl
- config_name: grpo
  description: curatorkit grpo export, sha256 40788dbf1792a04768e0508caa4e529da45f5acecc1ee20d2ff732ea43135f1a
  data_files:
  - split: train
    path: grpo.jsonl
---

# Dataset Card: autoheal_kyc_curated

## Summary

- **Run timestamp**: 2026-11-12T14:36:03.849340Z
- **Wall clock time**: 51.0s
- **Pipeline config hash**: `905a2da3592a44ff`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| KYCSeedReader | 529 | 529 | — | 0 | 529 |
| _WatchlistSafeSecretsGate | 529 | 527 | 0 | 2 | 527 |
| _CachedSysPromptPIIPseudonymizer | 527 | 527 | — | — | 527 |
| SchemaGate | 527 | 527 | 0 | 0 | 527 |
| OfflineSafeDiversityGate | 527 | 527 | 0 | 0 | 527 |
| MessagesExporter | — | 527 | — | — | 527 |
| DPOExporter | — | 527 | — | — | 527 |
| GRPOExporter | — | 527 | — | — | 527 |

## Gate Pass Rates and Rejection Breakdown

- **KYCSeedReader**: 529/0 passed (N/A), 0 rejected
- **_WatchlistSafeSecretsGate**: 527/529 passed (99.6%), 2 rejected
- **SchemaGate**: 527/527 passed (100.0%), 0 rejected
- **OfflineSafeDiversityGate**: 527/527 passed (100.0%), 0 rejected


**Rejection reasons:**
  - `secret_detected:Hex High Entropy String`: 2

## Deduplication

  - No deduplication step recorded

## Known Limitations

- No MinHash deduplication applied.
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.25
- **python**: 3.12.15

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `sft_messages` | train (207) | `messages`: list of {role, content}; role is system, user or assistant |
| `dpo` | train (160) | `prompt`: string, or list of {role, content}<br>`chosen`: string, or list of {role, content}<br>`rejected`: string, or list of {role, content} |
| `grpo` | train (160) | `prompt`: string<br>`responses`: list of string<br>`rewards`: list of float<br>`reference`: string; gold answer, present only when the input rows carried one |
