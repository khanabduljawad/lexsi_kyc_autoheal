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
