# Sanitization Report

## Removed categories

1. All SQLite databases and database sidecar files.
2. Raw note/comment corpora and full LLM label JSONL.
3. Row-level relation frames, reply pairs and adjudication outputs.
4. Note-level framework projection containing note IDs/titles.
5. Ability mention long tables and audit samples containing comment IDs.
6. Original package ZIPs and manifests that enumerate/hash private files.
7. Author-recovery and author-inference scripts designed to recover platform identities.
8. Anonymization secrets, API credentials and local absolute runtime paths.

## Code changes

- Replaced the hard-coded DeepSeek credential with `DEEPSEEK_API_KEY`.
- Replaced private `/mnt/data/...` roots in reproducible scripts with `NEWS_EDU_ROOT`.
- Disabled the private author-recovery enrichment block in the public governance script.
- Replaced explicitly marked raw-text Codebook examples with privacy-preserving paraphrases while retaining the same coding logic.
- Regenerated a manifest for the public-safe package only.

## Data policy

Only aggregate outputs that do not expose individual note/comment/user records are included under `results_aggregate/`.
