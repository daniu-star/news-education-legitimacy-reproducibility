# Legacy pipeline provenance

The scripts in this directory are historical source snapshots. They preserve
the analysis logic used during development, but they are not a drop-in public
rerun pipeline: the original scripts assume the private project layout around
the `analysis/` directory and require private databases, labels, embeddings,
and intermediate files that are intentionally not included here.

For manuscript-facing binary models, use:

```text
src/reproducible/final_analysis/run_formal_gee_models.py
```

For the exact formal model policy and input database provenance, see:

```text
docs/methods/final_model_policy.md
results_aggregate/models/formal_gee/model_registry_gee.json
```
