# Formal GEE provenance audit

The formal GEE aggregate tables in this release were regenerated with
`src/reproducible/final_analysis/run_formal_gee_models.py` using the explicit
`NEWS_EDU_DB` path to the frozen analysis database.

The machine-readable provenance is stored in:

```text
results_aggregate/models/formal_gee/model_registry_gee.json
```

That file records the input database SHA-256, the semantic/non-orphan comment
sample size, stance-state counts, and the formal GEE model registry.

The historical cluster-robust GLM tables use a different label/model version
and must not be used to infer the formal GEE sample size or coefficients.
