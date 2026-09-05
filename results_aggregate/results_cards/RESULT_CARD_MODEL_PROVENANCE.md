# Result-card provenance

The existing result-card CSV is retained as a historical writing handoff. Its
binary regression evidence was generated from the earlier cluster-robust GLM
tables. It must not be presented as a GEE result card without rerunning the
result-card builder against the files in `results_aggregate/models/formal_gee/`.

The formal model policy and runner are:

- `docs/methods/final_model_policy.md`
- `src/reproducible/final_analysis/run_formal_gee_models.py`
- `results_aggregate/models/model_registry_formal_gee.csv`

This distinction is intentional: the package preserves an auditable historical
record instead of silently changing published coefficient values.
