# Formal GEE aggregate outputs

These tables were generated from the controlled private analysis database by
`src/reproducible/final_analysis/run_formal_gee_models.py`.

- Binary family: binomial GEE with logit link.
- Working correlation: exchangeable.
- Cluster: `note_id`.
- RQ1-RQ4 binary outputs are included when the controlled private relation and
  ability files are supplied to the runner.
- Counts: not included here; count outcomes remain negative-binomial models.
- Row-level data, fitted values, and individual identifiers are not included.

The files under `results_aggregate/models/rq1/` and `rq4/` whose names contain
`cluster_glm` are historical outputs and are not interchangeable with these
tables.
