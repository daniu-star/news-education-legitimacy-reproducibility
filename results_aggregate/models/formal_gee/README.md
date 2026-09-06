# Formal GEE aggregate outputs

These tables were generated from the controlled frozen analysis database by
`src/reproducible/final_analysis/run_formal_gee_models.py`.

- Binary family: binomial GEE with logit link.
- Working correlation: exchangeable.
- Cluster: `note_id`.
- RQ1-RQ4 binary outputs are included when the controlled relation and ability
  files are supplied to the runner.
- Count outcomes remain separate negative-binomial models.
- The input database SHA-256 and stance-state counts are recorded in
  `model_registry_gee.json`.

The files under the historical `rq1/` and `rq4/` folders whose names contain
`cluster_glm` are not interchangeable with these tables.
