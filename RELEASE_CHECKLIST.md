# Release Checklist

Before pushing this ZIP to a public GitHub repository:

- [ ] Rotate/revoke the credential that was hard-coded in the original private package.
- [x] Confirm the formal binary model policy: manuscript-facing models use note-clustered exchangeable GEE; count outcomes remain negative-binomial models.
- [x] Label archived `*_cluster_glm.csv` files as historical outputs and do not cite them as GEE estimates.
- [x] Document the derivation of the paper-level FASTopic percentages: dominant-topic counts divided by 21,779.
- [x] Run `src/reproducible/final_analysis/run_formal_gee_models.py` against the explicit frozen database and publish its formal GEE aggregate outputs with database SHA-256 provenance.
- [ ] Replace the historical writing-handoff result-card CSV with a result-card build that reads only `results_aggregate/models/formal_gee/` before citing OR/CI values in the final manuscript.
- [x] Run `src/reproducible/recompute_fastopic_dominant_shares.py` with `seed=17`, `epochs=100`, and compare its counts against the archived audit table.
- [ ] Resolve the FASTopic discrepancy before claiming that 30.2% / 24.9% are deterministically reproducible; the current audit documents them as historical unseeded-run values.
- [ ] Choose an explicit software license; none is assigned by this cleaning step.
- [ ] Update citation metadata/title after the manuscript title is frozen.
- [ ] Run the privacy/secret scans described in `SANITIZATION_REPORT.md` after any new files are added.
