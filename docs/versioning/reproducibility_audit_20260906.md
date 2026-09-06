# Reproducibility audit 2026-09-06

This audit compares the public package with the controlled analysis materials.

## Confirmed matches

- The semantic/non-orphan comment sample is 21,779 in both controlled databases.
- Formal binary models use binomial GEE with an exchangeable working
  correlation and `note_id` as the cluster.
- Count models remain negative-binomial models with cluster-robust covariance.
- The archived FASTopic counts are internally closed:
  `6568 + 5418 + all other topic counts = 21779`.

## Corrections made in this release

- Formal GEE tables were regenerated from the explicit frozen database
  `00_freeze/analysis_v2_frozen.db`.
- The frozen database SHA-256 is recorded in
  `results_aggregate/models/formal_gee/model_registry_gee.json`.
- `NEWS_EDU_DB` can now override database discovery, preventing a silent fall
  back from the frozen database to a different `analysis.db`.
- The formal runner now uses the same RQ2 positive-case threshold (20) and the
  same six RQ3 properties as the integrated final pipeline.
- Legacy scripts are explicitly documented as historical snapshots; they are
  not claimed to be drop-in runnable without the private project layout.

## Remaining non-reproducibility

The original Stage 5 FASTopic retraining did not record its random seed. A
controlled rerun with `seed=17` and `epochs=100` produced dominant-topic counts
`4507, 1224, 602, 756, 3372, 4315, 1336, 608, 2860, 2199`, rather than the
archived topic-8/topic-9 counts `5418` and `6568`. Therefore the manuscript
values 30.2% and 24.9% remain historical unseeded-run values. They are
traceable through the archived aggregate table and the published script, but
they must not be described as deterministic exact reproduction until the
manuscript is revised or the original model artifact is recovered.

Historical cluster-robust GLM tables also remain in the package for audit
history. They are not interchangeable with the formal GEE tables.
