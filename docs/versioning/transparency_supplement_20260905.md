# Transparency supplement 2026-09-05

This supplement resolves two release-blocking provenance issues.

## Statistical model policy

- Manuscript-facing binary outcomes are defined as note-clustered binomial GEE
  models with an exchangeable working correlation.
- Count outcomes remain negative-binomial models.
- Historical `cluster_glm` tables are retained and explicitly labeled as
  historical rather than silently renamed.
- A standalone formal GEE runner and newly generated aggregate RQ1/RQ4 GEE
  coefficient tables are included.

## FASTopic percentages

- The 30.2% and 24.9% manuscript figures are documented as dominant-topic
  shares, calculated from `argmax(theta)` assignments over 21,779 comments.
- The archived counts are published as an aggregate audit table: 6,568 and
  5,418 comments respectively.
- A deterministic recomputation script is included.
- The original stage-5 retraining did not record its seed, so the release does
  not overstate exact reproducibility of that historical run.

## Packaging

- Mojibake filenames inherited from the earlier ZIP were repaired to readable
  UTF-8 Chinese filenames.
- The manifest is regenerated after all changes.
