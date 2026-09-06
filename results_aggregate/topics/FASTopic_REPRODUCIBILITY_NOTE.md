# FASTopic reproducibility note

The archived manuscript percentages are preserved in
`fastopic_k10_dominant_topic_shares_legacy.csv` because the original stage-5
retraining did not record its random seed. The table is therefore a faithful
aggregate record of the run that produced the manuscript values, but the old
run is not byte-level reproducible from the surviving metadata alone.

The accompanying script `src/reproducible/recompute_fastopic_dominant_shares.py`
sets an explicit seed, records top words, identifies topic labels from those
top words, and writes a new aggregate table. The script must be run before a
claim of exact reproduction is made. If the deterministic run differs from the
legacy table, the manuscript should either retain the legacy values with this
provenance limitation disclosed or be updated to the deterministic result.

## Controlled audit result

The script was run locally on the controlled 21,779-comment corpus with
`seed=17` and `epochs=100`. The resulting dominant-topic counts were:

```text
topic 0: 4507
topic 1: 1224
topic 2: 602
topic 3: 756
topic 4: 3372
topic 5: 4315
topic 6: 1336
topic 7: 608
topic 8: 2860
topic 9: 2199
```

This does not reproduce the archived `topic 8 = 5418` and `topic 9 = 6568`
counts. Under the deterministic run, the keyword audit identified only topic 5
as an employment-related topic and did not identify a separate AI topic. The
legacy 30.2% / 24.9% values therefore remain historical unseeded-run values,
not deterministic results that can currently be regenerated from the public
metadata.
