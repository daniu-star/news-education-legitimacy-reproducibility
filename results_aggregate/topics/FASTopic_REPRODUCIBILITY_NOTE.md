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
