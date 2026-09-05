# FASTopic dominant-topic shares

The manuscript percentages 30.2% and 24.9% are dominant-topic shares.

For each of the 21,779 semantic-eligible comments, FASTopic returns a topic
probability vector `theta`. The assigned topic is:

```text
dominant_topic = argmax(theta, axis=1)
```

The share is then:

```text
share(topic k) = count(dominant_topic == k) / 21,779
```

The archived stage-5 result is:

| Topic | Count | Share | Manuscript value |
|---|---:|---:|---:|
| Employment and occupational potential | 6,568 | 30.1574% | 30.2% |
| AI replacement anxiety | 5,418 | 24.8772% | 24.9% |

The audit table is `results_aggregate/topics/fastopic_k10_dominant_topic_shares_legacy.csv`.
The recomputation script is `src/reproducible/recompute_fastopic_dominant_shares.py`.

The archived stage-5 retraining did not explicitly record a random seed. The
recomputation script therefore uses an explicit seed, records top words, and
writes metadata that compares the new run with the archived counts. Exact
reproduction should only be claimed after that comparison has passed. See
`results_aggregate/topics/FASTopic_REPRODUCIBILITY_NOTE.md`.
