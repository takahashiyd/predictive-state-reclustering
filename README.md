# Predictive-state reclustering

Code and data to reproduce the results in *"Unsupervised discovery of vocal categories from
their sequences."* The method operates on a microcluster-labelled call sequence: it merges
microclusters by predictive equivalence and selects the number of categories by held-out
prediction of the next microcluster.

## Contents

- `data/microclusters.csv.gz` — microcluster-labelled marmoset call sequences.
- `reproduce_reclustering.py` — predictive-state reclustering and model selection.
- `finch/` — the second-species (Bengalese finch) pipeline.
- `requirements.txt`, `LICENSE`.

## Data

`data/microclusters.csv.gz`, one row per call unit:

| column | description |
|---|---|
| `session_id` | recording identifier |
| `order` | position of the unit within the recording |
| `split` | `train` or `test` (split by recording) |
| `animal_id` | individual identifier |
| `call_type` | expert label |
| `micro` | microcluster label (0–63) |
| `sound_cluster` | sound-only clustering label |

## Reproduce

```
pip install -r requirements.txt
python reproduce_reclustering.py
```

Reports held-out prediction of the next microcluster for the expert labels, a sound-only
clustering, the microcluster ceiling, and the discovered predictive states across K. The
discovered categories reach 3.099 nats at K=11.

## Apply to your own data

Cluster your acoustic features into microclusters (e.g. 64 k-means groups) and write a
`data/microclusters.csv.gz` with the columns above (`sound_cluster` and `animal_id` optional).
Then run `python reproduce_reclustering.py`.

## Second species (Bengalese finch)

`finch/` reproduces the finch analysis from the public BirdsongRecognition corpus
(Koumura & Okanoya 2016, figshare 3470165):

```
python finch/run_all_finches.py       # downloads the corpus and writes finch/finch_results.csv
python finch/finch_figures.py Bird5   # per-bird figures
```

The finch pipeline is the same method with a log-mel front-end. It downloads the corpus, forms
microclusters, and reclusters them by predictive equivalence.

## Citation

De Sales AGSMM, Neto JFR, Takahashi DY. Unsupervised discovery of vocal categories from their
sequences.

## License

MIT. See `LICENSE`.
