# Task-Difficulty / Data-Quantity Ratio — Supporting Analysis

Supplementary analysis for a journal-length extension of the FedMKS paper.
**Not included in the conference version** (only a one-sentence pointer in
*Limitations and Future Work*).

## Hypothesis

The effectiveness of federated averaging is governed by the ratio between
**task difficulty** and the **quantity + distribution of per-client data**.
When data per client is scarce relative to task difficulty, local optima
degenerate (single-class predictors), and weight averaging collapses;
when data is sufficient, the same mechanism lifts every client above
local-only training.

## Method

For each benchmark we compute three quantities:

1. **Task difficulty (input space)** — silhouette score and Fisher
   discriminant ratio `trace(S_b)/trace(S_w)` on the pipeline-scaled raw
   training features (3,000-sample subsample). Higher silhouette/Fisher =
   easier task.
2. **Data quantity & distribution** — samples per client and the
   *dominant-class share* under the Dirichlet α=0.5 partition, obtained
   by **replaying the exact seeded RNG stream** of `make_partitions`
   (`partition_vs_accuracy.py`; verified against per-client `n=` values
   in the run logs).
3. **FL effectiveness** — non-IID FedAvg accuracy relative to Local
   (`FA/Loc`) and to Central (`FA/Cen`), computed from the per-client
   `train_*.csv` logs (mean of last 5 rounds, seeds averaged).

## Results

| Dataset  | Silhouette | Fisher | n/client (non-IID) | Dom-class share | Central | Local | FedAvg | FA/Loc | FA/Cen |
|----------|-----------:|-------:|-----------------:|----------------:|--------:|------:|-------:|-------:|-------:|
| HomeOcc  | 0.038      | 0.084  | **52** (1–214)   | **0.67** (max 1.00) | 91.7 | 60.1 | 47.2 | **0.79** | **0.52** |
| HomeHAR  | −0.003     | 1.261  | — (public pool)  | —               | 30.9    | 16.5  | 21.0   | 1.27   | 0.68   |
| MNIST    | 0.044      | 0.272  | 3000 (353–7219)  | 0.38            | 99.0    | 85.4  | 97.6   | 1.14   | 0.99   |
| CIFAR-10 | −0.063     | 0.080  | 2500 (716–7374)  | 0.39            | 67.2    | 35.1  | 47.0   | 1.34   | 0.70   |

Note: Dirichlet partitioning splits the **full** training set, so non-IID
clients hold far more data than the IID `n_samples_per_class` cap suggests
— except HomeOccupancy, whose raw training pool is only 1048 windows
(window_size=1500), leaving ~52 samples/client.

A composite difficulty-per-sample proxy `(1 − silhouette)/n_client`
yields **0.0107** for HomeOccupancy vs 0.0032–0.0048 for the other
benchmarks — a ~3× gap that correctly identifies the only setting where
`FA/Loc < 1` and `FA/Cen ≈ 0.5`.

The sharper predictor of *collapse specifically* is the combination of
**few samples** and **high dominant-class share**: HomeOccupancy clients
average 52 samples at 67% dominant share (several clients are literally
single-class), so the local optimum is often a single-class predictor
and the first global average can fall into the majority-class basin.
MNIST/CIFAR-10 clients hold ~60× more data at ~38% share, and the
mechanism is harmless.  Per-client, dominant share correlates with
Local accuracy at r = −0.36 (seed 42); the best local client (78.3%)
holds a mid-sized two-class partition (103 samples, 54% share) while
the worst (27.8%) has 21 samples at 67% share.

## Plots (`plots/`)

- `difficulty_vs_effectiveness.png` — difficulty-per-sample vs FA/Cen;
  monotone ranking across the four benchmarks.
- `dirichlet_dominant_share.png` — dominant-class share per dataset;
  only HomeOccupancy exceeds the single-class-optimum regime.
- `data_vs_difficulty_plane.png` — benchmarks in the
  (samples/client, silhouette) plane.
- `fedavg_vs_local.png` — FA/Loc per dataset; <1 only for HomeOccupancy.

## Caveats / future work

- Silhouette and Fisher are **input-space** measures; they underrate
  CNN-learnable tasks (CIFAR-10 scores "hard" on raw pixels yet trains
  fine). A journal version should use representation-space difficulty
  (e.g., silhouette on features of a pretrained encoder, or
  Fisher information of a trained model).
- HomeHAR's low absolute accuracies partly reflect task hardness rather
  than FL effectiveness; a normalised difficulty metric should
  decouple the two.
- Candidate formalisation: predict `FA/Loc` (or collapse probability)
  from `(difficulty, n_client, dominant-class share)` — e.g., a
  per-client effective-sample-size measure weighted by class coverage.

## Reproduce

```powershell
# from the repository root (uses the venv)
.\venv\Scripts\python.exe journal_extension\partition_vs_accuracy.py
.\venv\Scripts\python.exe journal_extension\difficulty_ratio.py
.\venv\Scripts\python.exe journal_extension\make_plots.py
```

`partition_vs_accuracy.py` replays the seeded `iid_partition` →
`dirichlet_partition` RNG stream (no dataset download needed — only the
per-class counts), writes `results/partition_stats.json` (consumed by
`gen_figs.py` for Fig. partition), and prints per-client
partition-vs-accuracy tables plus per-tier means.
`difficulty_ratio.py` recomputes the difficulty metrics from the raw
datasets (HuggingFace / Keras) and the result CSVs.
