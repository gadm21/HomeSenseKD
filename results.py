#!/usr/bin/env python3
"""
results.py — Compare FL results across heterogeneity distributions for one algorithm.

Reads per-client training CSVs produced by run_fedkd.py:
  <results_dir>/<dataset>/<algorithm>/<distribution>/<setting>/train_N.csv

Distributions compared : all_small | uniform | skewed
Settings               : iid | noniid

Produces four figures:
  1. <dataset>_<algo>_curves.png         — training curves (avg ± std) per distribution
  2. <dataset>_<algo>_final_bars.png     — final accuracy bar chart (IID vs non-IID)
  3. <dataset>_<algo>_client_dist.png    — per-client accuracy box-plots
  4. <dataset>_<algo>_iid_vs_noniid.png  — IID vs non-IID side-by-side per distribution

Usage:
    python results.py
    python results.py --results-dir results/phase2 --dataset home_occupancy --algorithm fedmd
    python results.py --algorithm fedmd --local-epochs 4 --out-dir results/phase2/figures
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────

DISTRIBUTIONS  = ["all_small", "uniform", "skewed"]
SETTINGS       = ["iid", "noniid"]
DIST_LABELS    = {"all_small": "All-Small", "uniform": "Uniform", "skewed": "Skewed"}
SETTING_LABELS = {"iid": "IID", "noniid": "Non-IID"}

DIST_COLORS    = {
    "all_small": "#e07b54",
    "uniform":   "#4c8fbd",
    "skewed":    "#5cb85c",
}
SETTING_COLORS = {
    "iid":    "#4c8fbd",
    "noniid": "#e07b54",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Distribution comparison across heterogeneity modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--results-dir",  default="results/phase2",
                   help="Root results directory (default: results/phase2)")
    p.add_argument("--dataset",      default="home_occupancy",
                   help="Dataset subdirectory name (default: home_occupancy)")
    p.add_argument("--algorithm",    default="fedmd",
                   help="Algorithm subdirectory name (default: fedmd)")
    p.add_argument("--local-epochs", type=int, default=4,
                   help="Local epochs per FL round used during training (default: 4). "
                        "Used to split the per-client CSV into per-round groups.")
    p.add_argument("--out-dir",      default=None,
                   help="Output directory for figures (default: <results_dir>/figures)")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_client_csv(path: str, local_epochs: int):
    """Return (round_val_accs, round_train_accs) arrays for one client CSV.

    Each FL round appends `local_epochs` rows with epoch indices 0..local_epochs-1.
    We take the last epoch of each round as the round's representative accuracy.
    Returns empty arrays if the file is missing or malformed.
    """
    try:
        df = pd.read_csv(path)
        if "val_accuracy" not in df.columns or len(df) == 0:
            return np.array([]), np.array([])
        n_rounds = len(df) // local_epochs
        if n_rounds == 0:
            return np.array([]), np.array([])
        df = df.iloc[: n_rounds * local_epochs]
        val_acc   = df["val_accuracy"].values.reshape(n_rounds, local_epochs)[:, -1]
        train_acc = df["accuracy"].values.reshape(n_rounds, local_epochs)[:, -1]
        return val_acc, train_acc
    except Exception:
        return np.array([]), np.array([])


def collect_results(results_dir: str, dataset: str, algorithm: str,
                    local_epochs: int) -> dict:
    """Load all per-client CSVs and aggregate into per-round statistics.

    Returns
    -------
    data : dict  {dist: {setting: stats_dict}}

    stats_dict keys:
        round_accs  — np.ndarray (n_clients, n_rounds)  per-round val accuracy
        avg, std, min, max — np.ndarray (n_rounds,)     across clients
        final_accs  — list[float]  last-round acc per client
        n_rounds    — int
        n_clients   — int
    """
    data = {}
    for dist in DISTRIBUTIONS:
        data[dist] = {}
        for setting in SETTINGS:
            folder = os.path.join(results_dir, dataset, algorithm, dist, setting)
            if not os.path.isdir(folder):
                continue

            client_arrays = []
            cid = 0
            while True:
                fp = os.path.join(folder, f"train_{cid}.csv")
                if not os.path.exists(fp):
                    break
                val_acc, _ = _load_client_csv(fp, local_epochs)
                if len(val_acc):
                    client_arrays.append(val_acc)
                cid += 1

            if not client_arrays:
                continue

            # Align to the shortest completed run
            min_rounds = min(len(a) for a in client_arrays)
            arr = np.array([a[:min_rounds] for a in client_arrays])  # (C, R)

            data[dist][setting] = {
                "round_accs": arr,
                "avg":        arr.mean(axis=0),
                "std":        arr.std(axis=0),
                "min":        arr.min(axis=0),
                "max":        arr.max(axis=0),
                "final_accs": arr[:, -1].tolist(),
                "n_rounds":   min_rounds,
                "n_clients":  len(client_arrays),
            }

    return data


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(data: dict, algorithm: str):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Summary: {algorithm.upper()}")
    print(f"{sep}")
    header = f"  {'Distribution':14s} {'Setting':9s} | {'clients':>7}  {'rounds':>6} | "
    header += f"{'mean acc':>9}  {'std':>7}  {'min':>7}  {'max':>7}"
    print(header)
    print(f"  {'-'*68}")
    for dist in DISTRIBUTIONS:
        for setting in SETTINGS:
            d = data.get(dist, {}).get(setting)
            if d is None:
                print(f"  {DIST_LABELS[dist]:14s} {SETTING_LABELS[setting]:9s} | "
                      f"  — no data —")
                continue
            fa = d["final_accs"]
            print(f"  {DIST_LABELS[dist]:14s} {SETTING_LABELS[setting]:9s} | "
                  f"{d['n_clients']:>7}  {d['n_rounds']:>6} | "
                  f"{np.mean(fa):>9.4f}  {np.std(fa):>7.4f}  "
                  f"{np.min(fa):>7.4f}  {np.max(fa):>7.4f}")
    print(f"{sep}\n")


# ── Figure 1 — Training curves ────────────────────────────────────────────────

def plot_training_curves(data: dict, algorithm: str, save_path: str):
    """One column per setting (IID / Non-IID); one line per distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, setting in zip(axes, SETTINGS):
        plotted = False
        for dist in DISTRIBUTIONS:
            d = data.get(dist, {}).get(setting)
            if d is None:
                continue
            rounds = np.arange(1, d["n_rounds"] + 1)
            col    = DIST_COLORS[dist]
            ax.plot(rounds, d["avg"],
                    label=DIST_LABELS[dist], color=col, lw=2)
            ax.fill_between(rounds,
                            d["avg"] - d["std"],
                            d["avg"] + d["std"],
                            alpha=0.15, color=col)
            plotted = True

        ax.set(
            title=f"{algorithm.upper()} — {SETTING_LABELS[setting]}",
            xlabel="FL Round",
            ylabel="Avg client val-accuracy",
            xlim=(1, None),
            ylim=(0, 1),
        )
        if plotted:
            ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"{algorithm.upper()} — Training Curves (mean ± std across clients)\n"
        f"Heterogeneity distribution comparison",
        fontsize=13,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


# ── Figure 2 — Final accuracy bar chart ──────────────────────────────────────

def plot_final_accuracy_bars(data: dict, algorithm: str, save_path: str):
    """Grouped bars: distributions on x-axis, IID / non-IID as two groups."""
    dists_avail = [d for d in DISTRIBUTIONS if data.get(d)]
    if not dists_avail:
        return

    n = len(dists_avail)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 2.8), 5))

    for k, setting in enumerate(SETTINGS):
        means, stds = [], []
        for dist in dists_avail:
            d = data.get(dist, {}).get(setting)
            if d:
                means.append(float(np.mean(d["final_accs"])))
                stds.append(float(np.std(d["final_accs"])))
            else:
                means.append(0.0)
                stds.append(0.0)

        bars = ax.bar(
            x + (k - 0.5) * w, means, w,
            yerr=stds, capsize=5,
            alpha=0.85,
            label=SETTING_LABELS[setting],
            color=SETTING_COLORS[setting],
            error_kw={"elinewidth": 1.5, "ecolor": "black"},
        )
        for bar, m, s in zip(bars, means, stds):
            if m > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.012,
                    f"{m:.3f}",
                    ha="center", va="bottom", fontsize=9,
                )

    ax.set(
        xticks=x,
        ylim=(0, 1.18),
        ylabel="Final val-accuracy  (mean ± std over clients)",
        title=f"{algorithm.upper()} — Final Accuracy by Heterogeneity Distribution",
    )
    ax.set_xticklabels([DIST_LABELS[d] for d in dists_avail], fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


# ── Figure 3 — Per-client accuracy box plots ──────────────────────────────────

def plot_per_client_distributions(data: dict, algorithm: str, save_path: str):
    """Box + strip chart of final per-client accuracy, one column per distribution."""
    dists_avail = [d for d in DISTRIBUTIONS if data.get(d)]
    if not dists_avail:
        return

    n_d = len(dists_avail)
    fig, axes = plt.subplots(1, n_d, figsize=(5 * n_d, 5), sharey=True)
    if n_d == 1:
        axes = [axes]

    for ax, dist in zip(axes, dists_avail):
        groups, tick_labels, colors = [], [], []
        for setting in SETTINGS:
            d = data.get(dist, {}).get(setting)
            if d is None:
                continue
            groups.append(d["final_accs"])
            tick_labels.append(SETTING_LABELS[setting])
            colors.append(SETTING_COLORS[setting])

        if not groups:
            ax.set_title(DIST_LABELS[dist])
            continue

        bp = ax.boxplot(
            groups,
            labels=tick_labels,
            patch_artist=True,
            notch=False,
            widths=0.45,
        )
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)
        for element in ("whiskers", "caps", "medians", "fliers"):
            for item in bp[element]:
                item.set_color("black")

        # Individual client dots
        for i, (grp, col) in enumerate(zip(groups, colors)):
            jitter = np.random.default_rng(i).uniform(-0.08, 0.08, len(grp))
            ax.scatter(
                np.full(len(grp), i + 1) + jitter, grp,
                color=col, s=25, alpha=0.75, zorder=3,
            )

        ax.set(title=DIST_LABELS[dist], ylim=(0, 1.05))
        ax.set_ylabel("Final val-accuracy" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{algorithm.upper()} — Per-Client Final Accuracy Distribution",
        fontsize=13,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


# ── Figure 4 — IID vs Non-IID per distribution ───────────────────────────────

def plot_iid_vs_noniid(data: dict, algorithm: str, save_path: str):
    """One column per distribution; IID and non-IID as separate lines with shading."""
    dists_avail = [d for d in DISTRIBUTIONS if data.get(d)]
    if not dists_avail:
        return

    n_d = len(dists_avail)
    fig, axes = plt.subplots(1, n_d, figsize=(6 * n_d, 5), sharey=True)
    if n_d == 1:
        axes = [axes]

    for ax, dist in zip(axes, dists_avail):
        for setting in SETTINGS:
            d = data.get(dist, {}).get(setting)
            if d is None:
                continue
            rounds = np.arange(1, d["n_rounds"] + 1)
            col    = SETTING_COLORS[setting]
            ax.plot(rounds, d["avg"],
                    label=SETTING_LABELS[setting], color=col, lw=2)
            ax.fill_between(rounds, d["min"], d["max"],
                            alpha=0.10, color=col)

        ax.set(
            title=DIST_LABELS[dist],
            xlabel="FL Round",
            xlim=(1, None),
            ylim=(0, 1),
        )
        if ax is axes[0]:
            ax.set_ylabel("Avg client val-accuracy")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"{algorithm.upper()} — IID vs Non-IID per Distribution\n"
        f"(shaded band = per-client min/max range)",
        fontsize=13,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "figures")
    algo    = args.algorithm

    print(f"\nLoading results from:")
    print(f"  {os.path.abspath(os.path.join(args.results_dir, args.dataset, algo))}")
    print(f"  local_epochs per round = {args.local_epochs}")

    data = collect_results(
        args.results_dir, args.dataset, algo, args.local_epochs)

    print_summary(data, algo)

    base = f"{args.dataset}_{algo}"
    plot_training_curves(
        data, algo,
        os.path.join(out_dir, f"{base}_curves.png"))
    plot_final_accuracy_bars(
        data, algo,
        os.path.join(out_dir, f"{base}_final_bars.png"))
    plot_per_client_distributions(
        data, algo,
        os.path.join(out_dir, f"{base}_client_dist.png"))
    plot_iid_vs_noniid(
        data, algo,
        os.path.join(out_dir, f"{base}_iid_vs_noniid.png"))

    print(f"All figures saved to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
