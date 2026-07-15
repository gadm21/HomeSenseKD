#!/usr/bin/env python3
"""
results.py — Publication-quality figures for HomeSenseKD experiments.

Generates five figures saved to results/figures/:

  Phase 1  (results/phase1/)
    phase1_cifar10.png          — all algorithms, CIFAR-10
    phase1_home_har.png         — all algorithms, HomeHAR
    phase1_home_occupancy.png   — all algorithms, HomeOccupancy
    phase1_mnist.png            — all algorithms, MNIST

  Phase 2  (results/phase2/)
    phase2_fedmd_hetero.png     — FedMD: all_small / uniform / skewed ablation

Each figure has two subplots: IID (left) and Non-IID (right).
Lines show mean val-accuracy across clients; shaded band = ±1 std.

Usage:
    python results.py                          # both phases, auto-detect paths
    python results.py --results-dir results    # explicit root
    python results.py --phase 1                # Phase 1 only
    python results.py --phase 2                # Phase 2 only
    python results.py --out-dir my/figs        # custom output directory
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── Algorithm display config ──────────────────────────────────────────────────

ALGO_ORDER = ["central", "fedmd", "fedakd", "mks", "fedavg", "fedprox", "local"]

ALGO_LABELS = {
    "central": "Centralised",
    "fedmd":   "FedMD",
    "fedakd":  "FedAKD",
    "mks":     "MKS",
    "fedavg":  "FedAvg",
    "fedprox": "FedProx",
    "local":   "Local",
}

# Wong (2011) colour-blind-safe palette
ALGO_COLORS = {
    "central": "#56B4E9",   # sky blue
    "fedmd":   "#0072B2",   # blue
    "fedakd":  "#E69F00",   # orange
    "mks":     "#009E73",   # green
    "fedavg":  "#D55E00",   # vermilion
    "fedprox": "#CC79A7",   # reddish purple
    "local":   "#999999",   # grey
}

ALGO_LINESTYLES = {
    "central": (0, (4, 1.5)),   # densely dashed  — upper bound
    "fedmd":   "-",
    "fedakd":  "-",
    "mks":     "-",
    "fedavg":  "--",            # weight-sharing
    "fedprox": "--",
    "local":   ":",             # non-federated baseline
}

ALGO_LINEWIDTHS = {
    "central": 2.0,
    "fedmd":   1.9,
    "fedakd":  1.9,
    "mks":     1.9,
    "fedavg":  1.7,
    "fedprox": 1.7,
    "local":   1.5,
}

# ── Distribution display config ───────────────────────────────────────────────

DIST_ORDER = ["all_small", "uniform", "skewed"]

DIST_LABELS = {
    "all_small": "All-Small",
    "uniform":   "Uniform",
    "skewed":    "Skewed",
}

DIST_COLORS = {
    "all_small": "#D55E00",
    "uniform":   "#0072B2",
    "skewed":    "#009E73",
}

DIST_LINESTYLES = {
    "all_small": "-",
    "uniform":   "--",
    "skewed":    ":",
}

# ── Shared constants ──────────────────────────────────────────────────────────

SETTINGS = ["iid", "noniid"]
SETTING_LABELS = {"iid": "IID", "noniid": "Non-IID"}

DATASET_LABELS = {
    "cifar10":        "CIFAR-10",
    "home_har":       "HomeHAR",
    "home_occupancy": "HomeOccupancy",
    "mnist":          "MNIST",
}


# ── Publication style ─────────────────────────────────────────────────────────

def _pub_style():
    """Apply publication-ready rcParams."""
    plt.rcParams.update({
        "font.family":         "serif",
        "font.size":           10,
        "axes.labelsize":      11,
        "axes.titlesize":      11,
        "axes.titleweight":    "bold",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "grid.alpha":          0.25,
        "grid.linestyle":      "--",
        "grid.linewidth":      0.5,
        "lines.linewidth":     1.7,
        "legend.fontsize":     8.5,
        "legend.framealpha":   0.90,
        "legend.edgecolor":    "0.75",
        "legend.handlelength": 2.2,
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "figure.dpi":          150,
        "savefig.dpi":         300,
        "savefig.facecolor":   "white",
    })


# ── Data loading ──────────────────────────────────────────────────────────────

def _seed_stats(arr: np.ndarray, n_rounds: int) -> dict:
    """Compute per-seed statistics for publication training curves.

    Pipeline
    --------
    1. Split (n_clients, total_rounds) → (n_clients, n_seeds, n_rounds)
    2. Mean over clients per seed  → (n_seeds, n_rounds)
    3. avg / std over seeds        → line = mean across seeds,
                                     shadow = seed-to-seed fluctuation

    If only one seed is present (total_rounds <= n_rounds) the shadow
    reflects client variance instead.
    """
    n_clients, total_rounds = arr.shape

    if n_rounds <= 0 or total_rounds <= n_rounds:
        # Single seed: avg/std across clients
        s = _summarise(arr)
        s["n_seeds"] = 1
        return s

    n_seeds = total_rounds // n_rounds
    trimmed  = arr[:, : n_seeds * n_rounds]              # drop partial tail
    # (n_clients, n_seeds, n_rounds) → mean over clients → (n_seeds, n_rounds)
    per_seed = trimmed.reshape(n_clients, n_seeds, n_rounds).mean(axis=0)

    s = _summarise(per_seed)   # avg/std now across seeds
    s["n_seeds"] = n_seeds
    return s


def _compute_ylim(stat_dicts: list, padding: float = 0.04) -> tuple:
    """Tight y-limits from a list of stats dicts, rounded to nearest 5 pp."""
    vals = []
    for d in stat_dicts:
        if d is not None:
            vals.extend((d["avg"] - d["std"]).tolist())
            vals.extend((d["avg"] + d["std"]).tolist())
    if not vals:
        return (0.0, 1.0)
    lo = max(0.0, min(vals) - padding)
    hi = min(1.0, max(vals) + padding)
    lo = float(np.floor(lo * 20) / 20)   # round down to nearest 0.05
    hi = float(np.ceil (hi * 20) / 20)   # round up   to nearest 0.05
    if hi - lo < 0.10:                   # enforce minimum span
        mid = (lo + hi) / 2
        lo, hi = max(0.0, mid - 0.05), min(1.0, mid + 0.05)
    return (lo, hi)


def _detect_group_size(df: pd.DataFrame) -> int:
    """Auto-detect rows per FL round from epoch-column reset pattern."""
    if "epoch" not in df.columns or len(df) < 2:
        return 1
    ep = df["epoch"].values.astype(int)
    for i in range(1, len(ep)):
        if ep[i] == 0:
            return i
    return len(ep)


def _per_round_val(df: pd.DataFrame, group_size: int) -> np.ndarray:
    """Last val_accuracy of each round → 1-D array shape (n_rounds,)."""
    n = len(df) // group_size
    if n == 0:
        return np.array([])
    return df["val_accuracy"].values[: n * group_size].reshape(n, group_size)[:, -1]


def _load_folder(folder: str) -> np.ndarray | None:
    """Load all CSVs in *folder* and return shape (n_clients, n_rounds) or None.

    Handles both per-client train_N.csv layout and the single central.csv.
    """
    if not os.path.isdir(folder):
        return None

    # ── Central: single model ─────────────────────────────────────────────────
    central_fp = os.path.join(folder, "central.csv")
    if os.path.exists(central_fp):
        try:
            df = pd.read_csv(central_fp)
            gs  = _detect_group_size(df)
            arr = _per_round_val(df, gs)
            return arr[np.newaxis, :] if len(arr) else None   # (1, R)
        except Exception:
            return None

    # ── Per-client: train_0.csv, train_1.csv, ... ────────────────────────────
    rows, cid = [], 0
    while True:
        fp = os.path.join(folder, f"train_{cid}.csv")
        if not os.path.exists(fp):
            break
        try:
            df  = pd.read_csv(fp)
            gs  = _detect_group_size(df)
            arr = _per_round_val(df, gs)
            if len(arr):
                rows.append(arr)
        except Exception:
            pass
        cid += 1

    if not rows:
        return None
    min_r = min(len(a) for a in rows)
    return np.array([a[:min_r] for a in rows])   # (C, R)


def _summarise(arr: np.ndarray) -> dict:
    """(n_clients, n_rounds) → statistics dict."""
    return {
        "avg":       arr.mean(axis=0),
        "std":       arr.std(axis=0),
        "min":       arr.min(axis=0),
        "max":       arr.max(axis=0),
        "n_rounds":  arr.shape[1],
        "n_clients": arr.shape[0],
    }


def collect_phase1(results_dir: str, n_rounds: int = 50) -> dict:
    """Return {dataset: {algo: {setting: stats}}} for Phase 1 (uniform heterogeneity)."""
    p1 = os.path.join(results_dir, "phase1")
    data: dict = {}
    if not os.path.isdir(p1):
        return data
    for dataset in sorted(os.listdir(p1)):
        if not os.path.isdir(os.path.join(p1, dataset)):
            continue
        data[dataset] = {}
        for algo in ALGO_ORDER:
            data[dataset][algo] = {}
            for setting in SETTINGS:
                folder = os.path.join(p1, dataset, algo, "uniform", setting)
                arr = _load_folder(folder)
                if arr is not None:
                    data[dataset][algo][setting] = _seed_stats(arr, n_rounds)
    return data


def collect_phase2(results_dir: str, n_rounds: int = 50) -> dict:
    """Return {dist: {setting: stats}} for FedMD heterogeneity ablation."""
    data: dict = {}
    for dist in DIST_ORDER:
        data[dist] = {}
        for setting in SETTINGS:
            folder = os.path.join(
                results_dir, "phase2", "home_occupancy", "fedmd", dist, setting)
            arr = _load_folder(folder)
            if arr is not None:
                data[dist][setting] = _seed_stats(arr, n_rounds)
    return data


# ── Console summary ───────────────────────────────────────────────────────────

def _print_summary_p1(data: dict):
    sep = "=" * 70
    print(f"\n{sep}\n  Phase 1 — data loaded\n{sep}")
    for ds, algos in data.items():
        label = DATASET_LABELS.get(ds, ds)
        for algo, settings in algos.items():
            for sett, d in settings.items():
                print(f"  {label:16s}  {ALGO_LABELS[algo]:12s}  "
                      f"{SETTING_LABELS[sett]:7s}  "
                      f"seeds={d.get('n_seeds',1):1d}  "
                      f"rounds={d['n_rounds']:3d}  "
                      f"final={d['avg'][-1]:.4f}")
    print(sep)


def _print_summary_p2(data: dict):
    sep = "=" * 70
    print(f"\n{sep}\n  Phase 2 — FedMD heterogeneity ablation\n{sep}")
    for dist in DIST_ORDER:
        for sett in SETTINGS:
            d = data.get(dist, {}).get(sett)
            if d:
                print(f"  {DIST_LABELS[dist]:12s}  {SETTING_LABELS[sett]:7s}  "
                      f"seeds={d.get('n_seeds',1):1d}  "
                      f"rounds={d['n_rounds']:3d}  "
                      f"final={d['avg'][-1]:.4f}")
    print(sep)


# ── Shared axis helper ────────────────────────────────────────────────────────

def _style_ax(ax, setting: str, ylabel: bool = False,
              ylim: tuple = (0.0, 1.0)):
    ax.set_title(SETTING_LABELS[setting], pad=5)
    ax.set_xlabel("FL Round", labelpad=3)
    if ylabel:
        ax.set_ylabel("Validation Accuracy", labelpad=4)
    ax.set_xlim(1, None)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))


# ── Figure builder: Phase 1 ───────────────────────────────────────────────────

def plot_phase1_dataset(algo_data: dict, dataset_name: str, save_path: str):
    """One publication figure per Phase 1 dataset.

    Layout: 2 subplots side-by-side (IID | Non-IID).
    Lines: one per algorithm; shaded band = ±1 std.
    Legend: single shared legend below the axes.
    """
    # Collect all stats for ylim computation
    all_stats = [
        algo_data.get(algo, {}).get(sett)
        for algo in ALGO_ORDER
        for sett in SETTINGS
    ]
    ylim = _compute_ylim(all_stats)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

    legend_handles: list = []

    for col_idx, (ax, setting) in enumerate(zip(axes, SETTINGS)):
        for algo in ALGO_ORDER:
            d = algo_data.get(algo, {}).get(setting)
            if d is None:
                continue
            rounds = np.arange(1, d["n_rounds"] + 1)
            c  = ALGO_COLORS[algo]
            ls = ALGO_LINESTYLES[algo]
            lw = ALGO_LINEWIDTHS[algo]
            ax.plot(rounds, d["avg"], color=c, ls=ls, lw=lw,
                    label=ALGO_LABELS[algo])
            ax.fill_between(rounds,
                            d["avg"] - d["std"],
                            d["avg"] + d["std"],
                            alpha=0.08, color=c, linewidth=0)
        _style_ax(ax, setting, ylabel=(col_idx == 0), ylim=ylim)

    # Build legend from first subplot that has handles
    for ax in axes:
        h, _ = ax.get_legend_handles_labels()
        if h:
            legend_handles = h
            break

    legend_labels = [ALGO_LABELS[a] for a in ALGO_ORDER
                     if any(algo_data.get(a, {}).get(s) for s in SETTINGS)]

    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
        fontsize=8,
        handlelength=2.4,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    ds_label = DATASET_LABELS.get(dataset_name, dataset_name)
    fig.suptitle(ds_label, fontsize=12, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.30, wspace=0.10)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


# ── Figure builder: Phase 2 ───────────────────────────────────────────────────

def plot_phase2_fedmd(data: dict, save_path: str):
    """FedMD heterogeneity ablation figure.

    Layout: 2 subplots (IID | Non-IID).
    Lines: one per distribution (all_small, uniform, skewed).
    Shaded band = ±1 std.
    """
    # Collect all stats for ylim computation
    all_stats = [
        data.get(dist, {}).get(sett)
        for dist in DIST_ORDER
        for sett in SETTINGS
    ]
    ylim = _compute_ylim(all_stats)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

    for col_idx, (ax, setting) in enumerate(zip(axes, SETTINGS)):
        for dist in DIST_ORDER:
            d = data.get(dist, {}).get(setting)
            if d is None:
                continue
            rounds = np.arange(1, d["n_rounds"] + 1)
            c  = DIST_COLORS[dist]
            ls = DIST_LINESTYLES[dist]
            ax.plot(rounds, d["avg"], color=c, ls=ls, lw=1.9,
                    label=DIST_LABELS[dist])
            ax.fill_between(rounds,
                            d["avg"] - d["std"],
                            d["avg"] + d["std"],
                            alpha=0.10, color=c, linewidth=0)
        _style_ax(ax, setting, ylabel=(col_idx == 0), ylim=ylim)

    # Shared legend below
    handles = [
        Line2D([0], [0], color=DIST_COLORS[d], ls=DIST_LINESTYLES[d], lw=1.9,
               label=DIST_LABELS[d])
        for d in DIST_ORDER
        if any(data.get(d, {}).get(s) for s in SETTINGS)
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
        fontsize=9,
        handlelength=2.4,
        columnspacing=1.4,
    )

    fig.suptitle(
        "FedMD — HomeOccupancy Heterogeneity Ablation",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.subplots_adjust(bottom=0.24, wspace=0.10)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Publication-quality FL result figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--results-dir", default="results",
                   help="Root results directory containing phase1/ and phase2/ "
                        "(default: results)")
    p.add_argument("--phase", choices=["1", "2", "all"], default="all",
                   help="Which phase to plot: 1, 2, or all (default: all)")
    p.add_argument("--n-rounds", type=int, default=50,
                   help="FL rounds per seed (used to split multi-seed CSVs; "
                        "default 50). Set to 0 to skip splitting.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for figures "
                        "(default: <results_dir>/figures)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    _pub_style()

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if args.phase in ("1", "all"):
        print("\nLoading Phase 1 data ...")
        p1 = collect_phase1(args.results_dir, n_rounds=args.n_rounds)
        if not p1:
            print("  [warn] No Phase 1 data found — check --results-dir")
        else:
            _print_summary_p1(p1)
            print("\nRendering Phase 1 figures ...")
            for dataset, algo_data in sorted(p1.items()):
                if not any(algo_data[a] for a in algo_data):
                    continue
                save_path = os.path.join(out_dir, f"phase1_{dataset}.png")
                plot_phase1_dataset(algo_data, dataset, save_path)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if args.phase in ("2", "all"):
        print("\nLoading Phase 2 data ...")
        p2 = collect_phase2(args.results_dir, n_rounds=args.n_rounds)
        if not any(p2.get(d) for d in DIST_ORDER):
            print("  [warn] No Phase 2 data found — check --results-dir")
        else:
            _print_summary_p2(p2)
            print("\nRendering Phase 2 figure ...")
            plot_phase2_fedmd(p2, os.path.join(out_dir, "phase2_fedmd_hetero.png"))

    print(f"\nAll figures saved to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
