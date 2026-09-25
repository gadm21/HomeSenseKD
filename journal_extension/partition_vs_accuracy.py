"""Replay the exact seeded Dirichlet partitions and correlate each client's
label distribution with its final accuracy (Local / FedAvg / FedMD, non-IID).

Verified: replayed per-client sample counts match the run logs exactly
(e.g. HomeOcc seed-42 non-IID client sizes [13,12,25,17,47,103,...]).

Writes results/partition_stats.json for gen_figs.py and prints the
correlation tables.  Run from repo root:
    .\\venv\\Scripts\\python.exe journal_extension\\partition_vs_accuracy.py
"""
import numpy as np
import pandas as pd
import json, os

SEEDS = [42, 123, 456]
R, GS = 50, 4          # rounds per seed, local epochs per round (CSV rows/round)

# dataset -> (class counts in y_train, n_classes, n_parties, spc, alpha)
# HomeOcc counts measured from load_homeoccupancy(window_size=1500) output.
DS = {
    'home_occupancy': dict(counts=[375, 380, 293], n_cls=3,  spc=30),
    'mnist':          dict(counts=[6000]*10,       n_cls=10, spc=30),
    'cifar10':        dict(counts=[5000]*10,       n_cls=10, spc=30),
}
N_PARTIES, ALPHA = 20, 0.5


def replay_partitions(counts, n_cls, spc, seed):
    """Reproduce make_partitions RNG stream: iid_partition then
    dirichlet_partition.  Returns (n_parties, n_cls) sample counts."""
    np.random.seed(seed)
    y = np.repeat(np.arange(n_cls), counts)
    # iid_partition -> split_dataset: one choice() per class
    for c in range(n_cls):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        np.random.choice(idx, max(spc * N_PARTIES, len(idx)),
                         replace=True)[:spc * N_PARTIES]
    # dirichlet_partition
    cnt = np.zeros((N_PARTIES, n_cls), dtype=int)
    for c in range(n_cls):
        idxs = np.where(y == c)[0]
        if len(idxs) == 0:
            continue
        np.random.shuffle(idxs)
        props = np.random.dirichlet(np.repeat(ALPHA, N_PARTIES))
        splits = (np.cumsum(props[:-1]) * len(idxs)).astype(int)
        for i, chunk in enumerate(np.split(idxs, splits)):
            cnt[i, c] = len(chunk)
    return cnt


def client_finals(csv_dir, n_clients=20):
    """(n_clients, n_seeds) mean of last-5-round val_accuracy per seed."""
    first = pd.read_csv(os.path.join(csv_dir, "train_0.csv"))
    n_seeds = len(first) // (GS * R)
    out = np.full((n_clients, n_seeds), np.nan)
    for cid in range(n_clients):
        df = pd.read_csv(os.path.join(csv_dir, f"train_{cid}.csv"))
        v = df["val_accuracy"].values
        v = v[: len(v) // GS * GS].reshape(-1, GS)[:, -1]
        for s in range(min(n_seeds, len(v) // R)):
            out[cid, s] = v[s * R + R - 5:(s + 1) * R].mean()
    return out * 100


stats = {}
for ds, p in DS.items():
    stats[ds] = []
    for seed in SEEDS:
        cnt = replay_partitions(p['counts'], p['n_cls'], p['spc'], seed)
        n = cnt.sum(1)
        dom = cnt.max(1) / np.maximum(n, 1)
        # dominant-class mass: fraction of ALL client samples sitting in
        # clients whose plurality class is c; the max over c predicts
        # whether the first global average falls into one class's basin.
        argmax = cnt.argmax(1)
        mass = [n[argmax == c].sum() for c in range(p['n_cls'])]
        stats[ds].append({'seed': seed,
                          'n': n.tolist(),
                          'dom_share': dom.tolist(),
                          'dom_mass': (max(mass) / max(n.sum(), 1)),
                          'counts': cnt.tolist()})

os.makedirs('results', exist_ok=True)
json.dump(stats, open('results/partition_stats.json', 'w'), indent=1)
print('wrote results/partition_stats.json')

# ---- HomeOccupancy: partition vs accuracy ----
loc = client_finals("results/phase1/home_occupancy/local/uniform/noniid")
fa  = client_finals("results/phase1/home_occupancy/fedavg/uniform/noniid")
fmd = client_finals("oldresults/phase1/home_occupancy/fedmd/uniform/noniid")

print("\n=== HomeOccupancy non-IID: partition vs final accuracy ===")
for s, seed in enumerate(SEEDS):
    st = stats['home_occupancy'][s]
    print(f"\n--- seed {s+1} (np seed={seed}) ---")
    print(f"{'cid':>3} {'n':>4} {'dom%':>5}  {'counts':>14}  {'Local':>6} {'FedAvg':>6} {'FedMD':>6}")
    doms, locs, fas, fms = [], [], [], []
    for cid in range(20):
        cnt = st['counts'][cid]
        print(f"{cid:>3} {st['n'][cid]:>4} {st['dom_share'][cid]*100:>5.0f}  "
              f"{str(cnt):>14}  {loc[cid,s] if s < loc.shape[1] else float('nan'):>6.1f} "
              f"{fa[cid,s]:>6.1f} {fmd[cid,s] if s < fmd.shape[1] else float('nan'):>6.1f}")
        doms.append(st['dom_share'][cid])
        locs.append(loc[cid, s] if s < loc.shape[1] else np.nan)
        fas.append(fa[cid, s]); fms.append(fmd[cid, s] if s < fmd.shape[1] else np.nan)
    for nm, arr in [('Local', locs), ('FedAvg', fas), ('FedMD', fms)]:
        a = np.array(arr); m = ~np.isnan(a)
        if m.sum() > 2:
            print(f"corr(dom-share, {nm:6s}) = {np.corrcoef(np.array(doms)[m], a[m])[0,1]:+.3f}")
    print(f"dominant-class mass fraction = {st['dom_mass']:.3f}  "
          f"(FedAvg mean = {np.nanmean(fa[:, s]):.1f})")

# ---- Phase-2 FedMD per-tier means (tiers assigned in client order) ----
TIERFRAC = {'all_small': [1.0, 0.0, 0.0], 'uniform': [0.35, 0.35, 0.30],
            'skewed': [0.60, 0.30, 0.10]}
def tier_map(dist, n=20):
    counts = [round(f * n) for f in TIERFRAC[dist]]
    counts[0] += n - sum(counts)
    out = []
    for t, c in zip(['small', 'medium', 'large'], counts):
        out += [t] * max(0, c)
    return out[:n]

print("\n=== Phase-2 FedMD per-tier means (final, last-5-rounds) ===")
for h in ("all_small", "uniform", "skewed"):
    tiers = tier_map(h)
    for sett in ("iid", "noniid"):
        d = f"oldresults/phase2/home_occupancy/fedmd/{h}/{sett}"
        if not os.path.isdir(d):
            continue
        a = client_finals(d)
        for t in ("small", "medium", "large"):
            idx = [i for i, x in enumerate(tiers) if x == t]
            if idx:
                print(f"{h:>9} {sett:>6} {t:>6}: n={len(idx):2d}  "
                      f"mean={np.nanmean(a[idx]):5.1f}  "
                      f"range=[{np.nanmin(a[idx]):.1f},{np.nanmax(a[idx]):.1f}]")
