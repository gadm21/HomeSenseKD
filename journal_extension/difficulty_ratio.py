"""Task-difficulty vs data-quantity analysis.

For each benchmark: silhouette score + Fisher discriminant ratio on the
(raw, pipeline-scaled) training features, samples per client, and FL
effectiveness = non-IID FedAvg acc relative to Local and Central.
"""
import numpy as np, pandas as pd, os, json
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import utils  # loaders

def load(folder):
    cfp = os.path.join(folder, 'central.csv')
    if os.path.exists(cfp):
        df = pd.read_csv(cfp); ep = df['epoch'].values.astype(int)
        gs = next((i for i in range(1, len(ep)) if ep[i] == 0), len(ep))
        n = len(df) // gs
        return df['val_accuracy'].values[:n*gs].reshape(n, gs)[:, -1][np.newaxis, :]
    cid, rows = 0, []
    while os.path.exists(os.path.join(folder, f'train_{cid}.csv')):
        df = pd.read_csv(os.path.join(folder, f'train_{cid}.csv'))
        ep = df['epoch'].values.astype(int)
        gs = next((i for i in range(1, len(ep)) if ep[i] == 0), len(ep))
        n = len(df) // gs
        rows.append(df['val_accuracy'].values[:n*gs].reshape(n, gs)[:, -1]); cid += 1
    if not rows:
        return None
    m = min(len(r) for r in rows)
    return np.array([r[:m] for r in rows])

def final_acc(folder, R=50):
    a = load(folder)
    if a is None:
        return None
    C, T = a.shape
    ns = T // R
    # mean over seeds of (mean over clients of last-5-rounds mean)
    per_seed = [a[:, s*R + R - 5:(s+1)*R].mean() for s in range(ns)]
    return float(np.mean(per_seed) * 100), float(np.std(per_seed) * 100), ns

def fisher_ratio(X, y):
    """trace(S_b)/trace(S_w): higher = easier."""
    classes = np.unique(y)
    mu = X.mean(axis=0)
    sb = sw = 0.0
    for c in classes:
        Xc = X[y == c]
        muc = Xc.mean(axis=0)
        sb += len(Xc) * ((muc - mu) ** 2).sum()
        sw += ((Xc - muc) ** 2).sum()
    return sb / max(sw, 1e-12)

DATASETS = {
    'home_occupancy': dict(loader=lambda: utils.load_homeoccupancy(1500, 8),
                           root='results/phase1/home_occupancy',
                           fb={'fedmd': 'oldresults/phase1/home_occupancy/fedmd/uniform/noniid'},
                           n_classes=3, spc=30),
    'mnist':          dict(loader=utils.load_mnist,
                           root='oldresults/phase1/mnist',
                           fb={}, n_classes=10, spc=30),
    'cifar10':        dict(loader=utils.load_cifar10,
                           root='oldresults/phase1/cifar10',
                           fb={}, n_classes=10, spc=30),
}

print(f"{'dataset':<15}{'silhouette':>11}{'fisher':>8}{'n/client':>9}{'central':>9}"
      f"{'local':>7}{'fedavg':>7}{'fedmd':>7}{'mks':>7}{'FA/Loc':>7}{'FA/Cen':>7}")
for name, d in DATASETS.items():
    x_tr, y_tr, x_te, y_te = d['loader']()
    y_tr = np.asarray(y_tr).argmax(axis=1) if y_tr.ndim > 1 else np.asarray(y_tr)
    X = x_tr.reshape(len(x_tr), -1).astype(np.float32)
    X = MinMaxScaler().fit_transform(X)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), min(3000, len(X)), replace=False)
    sil = silhouette_score(X[idx], y_tr[idx])
    fr = fisher_ratio(X[idx], y_tr[idx])
    n_client = d['n_classes'] * d['spc']
    row = [name, sil, fr, n_client]
    accs = {}
    for algo in ['central', 'local', 'fedavg', 'fedmd', 'mks']:
        folder = d['fb'].get(algo, f"{d['root']}/{algo}/uniform/noniid")
        if algo == 'central':
            folder = f"{d['root']}/central/uniform/iid"
            if not os.path.exists(folder):
                folder = f"{d['root']}/central/uniform/noniid"
        r = final_acc(folder)
        accs[algo] = r[0] if r else np.nan
    fa_loc = accs['fedavg'] / accs['local'] if accs['local'] else np.nan
    fa_cen = accs['fedavg'] / accs['central'] if accs['central'] else np.nan
    print(f"{name:<15}{sil:>11.3f}{fr:>8.3f}{n_client:>9}"
          f"{accs['central']:>9.1f}{accs['local']:>7.1f}{accs['fedavg']:>7.1f}"
          f"{accs['fedmd']:>7.1f}{accs['mks']:>7.1f}{fa_loc:>7.2f}{fa_cen:>7.2f}")
