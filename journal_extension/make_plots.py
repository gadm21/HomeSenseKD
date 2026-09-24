"""Generate supporting plots for the difficulty/data-ratio analysis.
Run from the repository root:  python journal_extension/make_plots.py
Outputs PNGs into journal_extension/plots/.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUT, exist_ok=True)

# ---- measured values (see difficulty_ratio.py / README.md) ----
DS = ['HomeOcc', 'HomeHAR', 'MNIST', 'CIFAR-10']
sil       = np.array([0.038, -0.003, 0.044, -0.063])   # silhouette (3k subsample)
fisher    = np.array([0.084, 1.261, 0.272, 0.080])     # trace(Sb)/trace(Sw)
n_client  = np.array([90, 210, 300, 300])              # samples per client
maxshare  = np.array([0.79, 0.46, 0.44, 0.34])         # Dirichlet a=0.5 dominant share
fa_loc    = np.array([0.79, 1.27, 1.14, 1.34])         # FedAvg/Local non-IID
fa_cen    = np.array([0.52, 0.68, 0.99, 0.70])         # FedAvg/Central
colors    = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

diff_per_sample = (1 - sil) / n_client * 1e3           # x1000 for readability

# ---- plot 1: difficulty-per-sample vs FL effectiveness ----
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for i, d in enumerate(DS):
    ax.scatter(diff_per_sample[i], fa_cen[i], c=colors[i], s=70, zorder=3)
    ax.annotate(d, (diff_per_sample[i], fa_cen[i]),
                textcoords='offset points', xytext=(7, 5), fontsize=9)
ax.axhline(1.0, color='grey', ls='--', lw=0.8)
ax.set_xlabel('difficulty per sample  (1 - silhouette) / n$_{client}$  [$\\times 10^{-3}$]')
ax.set_ylabel('FL effectiveness  FedAvg / Central (non-IID)')
ax.set_title('Task difficulty vs. data quantity predicts FL effectiveness')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'difficulty_vs_effectiveness.png'), dpi=200)
plt.close(fig)

# ---- plot 2: Dirichlet dominant-class share ----
fig, ax = plt.subplots(figsize=(5.2, 3.2))
bars = ax.bar(DS, maxshare, color=colors)
ax.axhline(0.6, color='red', ls='--', lw=0.9)
ax.text(3.45, 0.61, 'single-class-optimum\nthreshold (~60%)', fontsize=7.5,
        color='red', ha='right')
for b, v in zip(bars, maxshare):
    ax.text(b.get_x() + b.get_width()/2, v + 0.015, f'{v:.2f}', ha='center', fontsize=9)
ax.set_ylabel('mean dominant-class share per client')
ax.set_ylim(0, 0.95)
ax.set_title('Dirichlet $\\alpha$=0.5 skew severity — only HomeOcc breeds\n'
             'single-class local optima (3 classes, 90 samples/client)')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'dirichlet_dominant_share.png'), dpi=200)
plt.close(fig)

# ---- plot 3: separability vs data quantity scatter ----
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for i, d in enumerate(DS):
    ax.scatter(n_client[i], sil[i], c=colors[i], s=70, zorder=3)
    ax.annotate(d, (n_client[i], sil[i]),
                textcoords='offset points', xytext=(7, 5), fontsize=9)
ax.axhline(0, color='grey', ls='--', lw=0.8)
ax.set_xlabel('samples per client')
ax.set_ylabel('silhouette score (input space)')
ax.set_title('Benchmarks in the data-quantity / task-difficulty plane')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'data_vs_difficulty_plane.png'), dpi=200)
plt.close(fig)

# ---- plot 4: FedAvg/Local ratio ----
fig, ax = plt.subplots(figsize=(5.2, 3.2))
bars = ax.bar(DS, fa_loc, color=colors)
ax.axhline(1.0, color='red', ls='--', lw=0.9)
ax.text(3.45, 1.01, 'sharing = local', fontsize=8, color='red', ha='right')
for b, v in zip(bars, fa_loc):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
ax.set_ylabel('FedAvg / Local (non-IID)')
ax.set_ylim(0, 1.55)
ax.set_title('Weight sharing beats local training everywhere except the\n'
             'low-data, high-skew HomeOccupancy benchmark')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fedavg_vs_local.png'), dpi=200)
plt.close(fig)

print('wrote', os.listdir(OUT))
