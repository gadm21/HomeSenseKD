import pandas as pd, numpy as np, os, json

def load(folder):
    cfp = os.path.join(folder, 'central.csv')
    rows = []
    if os.path.exists(cfp):
        df = pd.read_csv(cfp); ep = df['epoch'].values.astype(int)
        gs = next((i for i in range(1, len(ep)) if ep[i] == 0), len(ep))
        n = len(df) // gs
        return df['val_accuracy'].values[:n*gs].reshape(n, gs)[:, -1][np.newaxis, :]
    cid = 0
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

def seeds(folder, R):
    a = load(folder)
    if a is None:
        return None
    C, T = a.shape
    ns = T // R
    if ns == 0:
        return None
    return a[:, :ns*R].reshape(C, ns, R).mean(axis=0) * 100

def coords(v):
    out = []
    for i in range(0, len(v), 10):
        out.append('  ' + ' '.join(f'({j+1},{v[j]:.2f})' for j in range(i, min(i+10, len(v)))))
    return '\n'.join(out)

STYLE = {'central': ('black,thick,dashed', 'Central'),
         'fedmd': ('clBlue', 'FedMD'), 'fedakd': ('clTeal', 'FedAKD'),
         'mks': ('clRed,thick', 'FedMKS'), 'fedavg': ('clPurple', 'FedAvg'),
         'fedprox': ('clOlive', 'FedProx'), 'local': ('clGray,dotted', 'Local')}
FILL = {'central': 'black', 'fedmd': 'clBlue', 'fedakd': 'clTeal', 'mks': 'clRed',
        'fedavg': 'clPurple', 'fedprox': 'clOlive', 'local': 'clGray',
        'all_small': 'clOrange', 'uniform': 'clBlue', 'skewed': 'clGreen'}

def curve_block(algo, s, uid, legend=False):
    """Return (band_tex, line_tex); bands are emitted before all lines so
    later bands cannot cover earlier curves."""
    st, name = STYLE.get(algo, (FILL.get(algo, 'clBlue'), algo))
    col = FILL.get(algo, 'clBlue')
    mean = s.mean(axis=0)
    band = ''
    if s.shape[0] > 1:
        hi = s.max(axis=0); lo = s.min(axis=0)
        bc = list(zip(range(1, len(lo)+1), lo)) + list(zip(range(len(hi), 0, -1), hi[::-1]))
        bcoords = '\n'.join('  ' + ' '.join(f'({x},{y:.2f})' for x, y in bc[i:i+10])
                            for i in range(0, len(bc), 10))
        band = f'\\addplot[draw=none, fill={col}!15, forget plot] coordinates {{\n{bcoords}}};'
    line = f'\\addplot[{st}] coordinates {{\n{coords(mean)}}};'
    if legend:
        line += f'\n\\addlegendentry{{{name}}}'
    return band, line

ALGOS = ['central', 'fedmd', 'fedakd', 'mks', 'fedavg', 'fedprox', 'local']

# ---------- FIG mainiid ----------
panels = [('HomeOccupancy', 'results/phase1/home_occupancy', 'iid', 25, 100, True),
          ('MNIST', 'oldresults/phase1/mnist', 'iid', 88, 100, False),
          ('CIFAR-10', 'oldresults/phase1/cifar10', 'iid', 25, 75, False)]
tex = ['\\begin{figure*}[t]', '\\centering', '\\begin{tikzpicture}', '\\begin{groupplot}[',
       '  group style={', '    xlabels at=edge bottom,', '    ylabels at=edge left,',
       '    group size=3 by 1,', '    horizontal sep=0.55cm,', '  },', '  p1panel,',
       '  width=0.32\\textwidth,', '  height=3.6cm,', ']']
for i, (name, root, sett, ymin, ymax, leg) in enumerate(panels):
    if leg:
        tex.append(f'\\nextgroupplot[ymin={ymin}, ymax={ymax},')
        tex.append('  legend to name=legendPiid,')
        tex.append('  legend style={legend columns=7, font=\\tiny,')
        tex.append('    /tikz/every even column/.append style={column sep=0.3em}}]')
    else:
        tex.append(f'\\nextgroupplot[ymin={ymin}, ymax={ymax}]')
    bands, lines = [], []
    for a in ALGOS:
        s = seeds(f'{root}/{a}/uniform/{sett}', 50)
        if s is None:
            continue
        b, l = curve_block(a, s, f'P{i}{a}', legend=leg)
        if b:
            bands.append(b)
        lines.append(l)
    tex += bands + lines
tex += ['\\end{groupplot}', '\\end{tikzpicture}', '\\par\\vspace{4pt}', '\\ref{legendPiid}',
        '\\caption{Validation accuracy over 50 rounds under IID partitioning.',
        '  Panels: HomeOccupancy, MNIST, CIFAR-10.  Curves show the mean over',
        '  completed seeds (up to three); shaded bands show the seed range.',
        '  FedAvg/FedProx use the all-small architecture.}',
        '\\label{fig:mainiid}', '\\end{figure*}']
open('fig_mainiid.tex', 'w').write('\n'.join(tex))

# ---------- FIG mainnoniid ----------
# per-panel fallback folders for algos whose rerun in results/ is incomplete
# but which have a complete single-seed run in oldresults/
FALLBACK = {('HomeOccupancy', 'noniid'): {
    'fedmd': 'oldresults/phase1/home_occupancy/fedmd/uniform/noniid'}}
panels = [('HomeOccupancy', 'results/phase1/home_occupancy', 'noniid', 25, 100, False),
          ('MNIST', 'oldresults/phase1/mnist', 'noniid', 74, 100, True),
          ('CIFAR-10', 'oldresults/phase1/cifar10', 'noniid', 22, 82, False)]
tex = ['\\begin{figure*}[t]', '\\centering', '\\begin{tikzpicture}', '\\begin{groupplot}[',
       '  group style={', '    xlabels at=edge bottom,', '    ylabels at=edge left,',
       '    group size=3 by 1,', '    horizontal sep=0.55cm,', '  },', '  p1panel,',
       '  width=0.32\\textwidth,', '  height=3.6cm,', ']']
for i, (name, root, sett, ymin, ymax, leg) in enumerate(panels):
    if leg:
        tex.append(f'\\nextgroupplot[ymin={ymin}, ymax={ymax},')
        tex.append('  legend to name=legendPnoniid,')
        tex.append('  legend style={legend columns=7, font=\\tiny,')
        tex.append('    /tikz/every even column/.append style={column sep=0.3em}}]')
    else:
        tex.append(f'\\nextgroupplot[ymin={ymin}, ymax={ymax}]')
    bands, lines = [], []
    for a in ALGOS:
        s = seeds(f'{root}/{a}/uniform/{sett}', 50)
        if s is None and (name, sett) in FALLBACK and a in FALLBACK[(name, sett)]:
            s = seeds(FALLBACK[(name, sett)][a], 50)
        if s is None:
            continue
        b, l = curve_block(a, s, f'Q{i}{a}', legend=leg)
        if b:
            bands.append(b)
        lines.append(l)
    tex += bands + lines
tex += ['\\end{groupplot}', '\\end{tikzpicture}', '\\par\\vspace{4pt}', '\\ref{legendPnoniid}',
        '\\caption{Validation accuracy over 50 rounds under non-IID partitioning',
        '  (Dirichlet $\\alpha{=}0.5$).  Panels: HomeOccupancy, MNIST, CIFAR-10.',
        '  Curves show the mean over completed seeds; shaded bands show the seed',
        '  range.  FedAvg/FedProx use the all-small architecture.}',
        '\\label{fig:mainnoniid}', '\\end{figure*}']
open('fig_mainnoniid.tex', 'w').write('\n'.join(tex))

# ---------- FIG tierablation ----------
TIERSTYLE = {'all_small': ('clOrange', 'All-Small'), 'uniform': ('clBlue', 'Uniform'),
             'skewed': ('clGreen', 'Skewed')}
TIERCOL = {'small': 'clOrange', 'medium': 'clTeal', 'large': 'clPurple'}
TIERFRAC = {'all_small': [1.0, 0.0, 0.0], 'uniform': [0.35, 0.35, 0.30],
            'skewed': [0.60, 0.30, 0.10]}

def tier_map_for(dist, n=20):
    counts = [round(f * n) for f in TIERFRAC[dist]]
    counts[0] += n - sum(counts)
    tiers = []
    for t, c in zip(['small', 'medium', 'large'], counts):
        tiers += [t] * max(0, c)
    return tiers[:n]

def client_seed_blocks(folder, R=50):
    """Return list over seeds of per-client final-acc lists (last-5-round mean, %)."""
    a = load(folder)
    if a is None:
        return []
    C, T = a.shape
    return [(a[:, s*R + R - 5:(s+1)*R].mean(axis=1) * 100).tolist()
            for s in range(T // R)]

YMIN, YMAX = 25, 95
tex = ['\\begin{figure*}[t]', '\\centering', '\\begin{tikzpicture}', '\\begin{groupplot}[',
       '  group style={', '    xlabels at=edge bottom,', '    ylabels at=edge left,',
       '    group size=2 by 2,', '    horizontal sep=1.3cm,',
       '    vertical sep=1.1cm,', '  },', '  p1panel,',
       '  width=0.42\\textwidth,', '  height=3.6cm,', ']']
# top row: training curves, shared y-axis
for i, sett in enumerate(['iid', 'noniid']):
    if i == 0:
        tex.append(f'\\nextgroupplot[ymin={YMIN}, ymax={YMAX},')
        tex.append('  legend to name=legendPtwo,')
        tex.append('  legend style={legend columns=3, font=\\tiny}]')
    else:
        tex.append(f'\\nextgroupplot[ymin={YMIN}, ymax={YMAX}]')
    bands, lines = [], []
    for dist in ['all_small', 'uniform', 'skewed']:
        s = seeds(f'oldresults/phase2/home_occupancy/fedmd/{dist}/{sett}', 50)
        st, name = TIERSTYLE[dist]; col = FILL[dist]
        mean = s.mean(axis=0)
        band = ''
        if s.shape[0] > 1:
            hi = s.max(axis=0); lo = s.min(axis=0)
            bc = list(zip(range(1, len(lo)+1), lo)) + list(zip(range(len(hi), 0, -1), hi[::-1]))
            bcoords = '\n'.join('  ' + ' '.join(f'({x},{y:.2f})' for x, y in bc[i:i+10])
                                for i in range(0, len(bc), 10))
            band = f'\\addplot[draw=none, fill={col}!15, forget plot] coordinates {{\n{bcoords}}};'
        line = f'\\addplot[{st}] coordinates {{\n{coords(mean)}}};'
        if i == 0:
            line += f'\n\\addlegendentry{{{name}}}'
        if band:
            bands.append(band)
        lines.append(line)
    tex += bands + lines
# bottom row: per-client dots coloured by architecture tier
for i, sett in enumerate(['iid', 'noniid']):
    leg = (i == 0)
    hdr = f'\\nextgroupplot[ymin={YMIN}, ymax={YMAX}, xmin=0.5, xmax=3.5,'
    tex.append(hdr)
    tex.append('  xtick={1,2,3},')
    tex.append('  xticklabels={\\texttt{all\\_small},\\texttt{uniform},\\texttt{skewed}},')
    tex.append('  x tick label style={font=\\tiny},')
    if leg:
        tex.append('  legend to name=legendTier,')
        tex.append('  legend style={legend columns=3, font=\\tiny},')
    tex.append('  ylabel={Val.\\ Acc.\\ (\\%)}]')
    rng2 = np.random.RandomState(1)
    for j, dist in enumerate(['all_small', 'uniform', 'skewed']):
        blocks = client_seed_blocks(f'oldresults/phase2/home_occupancy/fedmd/{dist}/{sett}')
        if not blocks:
            continue
        tiers = tier_map_for(dist, len(blocks[0]))
        present = [t for t in ['small', 'medium', 'large'] if t in tiers]
        centres = ([j + 1] if len(present) == 1 else
                   np.linspace(j + 0.74, j + 1.26, len(present)))
        for tier, xc in zip(present, centres):
            pts = []
            for blk in blocks:
                pts += [blk[c] for c in range(len(blk)) if tiers[c] == tier]
            xs = [xc + rng2.uniform(-0.09, 0.09) for _ in pts]
            pc = '\n'.join('  ' + ' '.join(f'({x:.3f},{y:.2f})' for x, y in zip(xs[k:k+10], pts[k:k+10]))
                          for k in range(0, len(pts), 10))
            tex.append(f'\\addplot[only marks, mark=*, mark size=0.9pt, {TIERCOL[tier]}, opacity=0.6, forget plot] coordinates {{\n{pc}}};')
            mt = np.mean(pts)
            tex.append(f'\\addplot[{TIERCOL[tier]}!55!black, thick, forget plot] coordinates {{({xc-0.12},{mt:.2f}) ({xc+0.12},{mt:.2f})}};')
        allpts = [p for blk in blocks for p in blk]
        m = np.mean(allpts)
        tex.append(f'\\addplot[black, thick, forget plot] coordinates {{({j+0.68},{m:.2f}) ({j+1.32},{m:.2f})}};')
    if leg:
        for tier, nm in [('small', 'Small'), ('medium', 'Medium'), ('large', 'Large')]:
            tex.append(f'\\addlegendimage{{only marks, mark=*, {TIERCOL[tier]}}}')
            tex.append(f'\\addlegendentry{{{nm}}}')
tex += ['\\end{groupplot}', '\\end{tikzpicture}', '\\par\\vspace{4pt}',
        '\\ref{legendPtwo}\\hspace{1.5em}\\ref{legendTier}',
        '\\caption{FedMD tier-heterogeneity ablation on HomeOccupancy.',
        '  Top: mean validation-accuracy curves over completed seeds (shaded',
        '  bands: seed range), IID (left) and non-IID (right) on a shared',
        '  y-axis.  Bottom: per-client final accuracy (mean of the last five',
        '  rounds, seeds pooled).  Each distribution column is subdivided',
        '  by architecture tier---small (orange), medium (teal), large',
        '  (purple)---so tier counts are visible (e.g.\\ \\texttt{skewed}:',
        '  12/6/2); short coloured dashes mark tier means, the long black',
        '  dash the column mean.  Tier means differ by only a few points',
        '  within every setting: under soft-label sharing, per-client',
        '  accuracy is governed by the data partition, not by model',
        '  capacity.}',
        '\\label{fig:tierablation}', '\\end{figure*}']
open('fig_tierablation.tex', 'w').write('\n'.join(tex))

# ---------- FIG clientdist ----------
# Per-client final accuracy (mean of last 5 rounds) under non-IID.
# Each dot = one client; all completed seeds pooled.
def client_points(folder, R=50):
    a = load(folder)
    if a is None:
        return None
    C, T = a.shape
    ns = T // R
    pts = []
    for s in range(ns):
        pts.extend((a[:, s*R + R - 5:(s+1)*R].mean(axis=1) * 100).tolist())
    return pts

CD_METHODS = [('local', 'Local'), ('fedavg', 'FedAvg'), ('fedprox', 'FedProx'),
              ('fedmd', 'FedMD'), ('fedakd', 'FedAKD'), ('mks', 'FedMKS')]
CD_PANELS = [
    ('results/phase1/home_occupancy',
     {'fedmd': 'oldresults/phase1/home_occupancy/fedmd/uniform/noniid'},
     33.3, 20, 95),
    ('oldresults/phase1/mnist', {}, None, 55, 100),
    ('oldresults/phase1/cifar10', {}, 10.0, 15, 80),
]
rng = np.random.RandomState(0)
tex = ['\\begin{figure*}[t]', '\\centering', '\\begin{tikzpicture}', '\\begin{groupplot}[',
       '  group style={', '    xlabels at=edge bottom,', '    ylabels at=edge left,',
       '    group size=3 by 1,', '    horizontal sep=1.1cm,', '  },',
       '  width=0.31\\textwidth,', '  height=4.4cm,', ']']
for root, fb, chance, ymin, ymax in CD_PANELS:
    tex.append(f'\\nextgroupplot[ymin={ymin}, ymax={ymax}, xmin=0.4, xmax=6.6,')
    tex.append('  xtick={1,2,3,4,5,6},')
    tex.append('  xticklabels={Local,FedAvg,FedProx,FedMD,FedAKD,FedMKS},')
    tex.append('  x tick label style={font=\\tiny, rotate=25, anchor=east},')
    tex.append('  ylabel={Val.\\ Acc.\\ (\\%)}, grid=major,')
    tex.append('  grid style={gray!18, line width=0.3pt},')
    tex.append('  tick label style={font=\\tiny}, label style={font=\\tiny}]')
    if chance is not None:
        tex.append(f'\\addplot[clGray, dashed, forget plot] coordinates {{(0.4,{chance}) (6.6,{chance})}};')
    for i, (a, name) in enumerate(CD_METHODS):
        folder = fb.get(a, f'{root}/{a}/uniform/noniid')
        pts = client_points(folder)
        if pts is None:
            continue
        col = FILL.get(a, 'clBlue')
        xs = [i + 1 + rng.uniform(-0.28, 0.28) for _ in pts]
        pc = '\n'.join('  ' + ' '.join(f'({x:.3f},{y:.2f})' for x, y in zip(xs[j:j+10], pts[j:j+10]))
                       for j in range(0, len(pts), 10))
        tex.append(f'\\addplot[only marks, mark=*, mark size=0.9pt, {col}, opacity=0.6, forget plot] coordinates {{\n{pc}}};')
        m = np.mean(pts)
        tex.append(f'\\addplot[black, thick, forget plot] coordinates {{({i+0.68},{m:.2f}) ({i+1.32},{m:.2f})}};')
tex += ['\\end{groupplot}', '\\end{tikzpicture}',
        '\\caption{Per-client final validation accuracy (mean of the last five',
        "  rounds, evaluated on each client's model \\emph{after} its local",
        '  update) under non-IID partitioning.  Each dot is one client; all',
        '  completed seeds are pooled (60 dots per three-seed method, 20 per',
        '  single-seed method).  Black dashes mark method means; the grey dashed',
        '  line marks chance level (33.3\\% for HomeOccupancy, 10\\% for',
        '  CIFAR-10).  Left: HomeOccupancy---in the two collapsed',
        '  FedAvg/FedProx seeds \\emph{all} clients sit at chance, while',
        '  KD-based and local clients spread around ${\\approx}60\\%$.',
        '  Middle/right: MNIST and CIFAR-10---no client collapses; weight',
        '  sharing lifts every client above the local mean.}',
        '\\label{fig:clientdist}', '\\end{figure*}']
open('fig_clientdist.tex', 'w').write('\n'.join(tex))

# ---------- FIG clientseeds ----------
# HomeOccupancy non-IID: per-client final acc, Local vs FedAvg per seed.
loc_a = load('results/phase1/home_occupancy/local/uniform/noniid')
fa_a  = load('results/phase1/home_occupancy/fedavg/uniform/noniid')
R = 50
loc_pts = (loc_a[:, R-5:R].mean(axis=1) * 100).tolist()
tex = ['\\begin{figure*}[t]', '\\centering', '\\begin{tikzpicture}', '\\begin{groupplot}[',
       '  group style={', '    xlabels at=edge bottom,', '    ylabels at=edge left,',
       '    group size=3 by 1,', '    horizontal sep=0.9cm,', '  },',
       '  width=0.30\\textwidth,', '  height=4.2cm,', ']']
rng3 = np.random.RandomState(2)
for p in range(1, 4):
    title = f'FedAvg seed {p}'
    tex.append(f'\\nextgroupplot[ymin=20, ymax=95, xmin=0.4, xmax=2.6,')
    tex.append('  xtick={1,2}, xticklabels={Local,FedAvg},')
    tex.append('  x tick label style={font=\\tiny},')
    tex.append(f'  title={{\\footnotesize {title}}}, title style={{yshift=-2pt}},')
    tex.append('  ylabel={Val.\\ Acc.\\ (\\%)}, grid=major,')
    tex.append('  grid style={gray!18, line width=0.3pt},')
    tex.append('  tick label style={font=\\tiny}, label style={font=\\tiny}]')
    tex.append('\\addplot[clGray, dashed, forget plot] coordinates {(0.4,33.3) (2.6,33.3)};')
    xs = [1 + rng3.uniform(-0.25, 0.25) for _ in loc_pts]
    pc = '\n'.join('  ' + ' '.join(f'({x:.3f},{y:.2f})' for x, y in zip(xs[k:k+10], loc_pts[k:k+10]))
                  for k in range(0, len(loc_pts), 10))
    tex.append(f'\\addplot[only marks, mark=*, mark size=0.9pt, clGray, opacity=0.55, forget plot] coordinates {{\n{pc}}};')
    tex.append(f'\\addplot[black, thick, forget plot] coordinates {{(0.7,{np.mean(loc_pts):.2f}) (1.3,{np.mean(loc_pts):.2f})}};')
    fa_pts = (fa_a[:, p*R - 5:p*R].mean(axis=1) * 100).tolist()
    xs = [2 + rng3.uniform(-0.25, 0.25) for _ in fa_pts]
    pc = '\n'.join('  ' + ' '.join(f'({x:.3f},{y:.2f})' for x, y in zip(xs[k:k+10], fa_pts[k:k+10]))
                   for k in range(0, len(fa_pts), 10))
    tex.append(f'\\addplot[only marks, mark=*, mark size=0.9pt, clPurple, opacity=0.65, forget plot] coordinates {{\n{pc}}};')
    tex.append(f'\\addplot[black, thick, forget plot] coordinates {{(1.7,{np.mean(fa_pts):.2f}) (2.3,{np.mean(fa_pts):.2f})}};')
tex += ['\\end{groupplot}', '\\end{tikzpicture}',
        '\\caption{HomeOccupancy non-IID, per-client final accuracy by seed',
        "  (each client's model evaluated after its local update).",
        '  Grey dots (identical in every panel) are the Local-only clients;',
        '  purple dots are FedAvg clients in seeds 1--3.  In seeds 1 and 3',
        '  \\emph{every} FedAvg client sits at chance (33.3\\%)---the shared',
        '  model collapses collectively---while seed 2 retains a healthy',
        '  spread comparable to Local.}',
        '\\label{fig:clientseeds}', '\\end{figure*}']
open('fig_clientseeds.tex', 'w').write('\n'.join(tex))

# ---------- FIG partition ----------
# Data distribution as the deciding factor.  Three panels:
# (a) HomeOcc non-IID: dominant-class share vs per-client final accuracy.
# (b) per-seed mean accuracy: weight sharing collapses in 2/3 seeds while
#     KD/local references stay flat.
# (c) all four datasets in the (samples/client, dominant-share) plane.
pstats = json.load(open('results/partition_stats.json'))
ho = pstats['home_occupancy']

def share_acc_pts(folder):
    """[(share%, acc%)] per client per seed, paired with that seed's partition."""
    a = load(folder)
    if a is None:
        return []
    C, T = a.shape
    out = []
    for s in range(T // R):
        acc = a[:, s*R + R - 5:(s+1)*R].mean(axis=1) * 100
        out += [(ho[s]['dom_share'][c] * 100, acc[c]) for c in range(C)]
    return out

def seed_means(folder):
    """Per-seed mean of per-client last-5-round accuracy (%)."""
    a = load(folder)
    if a is None:
        return []
    C, T = a.shape
    return [float(a[:, s*R + R - 5:(s+1)*R].mean() * 100) for s in range(T // R)]

HO = 'results/phase1/home_occupancy'
MK = {'local': 'mark=*', 'fedavg': 'mark=square*', 'fedprox': 'mark=triangle*',
      'fedmd': 'mark=*', 'fedakd': 'mark=diamond*', 'mks': 'mark=pentagon*'}
tex = ['\\begin{figure*}[t]', '\\centering', '\\begin{tikzpicture}', '\\begin{groupplot}[',
       '  group style={', '    group size=3 by 1,', '    horizontal sep=1.15cm,', '  },',
       '  width=0.31\\textwidth,', '  height=4.4cm,', ']']
# ---- (a) dominant share vs accuracy, HomeOccupancy non-IID ----
tex += ['\\nextgroupplot[xlabel={Dominant-class share (\\%)},',
        '  ylabel={Val.\\ Acc.\\ (\\%)},',
        '  xmin=30, xmax=105, ymin=20, ymax=95, grid=major,',
        '  grid style={gray!18, line width=0.3pt},',
        '  tick label style={font=\\tiny}, label style={font=\\tiny},',
        '  legend to name=legendPartA,',
        '  legend style={legend columns=3, font=\\tiny}]']
for algo, nm, folder in [('local', 'Local', f'{HO}/local/uniform/noniid'),
                         ('fedmd', 'FedMD', 'oldresults/phase1/home_occupancy/fedmd/uniform/noniid'),
                         ('fedavg', 'FedAvg', f'{HO}/fedavg/uniform/noniid')]:
    pts = share_acc_pts(folder)
    pc = '\n'.join('  ' + ' '.join(f'({x:.1f},{y:.2f})' for x, y in pts[k:k+12])
                   for k in range(0, len(pts), 12))
    tex.append(f'\\addplot[only marks, {MK[algo]}, mark size=1.0pt, {FILL[algo]}, opacity=0.65] coordinates {{\n{pc}}};')
    tex.append(f'\\addlegendentry{{{nm}}}')
tex.append('\\addplot[black!50, dashed, forget plot] coordinates {(30,33.3) (105,33.3)};')
# ---- (b) per-seed mean accuracy ----
tex += ['\\nextgroupplot[xlabel={Seed}, ylabel={Mean val.\\ acc.\\ (\\%)},',
        '  xmin=0.6, xmax=3.4, ymin=25, ymax=95, xtick={1,2,3},',
        '  xticklabels={42,123,456}, grid=major,',
        '  grid style={gray!18, line width=0.3pt},',
        '  tick label style={font=\\tiny}, label style={font=\\tiny},',
        '  legend to name=legendPart,',
        '  legend style={legend columns=6, font=\\tiny}]']
for algo, nm, folder in [
        ('fedavg', 'FedAvg', f'{HO}/fedavg/uniform/noniid'),
        ('fedprox', 'FedProx', f'{HO}/fedprox/uniform/noniid'),
        ('fedmd', 'FedMD', 'oldresults/phase1/home_occupancy/fedmd/uniform/noniid'),
        ('fedakd', 'FedAKD', f'{HO}/fedakd/uniform/noniid'),
        ('mks', 'FedMKS', f'{HO}/mks/uniform/noniid'),
        ('local', 'Local', f'{HO}/local/uniform/noniid')]:
    ms = seed_means(folder)
    if not ms:
        continue
    xs = list(range(1, len(ms) + 1))
    if len(ms) > 1:   # per-seed points connected by a line
        pc = ' '.join(f'({x},{y:.2f})' for x, y in zip(xs, ms))
        tex.append(f'\\addplot[{FILL[algo]}, thick, {MK[algo]}, mark size=1.4pt] coordinates {{{pc}}};')
    else:             # single-seed reference: dashed horizontal line
        tex.append(f'\\addplot[{FILL[algo]}, dashed, thick] coordinates {{(0.6,{ms[0]:.2f}) (3.4,{ms[0]:.2f})}};')
    tex.append(f'\\addlegendentry{{{nm}}}')
tex.append('\\addplot[black!50, dotted, forget plot] coordinates {(0.6,33.3) (3.4,33.3)};')
# ---- (c) samples per client vs dominant share, all datasets ----
tex += ['\\nextgroupplot[xmode=log,',
        '  xlabel={Samples per client}, ylabel={Dominant-class share (\\%)},',
        '  xmin=8, xmax=9000, ymin=15, ymax=105, grid=major,',
        '  grid style={gray!18, line width=0.3pt},',
        '  tick label style={font=\\tiny}, label style={font=\\tiny},',
        '  legend to name=legendPartDS,',
        '  legend style={legend columns=3, font=\\tiny}]']
DSCOL = {'home_occupancy': ('clRed', 'mark=*', 'HomeOcc (3 cls)'),
         'mnist': ('clBlue', 'mark=triangle*', 'MNIST (10 cls)'),
         'cifar10': ('clOlive', 'mark=diamond*', 'CIFAR-10 (10 cls)')}
for ds, (col, mk, nm) in DSCOL.items():
    pts = [(n, sh * 100) for sd in pstats[ds]
           for n, sh in zip(sd['n'], sd['dom_share'])]
    pc = '\n'.join('  ' + ' '.join(f'({x},{y:.1f})' for x, y in pts[k:k+12])
                   for k in range(0, len(pts), 12))
    tex.append(f'\\addplot[only marks, {mk}, mark size=1.0pt, {col}, opacity=0.55] coordinates {{\n{pc}}};')
    tex.append(f'\\addlegendentry{{{nm}}}')
    mn = float(np.exp(np.mean(np.log([p[0] for p in pts]))))
    ms = float(np.mean([p[1] for p in pts]))
    tex.append(f'\\addplot[only marks, {mk}, mark size=2.4pt, {col}!45!black, forget plot] coordinates {{({mn:.0f},{ms:.1f})}};')
tex += ['\\end{groupplot}', '\\end{tikzpicture}', '\\par\\vspace{4pt}',
        '\\ref{legendPartA}\\hspace{1.2em}\\ref{legendPart}\\\\[2pt]\\ref{legendPartDS}',
        '\\caption{Data distribution decides whether weight averaging',
        '  survives.  (a)~HomeOccupancy non-IID: per-client final accuracy',
        '  vs.\\ each client\'s dominant-class share.  FedAvg clients in',
        '  the two collapsed seeds sit at chance at \\emph{every} share,',
        '  including nearly balanced ones---collapse is a property of the',
        '  averaged model, not of any client\'s data.  Per-client models',
        '  (Local, FedMD) never collapse; their degradation with share is',
        '  quantified in Fig.~\\ref{fig:imbalance}.',
        '  (b)~Mean accuracy per seed: FedAvg and FedProx collapse in two',
        '  of three seeds while the KD-based methods and Local (single',
        '  seed, dashed references) hold ${\\approx}54$--$63\\%$; FedMKS',
        '  never falls below 53\\%, so its worst case exceeds the',
        '  weight-sharing worst case by ${\\approx}20$ points.',
        '  (c)~Per-client partitions of all benchmarks in the',
        '  (samples, dominant-share) plane (all seeds pooled; large marks',
        '  are dataset means).  HomeOccupancy alone occupies the',
        '  high-skew/low-data corner.}',
        '\\label{fig:partition}', '\\end{figure*}']
open('fig_partition.tex', 'w').write('\n'.join(tex))

# ---------- FIG imbalance ----------
# Per-client accuracy decline (drop vs own IID mean) under non-IID, all
# three benchmarks.  Top row: decline vs dominant-class share, per method.
# Bottom row: same decline vs samples per client, pooled over methods and
# split at the dataset's median share.
# HomeOccupancy FedAvg/FedProx: only the surviving seed (123) is drawn;
# the collapsed seeds form the dotted band at ~56 pts.
IMBDS = [
    # (title, pstats key, method->folder, seed-only per method,
    #  xmin xmax ymin ymax share_edges, n_xmin n_xmax n_edges)
    ('HomeOccupancy', 'home_occupancy',
     {'local': f'{HO}/local/uniform/noniid',
      'fedmd': 'oldresults/phase1/home_occupancy/fedmd/uniform/noniid',
      'fedakd': f'{HO}/fedakd/uniform/noniid',
      'mks': f'{HO}/mks/uniform/noniid',
      'fedavg': f'{HO}/fedavg/uniform/noniid',
      'fedprox': f'{HO}/fedprox/uniform/noniid'},
     {'fedavg': 1, 'fedprox': 1},
     30, 105, -12, 68, [30, 50, 65, 80, 106], 8, 300, [8, 20, 40, 80, 160, 320]),
    ('MNIST', 'mnist',
     {a: f'oldresults/phase1/mnist/{a}/uniform/noniid' for a in
      ['local', 'fedmd', 'fedakd', 'mks', 'fedavg', 'fedprox']}, {},
     10, 75, -8, 42, [10, 30, 45, 55, 75], 300, 9000,
     [300, 900, 1800, 3600, 7200, 9000]),
    ('CIFAR-10', 'cifar10',
     {a: f'oldresults/phase1/cifar10/{a}/uniform/noniid' for a in
      ['local', 'fedmd', 'fedakd', 'mks', 'fedavg', 'fedprox']}, {},
     10, 80, -12, 24, [10, 30, 45, 60, 80], 500, 9000,
     [500, 1200, 2400, 4800, 9600]),
]
METHODS = ['local', 'fedmd', 'fedakd', 'mks', 'fedavg', 'fedprox']
MNAMES = {'local': 'Local', 'fedmd': 'FedMD', 'fedakd': 'FedAKD',
          'mks': 'FedMKS', 'fedavg': 'FedAvg', 'fedprox': 'FedProx'}

def imb_pool(ds, folders, sonly):
    """per-method and pooled lists of (n, share%, drop pts) per client."""
    st = pstats[ds]
    per_method, pool = {}, []
    for algo in METHODS:
        a = load(folders[algo])
        if a is None:
            continue
        iidm = np.mean(seed_means(folders[algo].replace('noniid', 'iid')))
        C, T = a.shape
        pts = []
        for s in range(T // R):
            if algo in sonly and s != sonly[algo]:
                continue
            acc = a[:, s*R + R - 5:(s+1)*R].mean(axis=1) * 100
            for c in range(C):
                pts.append((st[s]['n'][c], st[s]['dom_share'][c] * 100,
                            iidm - acc[c]))
        per_method[algo] = pts
        pool += pts
    return per_method, np.array(pool)

def binned(xs, ys, edges, centre, min_n=2):
    line = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (xs >= lo) & (xs < hi)
        if m.sum() >= min_n:
            line.append(f'({centre(lo, hi):.1f},{ys[m].mean():.2f})')
    return line

tex = ['\\begin{figure*}[t]', '\\centering', '\\begin{tikzpicture}', '\\begin{groupplot}[',
       '  group style={', '    xlabels at=edge bottom,', '    ylabels at=edge left,',
       '    group size=3 by 2,', '    horizontal sep=1.15cm,',
       '    vertical sep=1.2cm,', '  },',
       '  width=0.31\\textwidth,', '  height=4.0cm,', ']']
# ---- top row: drop vs dominant-class share, per method ----
for i, (nm, ds, folders, sonly, xmn, xmx, ymn, ymx, sedges,
        nmn, nmx, nedges) in enumerate(IMBDS):
    per_method, pool = imb_pool(ds, folders, sonly)
    hdr = f'\\nextgroupplot[xlabel={{Dominant-class share (\\%)}},'
    if i == 0:
        hdr += '\n  ylabel={Acc.\\ drop vs.\\ IID (pts)},'
    hdr += (f'\n  title={{\\footnotesize {nm}}}, title style={{yshift=-2pt}},'
            f'\n  xmin={xmn}, xmax={xmx}, ymin={ymn}, ymax={ymx}, grid=major,'
            '\n  grid style={gray!18, line width=0.3pt},'
            '\n  tick label style={font=\\tiny}, label style={font=\\tiny}')
    hdr += (',\n  legend to name=legendImb,'
            '\n  legend style={legend columns=6, font=\\tiny}]' if i == 0 else ']')
    tex.append(hdr)
    centre = lambda lo, hi: (lo + hi) / 2
    for algo in METHODS:
        if algo not in per_method:
            continue
        pts = per_method[algo]
        pc = '\n'.join('  ' + ' '.join(f'({p[1]:.1f},{p[2]:.2f})' for p in pts[k:k+12])
                       for k in range(0, len(pts), 12))
        tex.append(f'\\addplot[only marks, {MK[algo]}, mark size=0.8pt, {FILL[algo]}, opacity=0.3, forget plot] coordinates {{\n{pc}}};')
        xs = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
        tex.append(f'\\addplot[{FILL[algo]}, very thick, mark=*, mark size=1.1pt] coordinates {{{" ".join(binned(xs, ys, sedges, centre))}}};')
        if i == 0:
            tex.append(f'\\addlegendentry{{{MNAMES[algo]}}}')
    tex.append(f'\\addplot[black!50, dashed, forget plot] coordinates {{({xmn},0) ({xmx},0)}};')
    if ds == 'home_occupancy':
        tex.append('\\addplot[black!60, dotted, thick, forget plot] coordinates {(30,56) (105,56)};')
        tex.append('\\node[font=\\tiny, black!60, anchor=south west] at (rel axis cs:0.02,0.86) {collapsed seeds};')
# ---- bottom row: drop vs samples per client, split at median share ----
for i, (nm, ds, folders, sonly, xmn, xmx, ymn, ymx, sedges,
        nmn, nmx, nedges) in enumerate(IMBDS):
    per_method, pool = imb_pool(ds, folders, sonly)
    med = float(np.median(pool[:, 1]))
    hdr = f'\\nextgroupplot[xmode=log, xlabel={{Samples per client}},'
    hdr += (f'\n  xmin={nmn}, xmax={nmx}, ymin={ymn}, ymax={ymx}, grid=major,'
            '\n  grid style={gray!18, line width=0.3pt},'
            '\n  tick label style={font=\\tiny}, label style={font=\\tiny}')
    hdr += (',\n  legend to name=legendImbB,'
            '\n  legend style={legend columns=2, font=\\tiny}]' if i == 0 else ']')
    tex.append(hdr)
    centre = lambda lo, hi: (lo * hi) ** 0.5
    sub_lo = pool[pool[:, 1] < med]
    sub_hi = pool[pool[:, 1] >= med]
    # shared x positions: a bin appears on both lines only if BOTH halves
    # have >=3 points in it, so the two means are always comparable.
    shared = [i for i in range(len(nedges) - 1)
              if ((sub_lo[:, 0] >= nedges[i]) & (sub_lo[:, 0] < nedges[i+1])).sum() >= 3
              and ((sub_hi[:, 0] >= nedges[i]) & (sub_hi[:, 0] < nedges[i+1])).sum() >= 3]
    for col, lab, sub in [('clGreen', 'below-median share', sub_lo),
                          ('clOrange', 'above-median share', sub_hi)]:
        pc = '\n'.join('  ' + ' '.join(f'({p[0]:.0f},{p[2]:.2f})' for p in sub[k:k+12])
                       for k in range(0, len(sub), 12))
        tex.append(f'\\addplot[only marks, mark=*, mark size=0.8pt, {col}, opacity=0.25, forget plot] coordinates {{\n{pc}}};')
        line = []
        for j in shared:
            m = (sub[:, 0] >= nedges[j]) & (sub[:, 0] < nedges[j+1])
            line.append(f'({centre(nedges[j], nedges[j+1]):.1f},{sub[m, 2].mean():.2f})')
        tex.append(f'\\addplot[{col}, very thick, mark=*, mark size=1.1pt] coordinates {{{" ".join(line)}}};')
        if i == 0:
            tex.append(f'\\addlegendentry{{{lab}}}')
    tex.append(f'\\addplot[black!50, dashed, forget plot] coordinates {{({nmn},0) ({nmx},0)}};')
tex += ['\\end{groupplot}', '\\end{tikzpicture}', '\\par\\vspace{4pt}',
        '\\ref{legendImb}\\\\[2pt]\\ref{legendImbB}',
        '\\caption{Per-client accuracy decline under non-IID, measured as',
        "  the drop from each method's own IID mean (dots: individual",
        '  clients, seeds pooled; thick lines: binned means).  Top:',
        '  decline vs.\\ dominant-class share, per method.',
        '  (a)~HomeOccupancy: the drop grows with share for every method',
        "  that keeps per-client models ($r{=}{+}0.73$ for FedProx's",
        '  surviving seed), while the two collapsed FedAvg/FedProx seeds',
        '  form a flat band at ${\\approx}56$ points (dotted)---their bias',
        '  is total and independent of the partition.  (b)~MNIST: weight',
        '  sharing loses essentially nothing; only local-model methods',
        '  drop ${\\approx}9$ points.  (c)~CIFAR-10: the share--decline',
        '  link is the strongest of any benchmark ($r{\\approx}{+}0.6$).',
        '  Bottom: the same decline vs.\\ samples per client, pooled over',
        "  methods and split at each dataset's median share.  On",
        '  HomeOccupancy (d)~more data does not rescue the above-median',
        '  half (extra samples are mostly more of the dominant class);',
        '  on MNIST (e)~drops are negligible at every size; on CIFAR-10',
        '  (f)~larger partitions reduce the drop for both halves---size',
        '  helps only where samples are already plentiful.}',
        '\\label{fig:imbalance}', '\\end{figure*}']
open('fig_imbalance.tex', 'w').write('\n'.join(tex))

# ---------- FIG difficulty ----------
# Why HomeOccupancy breaks weight averaging: three bars per dataset.
# (a) mean dominant-class share per client (Dirichlet a=0.5, verified).
# (b) FedAvg/Local accuracy ratio under non-IID.
# (c) difficulty per sample = (1 - silhouette)/n_client x10^3.
DSETS = ['HomeOcc', 'MNIST', 'CIFAR-10']
DCOL  = ['clRed', 'clBlue', 'clOlive']
share = [66.7, 38.2, 38.9]          # mean dominant share, %
ratio = [0.79, 1.14, 1.34]          # FedAvg / Local, non-IID
sil   = [0.038, 0.044, -0.063]      # input-space silhouette
ncl   = [52.4, 3000.0, 2500.0]      # mean samples per client
dps   = [(1 - s) / n * 1e3 for s, n in zip(sil, ncl)]   # difficulty/sample
xtick = ('symbolic x coords={HomeOcc, MNIST, CIFAR-10}, xtick=data,'
         ' ybar, bar width=9pt, enlarge x limits=0.3,')
tex = ['\\begin{figure*}[t]', '\\centering', '\\begin{tikzpicture}', '\\begin{groupplot}[',
       '  group style={', '    group size=3 by 1,', '    horizontal sep=1.1cm,', '  },',
       '  width=0.31\\textwidth,', '  height=4.6cm,',
       '  tick label style={font=\\tiny},',
       '  label style={font=\\tiny},', '  grid=major,',
       '  grid style={gray!18, line width=0.3pt},', ']']
# (a) dominant share
tex.append(f'\\nextgroupplot[{xtick} ylabel={{Mean dominant share (\\%)}}, ymin=0, ymax=95]')
for d, c, v in zip(DSETS, DCOL, share):
    tex.append(f'\\addplot[{c}, fill={c}!55] coordinates {{({d},{v})}};')
tex.append('\\addplot[black!60, dashed, forget plot] coordinates {(HomeOcc,60) (CIFAR-10,60)};')
# (b) FedAvg/Local
tex.append(f'\\nextgroupplot[{xtick} ylabel={{FedAvg / Local (non-IID)}}, ymin=0, ymax=1.6]')
for d, c, v in zip(DSETS, DCOL, ratio):
    tex.append(f'\\addplot[{c}, fill={c}!55] coordinates {{({d},{v})}};')
tex.append('\\addplot[black!60, dashed, forget plot] coordinates {(HomeOcc,1) (CIFAR-10,1)};')
# (c) difficulty per sample
tex.append(f'\\nextgroupplot[{xtick} ylabel={{Difficulty per sample}}, ymin=0, ymax=21]')
for d, c, v in zip(DSETS, DCOL, dps):
    tex.append(f'\\addplot[{c}, fill={c}!55] coordinates {{({d},{v:.2f})}};')
tex += ['\\end{groupplot}', '\\end{tikzpicture}',
        '\\caption{Why weight averaging fails on HomeOccupancy.',
        '  (a)~Mean dominant-class share per client under',
        '  $\\mathrm{Dir}(\\alpha{=}0.5)$: only HomeOccupancy crosses the',
        '  ${\\approx}60\\%$ level at which a single class dominates the',
        '  local optimum.  (b)~FedAvg-to-Local accuracy ratio under',
        '  non-IID: sharing beats local training on MNIST and CIFAR-10',
        '  but loses ${\\approx}21\\%$ on HomeOccupancy.  (c)~Difficulty',
        '  per sample, $(1{-}\\mathrm{silhouette})/n_{\\mathrm{client}}$',
        '  ($\\times10^{3}$): combining low input-space separability with',
        '  the smallest per-client partitions makes HomeOccupancy',
        '  ${\\approx}45{\\times}$ harder per sample than the image',
        '  benchmarks.}',
        '\\label{fig:difficulty}', '\\end{figure*}']
open('fig_difficulty.tex', 'w').write('\n'.join(tex))
print('done')
