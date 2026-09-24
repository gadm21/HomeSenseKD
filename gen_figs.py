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
       '    group size=4 by 1,', '    horizontal sep=0.9cm,', '  },',
       '  width=0.23\\textwidth,', '  height=4.2cm,', ']']
rng3 = np.random.RandomState(2)
for p in range(4):
    title = 'Local' if p == 0 else f'FedAvg seed {p}'
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
    if p > 0:
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
# Data distribution as the deciding factor.
# (a) HomeOcc non-IID: dominant-class share vs per-client final accuracy.
# (b) all datasets: samples per client (log) vs dominant-class share.
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

tex = ['\\begin{figure*}[t]', '\\centering', '\\begin{tikzpicture}', '\\begin{groupplot}[',
       '  group style={', '    group size=2 by 1,', '    horizontal sep=1.5cm,', '  },',
       '  width=0.44\\textwidth,', '  height=4.6cm,', ']']
# ---- (a) dominant share vs accuracy, HomeOccupancy non-IID ----
tex += ['\\nextgroupplot[xlabel={Dominant-class share (\\%)},',
        '  ylabel={Val.\\ Acc.\\ (\\%)},',
        '  xmin=30, xmax=105, ymin=20, ymax=95, grid=major,',
        '  grid style={gray!18, line width=0.3pt},',
        '  tick label style={font=\\tiny}, label style={font=\\tiny},',
        '  legend style={font=\\tiny, at={(0.03,0.97)}, anchor=north west,',
        '    draw=none, fill=none}]']
PA = [('local', 'Local', 'results/phase1/home_occupancy/local/uniform/noniid'),
      ('fedmd', 'FedMD', 'oldresults/phase1/home_occupancy/fedmd/uniform/noniid'),
      ('fedavg', 'FedAvg', 'results/phase1/home_occupancy/fedavg/uniform/noniid')]
for algo, nm, folder in PA:
    pts = share_acc_pts(folder)
    pc = '\n'.join('  ' + ' '.join(f'({x:.1f},{y:.2f})' for x, y in pts[k:k+12])
                   for k in range(0, len(pts), 12))
    tex.append(f'\\addplot[only marks, mark=*, mark size=0.9pt, {FILL[algo]}, opacity=0.6] coordinates {{\n{pc}}};')
    tex.append(f'\\addlegendentry{{{nm}}}')
tex.append('\\addplot[clGray, dashed, forget plot] coordinates {(30,33.3) (105,33.3)};')
# ---- (b) samples per client vs dominant share, all datasets ----
tex += ['\\nextgroupplot[xmode=log,',
        '  xlabel={Samples per client}, ylabel={Dominant-class share (\\%)},',
        '  xmin=8, xmax=9000, ymin=15, ymax=105, grid=major,',
        '  grid style={gray!18, line width=0.3pt},',
        '  tick label style={font=\\tiny}, label style={font=\\tiny},',
        '  legend style={font=\\tiny, at={(0.97,0.03)}, anchor=south east,',
        '    draw=none, fill=none}]']
DSCOL = {'home_occupancy': ('clRed', 'HomeOcc (3 cls)'),
         'mnist': ('clBlue', 'MNIST (10 cls)'),
         'cifar10': ('clOlive', 'CIFAR-10 (10 cls)')}
for ds, (col, nm) in DSCOL.items():
    pts = [(n, sh * 100) for sd in pstats[ds]
           for n, sh in zip(sd['n'], sd['dom_share'])]
    pc = '\n'.join('  ' + ' '.join(f'({x},{y:.1f})' for x, y in pts[k:k+12])
                   for k in range(0, len(pts), 12))
    tex.append(f'\\addplot[only marks, mark=*, mark size=0.9pt, {col}, opacity=0.55] coordinates {{\n{pc}}};')
    tex.append(f'\\addlegendentry{{{nm}}}')
    mn = float(np.exp(np.mean(np.log([p[0] for p in pts]))))
    ms = float(np.mean([p[1] for p in pts]))
    tex.append(f'\\addplot[only marks, mark=*, mark size=2.2pt, {col}!45!black, forget plot] coordinates {{({mn:.0f},{ms:.1f})}};')
tex += ['\\end{groupplot}', '\\end{tikzpicture}',
        '\\caption{Data distribution decides per-client accuracy.',
        "  (a)~HomeOccupancy non-IID: each client's final accuracy vs.\\ the",
        '  dominant-class share of its Dirichlet partition.  Local and FedMD',
        '  accuracies degrade as the partition approaches a single class',
        '  ($r{=}{-}0.36$ for Local); FedAvg clients in the two collapsed',
        '  seeds sit at chance at \\emph{every} share---collapse is a global',
        "  property of the averaged model, not of any client's data.",
        '  (b)~Per-client partitions of all benchmarks in the',
        '  (samples, dominant-share) plane (all seeds pooled; large dots mark',
        '  dataset means).  HomeOccupancy occupies the high-skew/low-data',
        '  corner---the regime where weight averaging can collapse---while',
        '  MNIST and CIFAR-10 clients hold ${\\approx}60{\\times}$ more',
        '  samples at lower dominant shares.}',
        '\\label{fig:partition}', '\\end{figure*}']
open('fig_partition.tex', 'w').write('\n'.join(tex))
print('done')
