"""Apply all requested changes to paper.tex."""
import re, json

with open('paper.tex', encoding='utf-8') as f:
    text = f.read()

# ── 1. Replace groupplot body: high-res coords, no titles ────────────────────
with open('phase1_coords.json') as f:
    coords = json.load(f)

ALGO_STYLE = {
    'central': r'\addplot[black,thick,dashed]',
    'fedmd':   r'\addplot[clBlue]',
    'fedakd':  r'\addplot[clTeal]',
    'mks':     r'\addplot[clRed,thick]',
    'fedavg':  r'\addplot[clPurple]',
    'fedprox': r'\addplot[clOlive]',
    'local':   r'\addplot[clGray,dotted]',
}
ALGO_ORDER = ['central','fedmd','fedakd','mks','fedavg','fedprox','local']
LEGENDS = {'central':'Central','fedmd':'FedMD','fedakd':'FedAKD',
           'mks':'FedMKS','fedavg':'FedAvg','fedprox':'FedProx','local':'Local'}

PANELS = [
    ('home_occupancy','iid','25','100'),
    ('home_occupancy','noniid','25','100'),
    ('home_har','iid','14','38'),
    ('home_har','noniid','5','36'),
    ('mnist','iid','88','100'),
    ('mnist','noniid','74','100'),
    ('cifar10','iid','25','75'),
    ('cifar10','noniid','22','82'),
]

def make_groupplot_body():
    lines = []
    for idx,(ds,setting,ymin,ymax) in enumerate(PANELS):
        opts = f'ymin={ymin}, ymax={ymax}'
        if idx == 0:
            opts += (',\n  legend to name=globalLegend,\n'
                     '  legend style={legend columns=7, font=\\tiny,\n'
                     '    /tikz/every even column/.append style={column sep=0.3em}}')
        lines.append(f'\\nextgroupplot[{opts}]')
        for algo in ALGO_ORDER:
            key = f'{ds}__{algo}__{setting}'
            if key not in coords:
                continue
            vals = coords[key]
            style = ALGO_STYLE[algo]
            parts = [f'({i+1},{v})' for i,v in enumerate(vals)]
            coord_lines = []
            for j in range(0, len(parts), 10):
                coord_lines.append('  ' + ' '.join(parts[j:j+10]))
            block = style + ' coordinates {\n' + '\n'.join(coord_lines) + '};'
            lines.append(block)
            if idx == 0:
                lines.append(f'\\addlegendentry{{{LEGENDS[algo]}}}')
        lines.append('')
    return '\n'.join(lines)

new_body = make_groupplot_body()

# Find and replace everything between \begin{groupplot}[...] block and \end{groupplot}
# Pattern: from first \nextgroupplot up to (but not including) \end{groupplot}
old_gp_start = text.find(r'\nextgroupplot[title={HomeOccupancy (IID)}')
old_gp_end   = text.find(r'\end{groupplot}')
assert old_gp_start != -1, "groupplot start marker not found"
assert old_gp_end   != -1, "groupplot end marker not found"

text = text[:old_gp_start] + new_body + '\n' + text[old_gp_end:]

# ── 2. Remove Mixup and Tiers columns from Table 1 ───────────────────────────
# Header row
text = text.replace(
    r'\textbf{Algo.} & \textbf{Soft} & \textbf{Wts} & \textbf{Mixup} &'
    '\n  \\textbf{Tiers} & \\textbf{Topology}\\\\',
    r'\textbf{Algo.} & \textbf{Soft} & \textbf{Wts} & \textbf{Topology}\\'
)
# tabular format
text = text.replace(r'\begin{tabular}{@{}lccccl@{}}',
                    r'\begin{tabular}{@{}lccl@{}}')
# Data rows — remove the two middle columns
rows_old = [
    r'FedMD    & \checkmark & --         & --         & --         & Heterogeneous \\',
    r'FedAKD   & \checkmark & --         & \checkmark & --         & Heterogeneous \\',
    r'FedMKS   & \checkmark & \checkmark & \checkmark & \checkmark & Semi-het.     \\',
    r'FedAvg   & --         & \checkmark & --         & --         & Homogeneous   \\',
    r'FedProx  & --         & \checkmark & --         & --         & Homogeneous   \\',
    r'Local    & --         & --         & --         & --         & Isolated      \\',
    r'Central  & --         & --         & --         & --         & Centralised   \\',
]
rows_new = [
    r'FedMD    & \checkmark & --         & Heterogeneous \\',
    r'FedAKD   & \checkmark & --         & Heterogeneous \\',
    r'FedMKS   & \checkmark & \checkmark & Semi-het.     \\',
    r'FedAvg   & --         & \checkmark & Homogeneous   \\',
    r'FedProx  & --         & \checkmark & Homogeneous   \\',
    r'Local    & --         & --         & Isolated      \\',
    r'Central  & --         & --         & Centralised   \\',
]
for old, new in zip(rows_old, rows_new):
    text = text.replace(old, new)

# ── 3. Rewrite System Model section ─────────────────────────────────────────
old_sysmodel = r"""\section{System Model}
\label{sec:sysmodel}

\subsection{Preliminaries}

Let $\mathcal{K}=\{1,\ldots,K\}$ be a set of $K{=}20$ clients.  Each client
$k$ holds a private labelled dataset $\mathcal{D}_k = \{(x_i^k, y_i^k)\}$
drawn from local activity class set $\mathcal{Y}_k \subseteq \mathcal{Y}$.
All clients share access to a small public \emph{unlabelled or freely
available} dataset $\mathcal{D}^{\text{pub}}$ (e.g., HomeHAR for
HomeOccupancy clients) used solely for knowledge transfer.

Each client $k$ is assigned an \emph{architecture tier}
$t_k \in \{\text{Small, Medium, Large}\}$ reflecting its compute budget.
A \textbf{dual-head model} $f_k = (f_k^{\text{clf}}, f_k^{\text{dist}})$
shares a backbone $\phi_k$ parameterised by $\theta_k^{\text{backbone}}$:
the \emph{classification head} $f_k^{\text{clf}}$ predicts class probabilities
$\hat{y}$ on private data; the \emph{distillation head} $f_k^{\text{dist}}$
produces soft-label carrier scores $s_k = f_k^{\text{dist}}(\mathcal{D}^{\text{pub}})
\in \mathbb{R}^{|\mathcal{D}^{\text{pub}}| \times C}$.

\textbf{CSI feature pipeline.}
Raw 52-subcarrier complex CSI frames are amplitude-extracted, resampled to
150\,Hz, and segmented into non-overlapping windows of $T{=}1500$ samples.
Per window, the pipeline computes: (1)~mean subcarrier amplitude
($1$\,ch.), (2)~rolling variance at scales $W{\in}\{15,150,1500\}$
($3$\,ch.), and (3)~$B{=}8$ STFT magnitude bins ($8$\,ch.), concatenated to
a $(T,\,12)$ tensor.  All channels are normalised with a
\texttt{MinMaxScaler} fitted on the training split.

\subsection{Problem Definition}

\textbf{Goal.} Learn per-client classifiers $\{f_k^{\text{clf}}\}$ that
minimise validation loss on each client's held-out private data, subject to:
\begin{enumerate}
  \item \emph{Privacy}: no raw data $\mathcal{D}_k$ leaves client $k$.
  \item \emph{Heterogeneity}: clients $j,k$ may have different architectures
    ($t_j \neq t_k$), prohibiting direct weight averaging.
  \item \emph{Communication efficiency}: transmitted payloads per round are
    bounded.
\end{enumerate}

Under IID partitioning, each client receives $n_c{=}30$ samples per class via
random sampling.  Under non-IID partitioning, class proportions follow a
Dirichlet distribution $\text{Dir}(\alpha)$ with $\alpha{=}0.5$,
creating highly skewed per-client class distributions."""

new_sysmodel = r"""\section{System Model}
\label{sec:sysmodel}

\subsection{Notation}

\begin{table}[t]
\centering\small
\caption{Symbol table.}
\label{tab:symbols}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}cl@{}}
\toprule
\textbf{Symbol} & \textbf{Meaning}\\
\midrule
$K$                          & Number of clients ($K{=}20$)\\
$\mathcal{K}$                & Client index set $\{1,\ldots,K\}$\\
$R$                          & Number of FL communication rounds ($R{=}50$)\\
$r$                          & Round index, $r\!\in\!\{1,\ldots,R\}$\\
$\mathcal{D}_k$              & Private labelled dataset of client $k$\\
$n_k$                        & $|\mathcal{D}_k|$; $n = \sum_k n_k$\\
$\mathcal{Y}$                & Global label space; $C{=}|\mathcal{Y}|$\\
$\mathcal{Y}_k$              & Local label subset, $\mathcal{Y}_k \subseteq \mathcal{Y}$\\
$\mathcal{D}^{\text{pub}}$   & Shared public (unlabelled) dataset\\
$t_k$                        & Tier of client $k$: S, M, or L\\
$\theta_k$                   & All trainable parameters of model $f_k$\\
$\theta_k^{\text{bb}}$       & Backbone parameters (shared within tier for MKS)\\
$\mathbf{W}_t^{(r)}$         & Server-side averaged backbone weights for tier $t$ at round $r$\\
$s_k^{(r)}$                  & Soft-label scores from client $k$ at round $r$,
                               $s_k^{(r)}\!\in\!\mathbb{R}^{|\mathcal{D}^{\text{pub}}|\times C}$\\
$\bar{s}^{(r)}$              & Aggregated global soft labels at round $r$\\
$T,B$                        & CSI window length ($1500$) and STFT bins ($8$)\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Network and Data Model}

Consider a star-topology FL network with a central server and
$K{=}20$ clients.  Each client $k\!\in\!\mathcal{K}$ holds a private
labelled dataset
$\mathcal{D}_k = \{(x_i^k, y_i^k)\}_{i=1}^{n_k}$,
where $x_i^k \in \mathbb{R}^{T \times (4+B)}$ is a preprocessed CSI window
and $y_i^k \in \mathcal{Y}_k \subseteq \mathcal{Y}$ is the activity label.
Under \emph{IID} partitioning, each client receives $n_c{=}30$ labelled
examples per class.  Under \emph{non-IID} partitioning, class proportions are
drawn from a Dirichlet distribution
$\mathbf{p}_k \sim \mathrm{Dir}(\alpha)$ with $\alpha{=}0.5$,
creating highly skewed per-client class distributions.

All clients share access to a freely available \emph{public} dataset
$\mathcal{D}^{\text{pub}}$ (e.g.\ HomeHAR for HomeOccupancy clients) that
carries no private labels and is used exclusively for knowledge transfer.

\subsection{Heterogeneous Model Architecture}

Each client is assigned a compute tier
$t_k \in \{\text{S}, \text{M}, \text{L}\}$ reflecting its hardware budget.
Every tier instantiates a \emph{dual-head} model
$f_k = (\phi_k,\, h_k^{\text{clf}},\, h_k^{\text{dist}})$,
where $\phi_k$ is a shared backbone parameterised by
$\theta_k^{\text{bb}}$ and the two heads are trained with decoupled losses:

\begin{itemize}
  \item \textbf{Classification head} $h_k^{\text{clf}}$: trained on private
    data with cross-entropy loss
    $\mathcal{L}_{\text{CE}}(\theta_k) =
     -\frac{1}{n_k}\sum_{i} \log p_k(y_i^k \mid x_i^k)$.
  \item \textbf{Distillation head} $h_k^{\text{dist}}$: produces a
    soft-label carrier
    $s_k^{(r)} = h_k^{\text{dist}}(\phi_k(\mathcal{D}^{\text{pub}}))
     \in \mathbb{R}^{|\mathcal{D}^{\text{pub}}| \times C}$,
    aligned to the global aggregate via
    $\mathcal{L}_{\text{KD}}(\theta_k) =
     \frac{1}{|\mathcal{D}^{\text{pub}}|}
     \|s_k^{(r)} - \bar{s}^{(r-1)}\|_F^2$.
\end{itemize}

Backbone depth grows with tier: \textbf{S} uses Conv1D$\times3$;
\textbf{M} adds BiLSTM(128); \textbf{L} further adds a
Transformer attention block.

\subsection{Traditional FL Objective and Its Limitations}

Standard FL (FedAvg~\cite{mcmahan2017fedavg}) seeks a single global model
$\theta^*$ minimising the weighted empirical risk:
\begin{equation}
  \theta^* = \arg\min_{\theta}
  \sum_{k=1}^{K} \frac{n_k}{n}\,
  \mathcal{L}_{\text{CE}}^k(\theta),
  \label{eq:fedavg}
\end{equation}
aggregating client updates each round as
$\theta^{(r)} \gets \sum_k (n_k/n)\,\theta_k^{(r)}$.
This formulation requires \emph{all clients to share the same architecture},
making it inapplicable when $t_j \neq t_k$.  FedProx~\cite{li2020fedprox}
adds a proximal term
$\frac{\mu}{2}\|\theta_k - \theta^{(r-1)}\|^2$ to each client's loss to
bound client drift under heterogeneous data, but still enforces architectural
homogeneity.

\subsection{Knowledge-Distillation-Based FL Objective}

To support fully heterogeneous architectures,
FedMD~\cite{li2019fedmd} replaces weight aggregation with soft-label
aggregation.  The server maintains a global soft-label matrix
$\bar{s}^{(r)} = \sum_k (n_k/n)\,s_k^{(r)}$ and each client minimises the
combined objective:
\begin{equation}
  \mathcal{L}_k(\theta_k) =
  \underbrace{\mathcal{L}_{\text{CE}}(\theta_k;\,\mathcal{D}_k)}_{\text{private classification}}
  +\;
  \lambda\,\underbrace{\mathcal{L}_{\text{KD}}(\theta_k;\,\mathcal{D}^{\text{pub}},\bar{s}^{(r-1)})}_{\text{KD alignment}},
  \label{eq:fedkd}
\end{equation}
where $\lambda$ balances local accuracy with global knowledge absorption.
Because $\bar{s}^{(r)}$ lives in a \emph{shared label space} $\mathcal{Y}$,
clients with entirely different backbone architectures can participate in the
same aggregation round.

\subsection{Problem Definition}

\textbf{Goal.} Given $K$ clients with heterogeneous private datasets
$\{\mathcal{D}_k\}$ and heterogeneous model tiers $\{t_k\}$, find
per-client classifiers $\{f_k^{\text{clf}}\}$ that jointly maximise
mean validation accuracy across clients, subject to three constraints:
\begin{enumerate}
  \item \emph{Privacy}: $\mathcal{D}_k$ never leaves client $k$;
    only $s_k^{(r)}$ (and optionally $\theta_k^{\text{bb}}$ within-tier)
    are transmitted.
  \item \emph{Architectural heterogeneity}: no direct weight transfer
    between clients of different tiers ($t_j \neq t_k$).
  \item \emph{Communication bound}: payload per round per client is
    $\mathcal{O}(|\mathcal{D}^{\text{pub}}|\cdot C)$ for KD methods,
    versus $\mathcal{O}(|\theta|)$ for weight-sharing methods.
\end{enumerate}

\noindent FedMKS (Section~\ref{sec:methods}) addresses all three constraints
simultaneously by combining within-tier weight averaging
(Eq.~\ref{eq:fedavg} applied per tier) with cross-tier soft-label
distillation (Eq.~\ref{eq:fedkd}), bridging the gap between FedAvg
efficiency and FedMD flexibility."""

text = text.replace(old_sysmodel, new_sysmodel)

with open('paper.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print("Done. Verifying...")
assert r'\nextgroupplot[ymin=25, ymax=100,' in text, "groupplot not updated"
assert r'tab:symbols' in text, "symbol table not added"
assert r'\begin{tabular}{@{}lccl@{}}' in text, "table columns not reduced"
assert r'(1,78.74)' in text, "per-round coords not present"
assert r'title={HomeOccupancy' not in text, "subplot titles still present"
print("All assertions passed.")
