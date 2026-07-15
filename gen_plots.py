"""Generate pgfplots \addplot coordinate blocks from phase1_coords.json."""
import json

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

PANELS = [
    ('home_occupancy','iid'),
    ('home_occupancy','noniid'),
    ('home_har','iid'),
    ('home_har','noniid'),
    ('mnist','iid'),
    ('mnist','noniid'),
    ('cifar10','iid'),
    ('cifar10','noniid'),
]

YRANGES = {
    ('home_occupancy','iid'):    (25, 100),
    ('home_occupancy','noniid'): (25, 100),
    ('home_har','iid'):          (14,  38),
    ('home_har','noniid'):       ( 5,  36),
    ('mnist','iid'):             (88, 100),
    ('mnist','noniid'):          (74, 100),
    ('cifar10','iid'):           (25,  75),
    ('cifar10','noniid'):        (22,  82),
}

def coord_str(vals):
    return ' '.join(f'({i+1},{v})' for i, v in enumerate(vals))

lines = []
for idx, (ds, setting) in enumerate(PANELS):
    ymin, ymax = YRANGES[(ds, setting)]
    is_first = (idx == 0)
    header = f'\\nextgroupplot[ymin={ymin}, ymax={ymax}'
    if is_first:
        header += (',\n  legend to name=globalLegend,\n'
                   '  legend style={legend columns=7, font=\\tiny,\n'
                   '    /tikz/every even column/.append style={column sep=0.3em}}')
    header += ']'
    lines.append(header)
    for algo in ALGO_ORDER:
        key = f'{ds}__{algo}__{setting}'
        if key not in coords:
            lines.append(f'% {algo}: NO DATA')
            continue
        vals = coords[key]
        style = ALGO_STYLE[algo]
        lines.append(f'{style} coordinates {{')
        # wrap coordinates ~80 chars per line
        cstr = coord_str(vals)
        # split into lines of ≤10 points each
        parts = [f'  ({i+1},{v})' for i,v in enumerate(vals)]
        chunk = 10
        for j in range(0, len(parts), chunk):
            lines.append('  ' + ' '.join(parts[j:j+chunk]))
        lines.append('};')
        if is_first:
            legend = {'central':'Central','fedmd':'FedMD','fedakd':'FedAKD',
                      'mks':'FedMKS','fedavg':'FedAvg','fedprox':'FedProx','local':'Local'}
            lines.append(f'\\addlegendentry{{{legend[algo]}}}')
    lines.append('')

print('\n'.join(lines))
