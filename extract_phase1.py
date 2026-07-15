import csv, os, json

algos = ['fedmd','fedakd','mks','fedavg','fedprox','local','central']
datasets = ['home_occupancy','home_har','mnist','cifar10']
settings = ['iid','noniid']
base = 'results/phase1'

results = {}
for ds in datasets:
    for algo in algos:
        for setting in settings:
            path = os.path.join(base, ds, algo, 'uniform', setting)
            if not os.path.isdir(path):
                continue
            files = sorted([f for f in os.listdir(path)
                            if f.startswith('train_') and f.endswith('.csv')])
            if not files:
                continue
            client_rounds = []
            for fn in files:
                with open(os.path.join(path, fn)) as f:
                    rows = list(csv.DictReader(f))
                epochs = [int(r['epoch']) for r in rows]
                vals = [float(r['val_accuracy'])*100 for r in rows]
                resets = ([0] +
                          [i for i in range(1, len(epochs)) if epochs[i] < epochs[i-1]] +
                          [len(epochs)])
                n_chunks = len(resets) - 1
                chunk_vals = [vals[resets[i+1]-1] for i in range(n_chunks)]
                # skip pre-training if more than 50 chunks
                if n_chunks > 50:
                    chunk_vals = chunk_vals[n_chunks - 50:]
                client_rounds.append(chunk_vals)
            n_rounds = min(len(c) for c in client_rounds)
            avg = [sum(c[r] for c in client_rounds) / len(client_rounds)
                   for r in range(n_rounds)]
            results[(ds, algo, setting)] = avg

# Save pgfplots coordinates keyed by (ds, algo, setting)
import json as _json
out = {}
for (ds, algo, setting), vals in results.items():
    out[f"{ds}__{algo}__{setting}"] = [round(v, 2) for v in vals]

with open('phase1_coords.json', 'w') as fout:
    _json.dump(out, fout, indent=2)
print("Saved phase1_coords.json")

# Also print a summary
for ds in datasets:
    for setting in settings:
        print(f"\n%% {ds} {setting}")
        for algo in algos:
            key = (ds, algo, setting)
            if key not in results:
                print(f"  % {algo}: NO DATA")
                continue
            vals = results[key]
            print(f"  % {algo} final={vals[-1]:.2f} n={len(vals)}")
