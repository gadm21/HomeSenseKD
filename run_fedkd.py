#!/usr/bin/env python3
"""
run_fedkd.py — Modular federated-learning experiment runner.

Algorithms : fedmd | fedakd | mks | fedavg | fedprox | local | central
Datasets   : home_occupancy | home_har | mnist | cifar10

Usage examples
--------------
# Run all algorithms on HomeOccupancy with default config
python run_fedkd.py --config config/home_occupancy.yaml

# Run only FedMD + FedAKD, skewed heterogeneity, single seed
python run_fedkd.py --config config/mnist.yaml \\
    --algorithm fedmd fedakd --heterogeneity skewed --seed 42

# Override config keys inline
python run_fedkd.py --config config/cifar10.yaml \\
    --override federated.n_rounds=30 data.n_parties=10
"""

import os
import sys
import argparse
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import yaml
    _YAML = True
except ImportError:
    _YAML = False

from utils import (
    # data
    load_homeoccupancy, load_homehar, load_mnist, load_cifar10,
    resize_images, prepare_datasets, split_dataset,
    iid_partition, dirichlet_partition,
    # models
    build_tiered_model, assign_client_tiers,
    # nodes
    Node,
    # training helpers
    fedprox_train,
    # aggregation
    fedavg_aggregate, aggregate_soft_labels,
    aggregate_by_cluster, aggregate_by_tier,
    # plotting
    plot_data_partitioning, plot_training_curves,
    plot_final_comparison, plot_per_client_accuracy,
    plot_final_comparison_ci, plot_tier_distribution,
    # misc
    CLASS_NAMES,
)

# ── Default configuration ──────────────────────────────────────────────────────

DEFAULT_CFG = {
    "experiment": {
        "name": "fedkd_experiment",
        "results_dir": "results",
        "fig_dir": "results/figures",
        "log_dir": "logs",
        "seeds": [42, 123, 456],
    },
    "data": {
        "dataset": "home_occupancy",
        "public_dataset": "home_har",
        "n_classes": 3,
        "n_parties": 20,
        "n_samples_per_class": 30,
        "dirichlet_alpha": 0.5,
        "window_size": 100,
        "n_stft_bins": 8,
        "settings": ["iid", "noniid"],
    },
    "federated": {
        "n_rounds": 50,
        "local_epochs": 4,
        "kd_epochs": 2,
        "fedprox_mu": 0.01,
        "n_alignment": 10000,
    },
    "models": {
        "heterogeneity": "uniform",
        "distributions": {
            "all_small": [1.00, 0.00, 0.00],
            "uniform":   [0.35, 0.35, 0.30],
            "skewed":    [0.60, 0.30, 0.10],
        },
    },
    "algorithms": ["fedmd", "fedakd", "mks", "fedavg", "fedprox", "local", "central"],
}


def _deep_merge(base, override):
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def load_config(config_file=None, overrides=None):
    """Load YAML config, deep-merge over defaults, apply CLI overrides."""
    cfg = deepcopy(DEFAULT_CFG)
    if config_file:
        if os.path.exists(config_file):
            with open(config_file) as f:
                file_cfg = yaml.safe_load(f) if _YAML else {}
            cfg = _deep_merge(cfg, file_cfg or {})
        else:
            print(f"[warn] Config {config_file!r} not found — using defaults")
    if overrides:
        for kv in overrides:
            parts = kv.split("=", 1)
            if len(parts) != 2:
                continue
            keys = parts[0].strip().split(".")
            val = (yaml.safe_load(parts[1].strip()) if _YAML else parts[1].strip())
            d = cfg
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = val
    return cfg


# ── Dataset factory ────────────────────────────────────────────────────────────

def _get_input_shape(dc):
    """Return (input_shape, is_image) from data config."""
    if dc["dataset"] in ("home_occupancy", "home_har"):
        n_features = 4 + dc.get("n_stft_bins", 8)
        return (dc["window_size"], n_features), False
    h = dc.get("img_size", 28)
    c = dc.get("in_channels", 1)
    return (h, h, c), True


def get_dataset(dc):
    """Load target + public datasets.  Returns (x_tr, y_tr, x_te, y_te, pub_x, pub_y)."""
    ds = dc["dataset"]
    pub_ds = dc["public_dataset"]
    input_shape, is_image = _get_input_shape(dc)

    # ── Load target ──────────────────────────────────────────────────
    nstft = dc.get("n_stft_bins", 8)
    nf    = 4 + nstft
    if ds == "home_occupancy":
        x_tr, y_tr, x_te, y_te = load_homeoccupancy(dc["window_size"], nstft)
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
        x_tr = sc.fit_transform(x_tr.reshape(-1, nf)).reshape(x_tr.shape)
        x_te = sc.transform(x_te.reshape(-1, nf)).reshape(x_te.shape)
    elif ds == "home_har":
        x_tr_har, y_tr_har = load_homehar(dc["window_size"], nstft)
        split = int(0.8 * len(x_tr_har))
        x_tr, y_tr = x_tr_har[:split], y_tr_har[:split]
        x_te, y_te = x_tr_har[split:], y_tr_har[split:]
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
        x_tr = sc.fit_transform(x_tr.reshape(-1, nf)).reshape(x_tr.shape)
        x_te = sc.transform(x_te.reshape(-1, nf)).reshape(x_te.shape)
    elif ds == "mnist":
        x_tr, y_tr, x_te, y_te = load_mnist()
    elif ds == "cifar10":
        x_tr, y_tr, x_te, y_te = load_cifar10()
    else:
        raise ValueError(f"Unknown dataset: {ds!r}")

    # ── Load public ──────────────────────────────────────────────────
    if pub_ds == "home_har":
        pub_x, pub_y = load_homehar(dc["window_size"], nstft)
        from sklearn.preprocessing import MinMaxScaler
        sc2 = MinMaxScaler()
        pub_x = sc2.fit_transform(pub_x.reshape(-1, nf)).reshape(pub_x.shape)
    elif pub_ds == "home_occupancy":
        pub_x, pub_y, _, _ = load_homeoccupancy(dc["window_size"], nstft)
        from sklearn.preprocessing import MinMaxScaler
        sc2 = MinMaxScaler()
        pub_x = sc2.fit_transform(pub_x.reshape(-1, nf)).reshape(pub_x.shape)
    elif pub_ds == "mnist":
        pub_x, pub_y, _, _ = load_mnist()
    elif pub_ds == "cifar10":
        pub_x, pub_y, _, _ = load_cifar10()
    else:
        raise ValueError(f"Unknown public_dataset: {pub_ds!r}")

    # ── Resize public data to match target input shape ───────────────
    if is_image and pub_x.ndim == 4 and pub_x.shape[1:] != input_shape:
        pub_x = resize_images(pub_x, input_shape)
    elif not is_image and pub_x.ndim == 3 and pub_x.shape[1:] != input_shape:
        pass  # same window_size/n_features assumed for CSI datasets

    y_te_cat = tf.keras.utils.to_categorical(y_te, num_classes=dc["n_classes"])
    return x_tr, y_tr, x_te, y_te_cat, pub_x, pub_y


def make_partitions(x_tr, y_tr, dc, fc):
    """Return IID and non-IID private splits for all clients (categorical y)."""
    n_cls = dc["n_classes"]
    n_par = dc["n_parties"]
    spc   = dc["n_samples_per_class"]
    alpha = dc["dirichlet_alpha"]

    pri_x_iid, pri_y_iid = iid_partition(
        x_tr, y_tr, n_par, spc,
        n_classes_total=n_cls, to_categorical_flag=True)

    pri_x_nid, pri_y_nid = dirichlet_partition(
        x_tr, y_tr, n_par, alpha,
        n_classes_total=n_cls, to_categorical_flag=True)

    return pri_x_iid, pri_y_iid, pri_x_nid, pri_y_nid


# ── Model factory ──────────────────────────────────────────────────────────────

def build_client_models(algo, tiers, input_shape, n_classes):
    """Build per-client (model_A, model_B) pairs.

    FedAvg / FedProx: all clients use 'small' (smallest common architecture).
    KD algorithms   : each client uses their assigned tier architecture.
    """
    if algo in ("fedavg", "fedprox"):
        return [build_tiered_model("small", input_shape, n_classes) for _ in tiers]
    return [build_tiered_model(t, input_shape, n_classes) for t in tiers]


# ── Flower clients ─────────────────────────────────────────────────────────────

class KDClient:
    """Knowledge-distillation client (FedMD / FedAKD / MKS).

    Parameter encoding:
      FedMD/FedAKD : parameters = [soft_labels]
      MKS          : parameters = [soft_labels, *cluster_model_weights]
    """

    def __init__(self, cid, node, exp_dir, setting, kd_epochs, local_epochs, tier="medium"):
        self.cid          = int(cid)
        self.node         = node
        self.exp_dir      = exp_dir
        self.setting      = setting
        self.kd_epochs    = kd_epochs
        self.local_epochs = local_epochs
        self.tier         = tier

    def fit(self, parameters, config):
        rnd           = int(config["round_num"])
        seed          = int(config.get("seed", 0))
        alpha         = float(config.get("alpha", 0.5))
        use_cluster_w = bool(config.get("use_cluster_weights", False))

        if rnd > 1 and len(parameters) > 0:
            self.node.receive_training_metadata(parameters[0])
            if use_cluster_w and len(parameters) > 1:
                self.node.model[0].set_weights(parameters[1:])
            self.node.train_on_public(epochs=self.kd_epochs, verbose=False)

        log = os.path.join(self.exp_dir, self.setting, f"train_{self.cid}.csv")
        self.node.train_on_target(epochs=self.local_epochs, verbose=False,
                                  logger_file=log, evaluate=True)
        scores, acc = self.node.get_training_metadata(seed, alpha)
        weights      = self.node.model[0].get_weights()
        n            = len(self.node.local_target_dataset[0])
        return [scores] + weights, n, {"accuracy": float(acc)}


class WeightClient:
    """Weight-sharing client (FedAvg / FedProx).

    All clients share the same (small) architecture.
    FedProx adds proximal term: L_ce + (mu/2)||w - w_global||^2.
    """

    def __init__(self, cid, node, exp_dir, setting, local_epochs,
                 fedprox=False, mu=0.01):
        self.cid          = int(cid)
        self.node         = node
        self.exp_dir      = exp_dir
        self.setting      = setting
        self.local_epochs = local_epochs
        self.fedprox      = fedprox
        self.mu           = mu

    def fit(self, parameters, config):
        rnd = int(config["round_num"])
        if rnd > 1 and len(parameters) > 0:
            self.node.model[0].set_weights(parameters)

        log = os.path.join(self.exp_dir, self.setting, f"train_{self.cid}.csv")
        x, y = self.node.local_target_dataset
        val  = self.node.target_validation_gen

        if self.fedprox and rnd > 1 and len(parameters) > 0:
            fedprox_train(self.node.model[0], x, y, parameters,
                          mu=self.mu, epochs=self.local_epochs,
                          validation_data=val, csv_path=log)
        else:
            self.node.train_on_target(epochs=self.local_epochs, verbose=False,
                                      logger_file=log, evaluate=True)

        acc = self.node.evaluate_on_validation_set(save=False)[1]
        return self.node.model[0].get_weights(), len(x), {"accuracy": float(acc)}


# ── Simulation runners ─────────────────────────────────────────────────────────

def _log_round(rnd, total, accs, stats):
    avg = float(np.mean(accs))
    mn  = float(np.min(accs))
    mx  = float(np.max(accs))
    stats["avg"].append(avg)
    stats["min"].append(mn)
    stats["max"].append(mx)
    print(f"  Round {rnd:3d}/{total} | Avg: {avg:.4f} | Min: {mn:.4f} | Max: {mx:.4f}")


def run_kd(clients, n_rounds, name):
    """FedMD / FedAKD: exchange soft labels only."""
    print(f"\n{'='*65}\n  {name}\n{'='*65}")
    soft_labels = None
    stats = {"avg": [], "min": [], "max": []}
    for rnd in range(1, n_rounds + 1):
        seed  = int(np.random.randint(0, 10000))
        alpha = float(np.random.rand())
        params_in   = [soft_labels] if soft_labels is not None else []
        all_scores, accs = [], []
        for c in clients:
            res, _, m = c.fit(params_in, {"round_num": rnd, "seed": seed, "alpha": alpha})
            all_scores.append(res[0])
            accs.append(float(m["accuracy"]))
        soft_labels = aggregate_soft_labels(all_scores, accs)
        _log_round(rnd, n_rounds, accs, stats)
    print(f"  {name} done.")
    return stats


def run_mks(clients, n_rounds, name, tier_map):
    """MKS: tier-based FedAvg within clusters + global KD across all.

    Server clusters clients by model tier (small/medium/large).
    Each round:
      1. Clients send soft labels + model weights.
      2. Server: global soft-label average + per-tier FedAvg.
      3. Server returns cluster model + global soft labels to each client.
    """
    print(f"\n{'='*65}\n  {name}\n{'='*65}")
    soft_labels    = None
    cluster_w      = {}           # tier -> averaged weights
    stats          = {"avg": [], "min": [], "max": []}

    for rnd in range(1, n_rounds + 1):
        seed  = int(np.random.randint(0, 10000))
        alpha = float(np.random.rand())
        all_scores, all_w, all_n, accs = [], [], [], []
        tier_results = {}

        for c in clients:
            tier = tier_map[c.cid]
            cw   = cluster_w.get(tier, [])
            if soft_labels is not None and cw:
                params_in = [soft_labels] + cw
                use_cw    = True
            elif soft_labels is not None:
                params_in = [soft_labels]
                use_cw    = False
            else:
                params_in = []
                use_cw    = False

            cfg_r = {"round_num": rnd, "seed": seed, "alpha": alpha,
                     "use_cluster_weights": use_cw}
            res, n, m = c.fit(params_in, cfg_r)
            all_scores.append(res[0])
            all_w.append(res[1:])
            all_n.append(n)
            accs.append(float(m["accuracy"]))
            tier_results.setdefault(tier, {"w": [], "n": []})
            tier_results[tier]["w"].append(res[1:])
            tier_results[tier]["n"].append(n)

        soft_labels = aggregate_soft_labels(all_scores, accs)
        cluster_w   = {t: fedavg_aggregate(d["w"], d["n"])
                       for t, d in tier_results.items()}
        _log_round(rnd, n_rounds, accs, stats)

    print(f"  {name} done.")
    return stats


def run_weight_sharing(clients, n_rounds, name):
    """FedAvg / FedProx: global model broadcast each round."""
    print(f"\n{'='*65}\n  {name}\n{'='*65}")
    global_w = []
    stats    = {"avg": [], "min": [], "max": []}

    for rnd in range(1, n_rounds + 1):
        all_w, all_n, accs = [], [], []
        for c in clients:
            res, n, m = c.fit(global_w, {"round_num": rnd})
            all_w.append(res)
            all_n.append(n)
            accs.append(float(m["accuracy"]))
        global_w = fedavg_aggregate(all_w, all_n)
        _log_round(rnd, n_rounds, accs, stats)

    print(f"  {name} done.")
    return stats


def run_local(nodes, n_rounds, local_epochs, exp_dir, setting):
    """Standalone local training — no communication."""
    print(f"\n{'='*65}\n  Local — {setting.upper()}\n{'='*65}")
    os.makedirs(os.path.join(exp_dir, setting), exist_ok=True)
    stats = {"avg": [], "min": [], "max": []}
    for rnd in range(1, n_rounds + 1):
        accs = []
        for i, node in enumerate(nodes):
            log = os.path.join(exp_dir, setting, f"train_{i}.csv")
            node.train_on_target(epochs=local_epochs, verbose=False,
                                 logger_file=log, evaluate=True)
            accs.append(node.evaluate_on_validation_set(save=False)[1])
        _log_round(rnd, n_rounds, accs, stats)
    print("  Local done.")
    return stats


def run_central(pri_x_list, pri_y_list, val_data, n_rounds, local_epochs,
                input_shape, n_classes, exp_dir, setting):
    """Centralised training — all private data pooled on server."""
    print(f"\n{'='*65}\n  Central — {setting.upper()}\n{'='*65}")
    os.makedirs(os.path.join(exp_dir, setting), exist_ok=True)
    x_all = np.concatenate([x for x in pri_x_list if len(x)])
    y_all = np.concatenate([y for y in pri_y_list if len(y)])
    model, _ = build_tiered_model("large", input_shape, n_classes)
    log = os.path.join(exp_dir, setting, "central.csv")
    stats = {"avg": [], "min": [], "max": []}
    for rnd in range(1, n_rounds + 1):
        cbs = [tf.keras.callbacks.CSVLogger(log, append=True)]
        model.fit(x_all, y_all, epochs=local_epochs, verbose=False,
                  validation_data=val_data, callbacks=cbs)
        _, acc = model.evaluate(*val_data, verbose=0)
        _log_round(rnd, n_rounds, [acc], stats)
    print("  Central done.")
    return stats


# ── Per-algorithm experiment orchestration ────────────────────────────────────

def _make_exp_dir(results_dir, dataset, algo, hetero):
    return os.path.join(results_dir, dataset, algo, hetero)


def save_metadata(exp_dir, setting, cfg, algo, hetero, stats):
    """Save experiment metadata to a JSON file."""
    metadata = {
        "algorithm": algo,
        "dataset": {
            "private": cfg["data"]["dataset"],
            "public": cfg["data"]["public_dataset"],
            "n_classes": cfg["data"]["n_classes"],
            "n_parties": cfg["data"]["n_parties"],
        },
        "setting": setting,
        "heterogeneity": hetero,
        "model_sharing": algo in ("fedavg", "fedprox", "central"),
        "knowledge_distillation": algo in ("fedmd", "fedakd", "mks"),
        "training": {
            "n_rounds": cfg["federated"]["n_rounds"],
            "local_epochs": cfg["federated"]["local_epochs"],
            "kd_epochs": cfg["federated"]["kd_epochs"],
            "fedprox_mu": cfg["federated"]["fedprox_mu"],
        },
        "data_config": {
            "n_samples_per_class": cfg["data"]["n_samples_per_class"],
            "dirichlet_alpha": cfg["data"]["dirichlet_alpha"],
            "window_size": cfg["data"].get("window_size", None),
            "n_stft_bins": cfg["data"].get("n_stft_bins", None),
        },
        "results": {
            "last_round_avg_accuracy": float(stats["avg"][-1]) if stats["avg"] else None,
            "last_round_min_accuracy": float(stats["min"][-1]) if stats["min"] else None,
            "last_round_max_accuracy": float(stats["max"][-1]) if stats["max"] else None,
            "final_avg_accuracy": float(np.mean(stats["avg"][-5:])) if len(stats["avg"]) >= 5 else None,
        },
    }
    
    metadata_path = os.path.join(exp_dir, setting, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {metadata_path}")


def run_one_algo(algo, cfg, tiers, tier_map,
                 partitions, pub_data, val_data, input_shape):
    """Run a single algorithm on both settings (iid + noniid).

    Returns {setting: stats_dict}.
    """
    dc = cfg["data"]
    fc = cfg["federated"]
    mc = cfg["models"]
    n_cls  = dc["n_classes"]
    n_par  = dc["n_parties"]
    n_rnd  = fc["n_rounds"]
    l_ep   = fc["local_epochs"]
    kd_ep  = fc["kd_epochs"]
    mu     = fc["fedprox_mu"]
    rdir   = cfg["experiment"]["results_dir"]
    hetero = mc["heterogeneity"]

    pri_x_iid, pri_y_iid, pri_x_nid, pri_y_nid = partitions
    all_stats = {}

    for setting in dc["settings"]:
        pri_x = pri_x_iid if setting == "iid" else pri_x_nid
        pri_y = pri_y_iid if setting == "iid" else pri_y_nid
        exp_dir = _make_exp_dir(rdir, dc["dataset"], algo, hetero)
        os.makedirs(os.path.join(exp_dir, setting), exist_ok=True)

        if algo == "local":
            # Use medium-tier models for local baseline
            models = [build_tiered_model(t, input_shape, n_cls) for t in tiers]
            nodes  = [Node(models[i], (pri_x[i], pri_y[i]), pub_data, val_data)
                      for i in range(n_par)]
            all_stats[setting] = run_local(nodes, n_rnd, l_ep, exp_dir, setting)
            save_metadata(exp_dir, setting, cfg, algo, hetero, all_stats[setting])

        elif algo == "central":
            all_stats[setting] = run_central(
                pri_x, pri_y, val_data, n_rnd, l_ep,
                input_shape, n_cls, exp_dir, setting)
            save_metadata(exp_dir, setting, cfg, algo, hetero, all_stats[setting])

        elif algo in ("fedavg", "fedprox"):
            models  = build_client_models(algo, tiers, input_shape, n_cls)
            pub_cp  = (pub_data[0].copy(), pub_data[1].copy())
            nodes   = [Node(models[i], (pri_x[i], pri_y[i]), pub_cp, val_data)
                       for i in range(n_par)]
            clients = [WeightClient(str(i), nodes[i], exp_dir, setting,
                                    local_epochs=l_ep,
                                    fedprox=(algo == "fedprox"), mu=mu)
                       for i in range(n_par)]
            all_stats[setting] = run_weight_sharing(
                clients, n_rnd, f"{algo.upper()} — {setting.upper()}")
            save_metadata(exp_dir, setting, cfg, algo, hetero, all_stats[setting])

        elif algo in ("fedmd", "fedakd"):
            use_mixup = (algo == "fedakd")
            models    = build_client_models(algo, tiers, input_shape, n_cls)
            pub_cp    = (pub_data[0].copy(), pub_data[1].copy())
            nodes     = [Node(models[i], (pri_x[i], pri_y[i]), pub_cp, val_data,
                              use_mixup=use_mixup)
                         for i in range(n_par)]
            clients   = [KDClient(str(i), nodes[i], exp_dir, setting,
                                  kd_epochs=kd_ep, local_epochs=l_ep,
                                  tier=tiers[i])
                         for i in range(n_par)]
            all_stats[setting] = run_kd(
                clients, n_rnd, f"{algo.upper()} — {setting.upper()}")
            save_metadata(exp_dir, setting, cfg, algo, hetero, all_stats[setting])

        elif algo == "mks":
            models  = build_client_models(algo, tiers, input_shape, n_cls)
            pub_cp  = (pub_data[0].copy(), pub_data[1].copy())
            nodes   = [Node(models[i], (pri_x[i], pri_y[i]), pub_cp, val_data,
                            use_mixup=True)
                       for i in range(n_par)]
            clients = [KDClient(str(i), nodes[i], exp_dir, setting,
                                kd_epochs=kd_ep, local_epochs=l_ep,
                                tier=tiers[i])
                       for i in range(n_par)]
            all_stats[setting] = run_mks(
                clients, n_rnd, f"MKS — {setting.upper()}", tier_map)
            save_metadata(exp_dir, setting, cfg, algo, hetero, all_stats[setting])

        else:
            raise ValueError(f"Unknown algorithm: {algo!r}")

    return all_stats


# ── Multi-seed runner ──────────────────────────────────────────────────────────

def run_multiseed(algo, cfg, seeds):
    """Run algo over multiple seeds, return mean ± std of final accuracy."""
    dc = cfg["data"]
    fc = cfg["federated"]
    mc = cfg["models"]
    input_shape, _ = _get_input_shape(dc)
    hetero = mc["heterogeneity"]
    dist   = mc.get("distributions", {}).get(hetero)
    tiers  = assign_client_tiers(dc["n_parties"], hetero, custom_dist=dist)
    tier_map = {i: t for i, t in enumerate(tiers)}

    seed_results = []
    for seed in seeds:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(f"\n[seed={seed}]  {algo.upper()}  dataset={dc['dataset']}  hetero={hetero}")

        x_tr, y_tr, x_te, y_te_cat, pub_x, pub_y = get_dataset(dc)
        partitions = make_partitions(x_tr, y_tr, dc, fc)
        n_align    = fc["n_alignment"]
        pub_data   = (pub_x[:n_align], pub_y[:n_align])
        val_data   = (x_te, y_te_cat)

        stats = run_one_algo(algo, cfg, tiers, tier_map,
                             partitions, pub_data, val_data, input_shape)
        seed_results.append(stats)

    # Aggregate: mean ± std of last-5-round average per setting
    agg = {}
    for setting in dc["settings"]:
        finals = [float(np.mean(r[setting]["avg"][-5:]))
                  for r in seed_results if setting in r]
        if finals:
            agg[setting] = {"mean": float(np.mean(finals)),
                            "std":  float(np.std(finals)),
                            "raw":  finals}
    return agg


# ── Plotting helpers ───────────────────────────────────────────────────────────

def collect_per_client(results_dir, dataset, algo, hetero, settings, n_parties):
    per = {}
    for s in settings:
        accs = []
        for i in range(n_parties):
            fp = os.path.join(results_dir, dataset, algo, hetero, s, f"train_{i}.csv")
            if os.path.exists(fp):
                try:
                    accs.append(float(pd.read_csv(fp)["val_accuracy"].iloc[-1]))
                except Exception:
                    accs.append(0.0)
        per[s] = accs
    return per


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Federated learning experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", help="YAML config file (dataset-specific)")
    p.add_argument("--algorithm", nargs="+", metavar="ALGO",
                   help="Algorithm(s) to run (default: all from config)")
    p.add_argument("--dataset", help="Dataset override")
    p.add_argument("--setting", choices=["iid", "noniid", "both"], default="both",
                   help="Data partition setting")
    p.add_argument("--heterogeneity", choices=["all_small", "uniform", "skewed"],
                   help="Model heterogeneity distribution override")
    p.add_argument("--seed", type=int, help="Single seed (skips multi-seed CI)")
    p.add_argument("--seeds", nargs="+", type=int,
                   help="Explicit seed list override")
    p.add_argument("--override", nargs="+", dest="overrides", metavar="KEY=VAL",
                   help="Inline config overrides, e.g. federated.n_rounds=30")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config, args.overrides)

    # CLI arg overrides
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    if args.heterogeneity:
        cfg["models"]["heterogeneity"] = args.heterogeneity
    if args.setting != "both":
        cfg["data"]["settings"] = [args.setting]

    algos = args.algorithm or cfg["algorithms"]
    seeds = ([args.seed] if args.seed else
             args.seeds  if args.seeds else
             cfg["experiment"]["seeds"])

    dc     = cfg["data"]
    mc     = cfg["models"]
    rdir   = cfg["experiment"]["results_dir"]
    fig_dir = cfg["experiment"]["fig_dir"]
    hetero = mc["heterogeneity"]
    dist   = mc.get("distributions", {}).get(hetero)
    tiers  = assign_client_tiers(dc["n_parties"], hetero, custom_dist=dist)
    tier_map = {i: t for i, t in enumerate(tiers)}
    input_shape, _ = _get_input_shape(dc)

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(cfg["experiment"]["log_dir"], exist_ok=True)

    # Tier distribution figure
    plot_tier_distribution(tiers,
        os.path.join(fig_dir, f"tiers_{dc['dataset']}_{hetero}.png"))

    sep = "=" * 65
    print(sep)
    print(f"  Dataset    : {dc['dataset']}  (public: {dc['public_dataset']})")
    print(f"  Algorithms : {algos}")
    print(f"  Heterogen. : {hetero}  ->  {tiers[:6]}...")
    print(f"  Seeds      : {seeds}")
    print(f"  Settings   : {dc['settings']}")
    print(sep)

    # ── Load data once for single-seed runs, or per-seed inside multiseed ──
    if len(seeds) == 1:
        np.random.seed(seeds[0])
        tf.random.set_seed(seeds[0])
        x_tr, y_tr, x_te, y_te_cat, pub_x, pub_y = get_dataset(dc)
        partitions = make_partitions(x_tr, y_tr, dc, cfg["federated"])
        n_align    = cfg["federated"]["n_alignment"]
        pub_data   = (pub_x[:n_align], pub_y[:n_align])
        val_data   = (x_te, y_te_cat)

        plot_data_partitioning(
            partitions[1], partitions[3], ["c" + str(i) for i in range(dc["n_classes"])],
            fig_dir)

        all_round_stats = {}
        for algo in algos:
            all_round_stats[algo] = run_one_algo(
                algo, cfg, tiers, tier_map,
                partitions, pub_data, val_data, input_shape)

        # Training-curve figure (first seed only)
        flat = {a: s for a, s in all_round_stats.items()}
        iid_stats  = {a: flat[a].get("iid",    {}) for a in flat}
        nid_stats  = {a: flat[a].get("noniid", {}) for a in flat}
        plot_training_curves(
            {a: {"iid": iid_stats[a], "noniid": nid_stats[a]} for a in flat},
            os.path.join(fig_dir, f"curves_{dc['dataset']}_{hetero}.png"))

        # Final accuracy bar chart
        final_accs = {
            a: {s: float(np.mean(all_round_stats[a][s]["avg"][-5:]))
                for s in dc["settings"] if s in all_round_stats[a]}
            for a in algos
        }
        plot_final_comparison(
            final_accs,
            os.path.join(fig_dir, f"final_{dc['dataset']}_{hetero}.png"))

    else:
        # Multi-seed: collect CI stats per algorithm
        ci_stats = {}
        for algo in algos:
            ci_stats[algo] = run_multiseed(algo, cfg, seeds)

        plot_final_comparison_ci(
            ci_stats,
            os.path.join(fig_dir, f"final_ci_{dc['dataset']}_{hetero}.png"))

    # Per-client accuracy (uses CSV logs regardless of seeds)
    per_client = {}
    for algo in algos:
        per_client[algo] = collect_per_client(
            rdir, dc["dataset"], algo, hetero, dc["settings"], dc["n_parties"])
    plot_per_client_accuracy(
        per_client, algos,
        os.path.join(fig_dir, f"per_client_{dc['dataset']}_{hetero}.png"))

    print(f"\n{sep}")
    print("  All experiments complete.")
    print(f"  Results  ->  {os.path.abspath(rdir)}")
    print(f"  Figures  ->  {os.path.abspath(fig_dir)}")
    print(sep)


if __name__ == "__main__":
    main()
