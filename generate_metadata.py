#!/usr/bin/env python3
"""
generate_metadata.py — Generate metadata.json files for existing experiment results.

This script walks through the results directory and creates metadata files
without rerunning experiments.
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path


# Default configuration (same as run_fedkd.py)
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
}


def load_config_for_dataset(dataset):
    """Load config file for a specific dataset."""
    config_file = f"config/{dataset}.yaml"
    if os.path.exists(config_file):
        with open(config_file) as f:
            file_cfg = yaml.safe_load(f)
        # Merge with defaults
        cfg = DEFAULT_CFG.copy()
        cfg["data"].update(file_cfg.get("data", {}))
        cfg["federated"].update(file_cfg.get("federated", {}))
        return cfg
    else:
        # Return default with dataset name updated
        cfg = DEFAULT_CFG.copy()
        cfg["data"]["dataset"] = dataset
        return cfg


def extract_accuracy_from_csv(setting_dir, n_parties=20):
    """Extract accuracy statistics from CSV files in a setting directory."""
    accuracies = []
    
    for i in range(n_parties):
        csv_file = os.path.join(setting_dir, f"train_{i}.csv")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                if "val_accuracy" in df.columns:
                    last_acc = float(df["val_accuracy"].iloc[-1])
                    accuracies.append(last_acc)
            except Exception as e:
                print(f"    Warning: Could not read {csv_file}: {e}")
    
    if accuracies:
        return {
            "avg": float(np.mean(accuracies)),
            "min": float(np.min(accuracies)),
            "max": float(np.max(accuracies)),
            "std": float(np.std(accuracies)),
            "n_clients": len(accuracies)
        }
    else:
        return {
            "avg": None,
            "min": None,
            "max": None,
            "std": None,
            "n_clients": 0
        }


def generate_metadata_for_run(results_dir, dataset, algo, hetero, setting, cfg):
    """Generate metadata for a single experiment run."""
    setting_dir = os.path.join(results_dir, dataset, algo, hetero, setting)
    
    if not os.path.exists(setting_dir):
        print(f"    Skipping {setting_dir} - does not exist")
        return False
    
    # Check if metadata already exists
    metadata_file = os.path.join(setting_dir, "metadata.json")
    if os.path.exists(metadata_file):
        print(f"    Metadata already exists for {dataset}/{algo}/{hetero}/{setting}")
        return True
    
    # Extract accuracy from CSV files
    acc_stats = extract_accuracy_from_csv(setting_dir, cfg["data"]["n_parties"])
    
    # Build metadata
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
            "final_avg_accuracy": acc_stats["avg"],
            "final_min_accuracy": acc_stats["min"],
            "final_max_accuracy": acc_stats["max"],
            "final_std_accuracy": acc_stats["std"],
            "n_clients": acc_stats["n_clients"],
        },
    }
    
    # Save metadata
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    ✓ Generated metadata for {dataset}/{algo}/{hetero}/{setting}")
    return True


def main():
    """Generate metadata for all phase1 results."""
    results_dir = "results/phase1"
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory {results_dir} does not exist")
        return
    
    # Datasets, algorithms, and heterogeneity settings from run_experiments.sh
    datasets = ["home_occupancy", "home_har", "mnist", "cifar10"]  # All datasets
    algorithms = ["fedmd", "fedakd", "mks", "fedavg", "fedprox", "local", "central"]
    heterogeneity = "uniform"  # Phase 1 uses uniform heterogeneity
    settings = ["iid", "noniid"]
    
    print(f"Generating metadata for phase1 results in {results_dir}")
    print("=" * 65)
    
    total_generated = 0
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        cfg = load_config_for_dataset(dataset)
        
        for algo in algorithms:
            for hetero in [heterogeneity]:
                for setting in settings:
                    if generate_metadata_for_run(results_dir, dataset, algo, hetero, setting, cfg):
                        total_generated += 1
    
    print("\n" + "=" * 65)
    print(f"Metadata generation complete. Total files generated: {total_generated}")
    print("=" * 65)


if __name__ == "__main__":
    main()
