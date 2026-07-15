#!/usr/bin/env python3
"""
dl.py — Centralised Transformer baseline for HomeOccupancy CSI classification.

Preprocessing is identical to run_fedkd.py (get_dataset path):
  - load_homeoccupancy(window_size, n_stft_bins)
  - MinMaxScaler fitted on training split, applied to test split
  - window_size=1500, n_stft_bins=8  (n_features = 4+8 = 12)

Model: pure Transformer encoder
  Conv1D tokeniser (2×, 2× downsample) → learned positional encoding
  → N × (MultiHeadAttention + LayerNorm + FFN + LayerNorm)
  → GlobalAveragePooling1D → Dense(n_classes, softmax)

Train for 100 epochs (with early-stopping patience=20 and ReduceLR).
Saves per-epoch CSV log, training-curve PNG, and a one-row summary CSV.

Usage:
    python dl.py
    python dl.py --epochs 100 --seed 42 --results-dir results/dl
    python dl.py --window-size 1500 --n-stft-bins 8 --n-classes 3
"""

import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from utils import load_homeoccupancy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Centralised Transformer baseline — HomeOccupancy CSI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--window-size",  type=int,   default=1500,
                   help="CSI window length (must match FL config; default 1500)")
    p.add_argument("--n-stft-bins",  type=int,   default=8,
                   help="STFT bins retained (n_features = 4 + n_stft_bins)")
    p.add_argument("--n-classes",    type=int,   default=3,
                   help="Number of output classes")
    p.add_argument("--epochs",       type=int,   default=100,
                   help="Maximum training epochs (default 100)")
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--n-blocks",     type=int,   default=3,
                   help="Number of Transformer encoder blocks")
    p.add_argument("--n-heads",      type=int,   default=4,
                   help="Number of attention heads per block")
    p.add_argument("--key-dim",      type=int,   default=64,
                   help="Key/query dimension per head")
    p.add_argument("--ff-dim",       type=int,   default=256,
                   help="Feed-forward hidden dimension (also d_model)")
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--results-dir",  type=str,   default="results/dl")
    return p.parse_args()


# ── Preprocessing — identical to get_dataset() in run_fedkd.py ────────────────

def load_and_preprocess(window_size: int, n_stft_bins: int, n_classes: int):
    """Load HomeOccupancy and scale exactly as run_fedkd.py does.

    Uses the training-split MinMaxScaler on test data (same scaler, not a new fit).
    Returns float32 arrays ready for model.fit().
    """
    nf = 4 + n_stft_bins
    x_tr, y_tr, x_te, y_te = load_homeoccupancy(window_size, n_stft_bins)

    sc = MinMaxScaler()
    x_tr = sc.fit_transform(x_tr.reshape(-1, nf)).reshape(x_tr.shape)
    x_te = sc.transform(x_te.reshape(-1, nf)).reshape(x_te.shape)

    y_tr_cat = tf.keras.utils.to_categorical(y_tr, num_classes=n_classes)
    y_te_cat  = tf.keras.utils.to_categorical(y_te, num_classes=n_classes)

    print(f"  Train : {x_tr.shape}  ({len(x_tr)} windows)")
    print(f"  Test  : {x_te.shape}  ({len(x_te)} windows)")
    return x_tr, y_tr_cat, x_te, y_te_cat


# ── Learned positional encoding ───────────────────────────────────────────────

class LearnedPositionalEncoding(tf.keras.layers.Layer):
    """Additive learned positional encoding (shape: 1 × seq_len × d_model).

    The weight is initialised to zero so the model starts with no positional
    bias and learns it from data.
    """

    def build(self, input_shape):
        _, seq_len, d_model = input_shape
        self.pe = self.add_weight(
            name="pe",
            shape=(1, seq_len, d_model),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        return x + self.pe


# ── Transformer model ─────────────────────────────────────────────────────────

def build_transformer(
    input_shape: tuple,
    n_classes:   int,
    n_blocks:    int   = 3,
    n_heads:     int   = 4,
    key_dim:     int   = 64,
    ff_dim:      int   = 256,
    dropout:     float = 0.3,
) -> tf.keras.Model:
    """Pure Transformer encoder for CSI time-series classification.

    Architecture
    ------------
    Input (window_size, n_features)
      → Conv1D(64, 5) + MaxPool(2)     # T → T/2
      → Conv1D(128, 3) + MaxPool(2)    # T/2 → T/4
      → Dense(ff_dim)                  # project to d_model = ff_dim
      → LearnedPositionalEncoding
      → Dropout
      → n_blocks × TransformerEncoderBlock:
            MultiHeadAttention → Add & LayerNorm
            FFN (ff_dim*2 → ff_dim) → Dropout → Add & LayerNorm
      → GlobalAveragePooling1D
      → Dropout
      → Dense(n_classes, softmax)
    """
    d_model = ff_dim

    inp = tf.keras.layers.Input(shape=input_shape, name="input")

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    x = tf.keras.layers.Conv1D(
        64, 5, activation="relu", padding="same", name="tok_conv1")(inp)
    x = tf.keras.layers.MaxPooling1D(2, name="tok_pool1")(x)
    x = tf.keras.layers.Conv1D(
        128, 3, activation="relu", padding="same", name="tok_conv2")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="tok_pool2")(x)
    x = tf.keras.layers.Dense(d_model, name="tok_proj")(x)   # (B, T/4, d_model)

    # ── Positional encoding ───────────────────────────────────────────────────
    x = LearnedPositionalEncoding(name="pos_enc")(x)
    x = tf.keras.layers.Dropout(dropout, name="pos_drop")(x)

    # ── Transformer encoder blocks ────────────────────────────────────────────
    for i in range(n_blocks):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=key_dim,
            dropout=dropout, name=f"mha_{i}")(x, x)
        x = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"ln1_{i}")(x + attn)

        ff = tf.keras.layers.Dense(
            d_model * 2, activation="gelu", name=f"ff1_{i}")(x)
        ff = tf.keras.layers.Dense(d_model, name=f"ff2_{i}")(ff)
        ff = tf.keras.layers.Dropout(dropout, name=f"ff_drop_{i}")(ff)
        x  = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"ln2_{i}")(x + ff)

    # ── Classification head ───────────────────────────────────────────────────
    x   = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x   = tf.keras.layers.Dropout(dropout, name="head_drop")(x)
    out = tf.keras.layers.Dense(
        n_classes, activation="softmax", name="clf_out")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="TransformerCSI")
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading and preprocessing data ...")
    x_tr, y_tr, x_te, y_te = load_and_preprocess(
        args.window_size, args.n_stft_bins, args.n_classes)

    input_shape = (args.window_size, 4 + args.n_stft_bins)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_transformer(
        input_shape, args.n_classes,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        key_dim=args.key_dim,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    print(f"  Trainable params: {model.count_params():,}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.results_dir, "dl_training.csv")
    callbacks = [
        tf.keras.callbacks.CSVLogger(csv_path),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10,
            min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20,
            restore_best_weights=True, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  Training TransformerCSI — max {args.epochs} epochs")
    print(f"  Early-stop patience=20 on val_accuracy")
    print(sep)

    model.fit(
        x_tr, y_tr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_te, y_te),
        callbacks=callbacks,
        verbose=2,
    )

    # ── Final evaluation ──────────────────────────────────────────────────────
    loss, acc = model.evaluate(x_te, y_te, batch_size=args.batch_size, verbose=0)
    print(f"\n{sep}")
    print(f"  Final test accuracy : {acc:.4f}")
    print(f"  Final test loss     : {loss:.4f}")
    print(sep)

    # ── Plot ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df["epoch"] + 1, df["accuracy"],
                 label="Train", lw=2, color="#4c8fbd")
    axes[0].plot(df["epoch"] + 1, df["val_accuracy"],
                 label="Val/Test", lw=2, ls="--", color="#e07b54")
    axes[0].axhline(acc, color="green", ls=":", lw=1.5,
                    label=f"Final test {acc:.4f}")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"] + 1, df["loss"],
                 label="Train", lw=2, color="#4c8fbd")
    axes[1].plot(df["epoch"] + 1, df["val_loss"],
                 label="Val/Test", lw=2, ls="--", color="#e07b54")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Centralised Transformer — HomeOccupancy CSI\n"
        f"n_blocks={args.n_blocks}, n_heads={args.n_heads}, "
        f"ff_dim={args.ff_dim}, dropout={args.dropout}\n"
        f"Final test accuracy: {acc:.4f}  (seed={args.seed})",
        fontsize=11,
    )
    plt.tight_layout()
    fig_path = os.path.join(args.results_dir, "dl_training_curves.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_path}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary = pd.DataFrame([{
        "test_accuracy":  round(acc,  6),
        "test_loss":      round(loss, 6),
        "best_val_acc":   round(float(df["val_accuracy"].max()), 6),
        "best_val_epoch": int(df["val_accuracy"].idxmax()) + 1,
        "n_epochs_run":   len(df),
        "seed":           args.seed,
        "window_size":    args.window_size,
        "n_stft_bins":    args.n_stft_bins,
        "n_classes":      args.n_classes,
        "n_blocks":       args.n_blocks,
        "n_heads":        args.n_heads,
        "key_dim":        args.key_dim,
        "ff_dim":         args.ff_dim,
        "dropout":        args.dropout,
        "lr":             args.lr,
        "batch_size":     args.batch_size,
        "n_params":       model.count_params(),
    }])
    summary_path = os.path.join(args.results_dir, "dl_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Saved {summary_path}")
    print(f"\n  Results dir: {os.path.abspath(args.results_dir)}")


if __name__ == "__main__":
    train(parse_args())
