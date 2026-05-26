"""
utils.py — Shared utilities for WS_FedAKD.ipynb and run_fedkd.py
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, MaxPooling1D, Conv1D, Flatten
from tensorflow.keras.callbacks import CSVLogger
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CLASS_NAMES = ["empty", "sleep", "work"]
COLORS = plt.cm.tab10.colors


# ── CSI PREPROCESSING ─────────────────────────────────────────────

def parse_csi_data(data_str):
    if not isinstance(data_str, str):
        return None
    try:
        vals = list(map(float, data_str.strip().strip('[]"').split(",")))
        if len(vals) < 128:
            return None
        c = np.array([vals[i] + 1j * vals[i+1] for i in range(0, min(128, len(vals)), 2)])
        return np.concatenate([c[6:32], c[33:59]])
    except (ValueError, IndexError):
        return None


def process(raw_content, label, window_length, n_stft_bins=8):
    """Process raw CSV file content into windowed feature arrays.

    Steps:
        1. Parse CSI rows → complex subcarrier amplitudes  (T, 52)
        2. Resample to 150 Hz for consistent temporal density
        3. Segment into non-overlapping windows of *window_length* steps
        4. Per window, compute per-timestep features:
              mean over 52 subcarriers        → (window_length, 1)
              rolling variance  scale=15      → (window_length, 1)
              rolling variance  scale=150     → (window_length, 1)
              rolling variance  scale=1500    → (window_length, 1)
              STFT magnitude bins             → (window_length, n_stft_bins)
        5. Concatenate → (window_length, 4 + n_stft_bins)

    Args:
        raw_content  : string, full CSV file content (from open(...).read())
        label        : int, class label applied to every window in this file
        window_length: int, from config
        n_stft_bins  : int, STFT frequency bins to retain (excluding DC)

    Returns:
        X : np.ndarray  (n_windows, window_length, 4 + n_stft_bins)  float32
        y : np.ndarray  (n_windows,)  int32
    """
    import io
    from scipy.signal import stft as _stft

    n_out = 4 + n_stft_bins
    empty_X = np.empty((0, window_length, n_out), dtype=np.float32)
    empty_y = np.array([], dtype=np.int32)

    try:
        df = pd.read_csv(io.StringIO(raw_content), on_bad_lines="skip")
    except Exception:
        return empty_X, empty_y

    if "data" not in df.columns:
        return empty_X, empty_y

    parsed = [r for d in df["data"] if (r := parse_csi_data(d)) is not None]
    if not parsed:
        return empty_X, empty_y

    amps = np.abs(np.array(parsed, dtype=np.float32))          # (T, 52)
    target_len = max(int(len(amps) * 150 / 100), window_length)
    amps = resample(amps, target_len, axis=0).astype(np.float32)

    n_windows = len(amps) // window_length
    if n_windows == 0:
        return empty_X, empty_y

    windows = amps[:n_windows * window_length].reshape(n_windows, window_length, 52)

    nperseg = min(16, window_length)
    actual_bins = nperseg // 2        # non-DC bins available from STFT

    samples = []
    for w in windows:
        mean_sig = w.mean(axis=1)     # (window_length,) — mean over subcarriers

        # Rolling population variance at three temporal scales (pandas is vectorised)
        s = pd.Series(mean_sig)
        var15   = s.rolling(15,   min_periods=1).var(ddof=0).to_numpy()
        var150  = s.rolling(150,  min_periods=1).var(ddof=0).to_numpy()
        var1500 = s.rolling(1500, min_periods=1).var(ddof=0).to_numpy()

        # STFT with 1-sample hop; boundary zeros give ≈ window_length time frames
        _, _, Zxx = _stft(mean_sig, nperseg=nperseg, noverlap=nperseg - 1,
                          boundary="zeros", padded=False)
        stft_mag = np.abs(Zxx[1:, :])                          # drop DC → (actual_bins, n_frames)
        n_frames = stft_mag.shape[1]

        # Pad to n_stft_bins if fewer bins than requested
        if actual_bins < n_stft_bins:
            pad = np.zeros((n_stft_bins - actual_bins, n_frames), dtype=np.float32)
            stft_mag = np.concatenate([stft_mag, pad], axis=0)
        else:
            stft_mag = stft_mag[:n_stft_bins, :]               # (n_stft_bins, n_frames)

        # Interpolate time axis to exactly window_length
        if n_frames != window_length:
            t_src = np.linspace(0, 1, n_frames)
            t_dst = np.linspace(0, 1, window_length)
            stft_mag = np.stack([np.interp(t_dst, t_src, stft_mag[b])
                                 for b in range(n_stft_bins)])  # (n_stft_bins, window_length)
        stft_feats = stft_mag.T                                 # (window_length, n_stft_bins)

        feat = np.concatenate([
            mean_sig[:, None],
            var15[:, None],
            var150[:, None],
            var1500[:, None],
            stft_feats,
        ], axis=1).astype(np.float32)                          # (window_length, 4+n_stft_bins)
        samples.append(feat)

    X = np.stack(samples)                                       # (n_windows, window_length, n_out)
    y = np.full(n_windows, label, dtype=np.int32)
    return X, y


def _hf(repo_id, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")


def load_homeoccupancy(window_size=100, n_stft_bins=8):
    print("Loading HomeOccupancy...")
    labels = {"empty": 0, "sleep": 1, "work": 2}
    n_out = 4 + n_stft_bins
    xs_tr, ys_tr, xs_te, ys_te = [], [], [], []
    for name, lid in labels.items():
        for sess in [1, 2]:
            try:
                raw = open(_hf("gadgadgad/HomeOccupancy", f"{name}_{sess}.csv")).read()
                X, y = process(raw, lid, window_size, n_stft_bins)
                if len(X): xs_tr.append(X); ys_tr.append(y); print(f"  {name}_{sess}: {len(X)}")
            except Exception as e: print(f"  skip {name}_{sess}: {e}")
        try:
            raw = open(_hf("gadgadgad/HomeOccupancy", f"{name}_3.csv")).read()
            X, y = process(raw, lid, window_size, n_stft_bins)
            if len(X): xs_te.append(X); ys_te.append(y); print(f"  {name}_3 test: {len(X)}")
        except Exception as e: print(f"  skip {name}_3: {e}")
    e0 = np.empty((0, window_size, n_out), dtype=np.float32)
    x_tr = np.concatenate(xs_tr) if xs_tr else e0
    y_tr = np.concatenate(ys_tr) if ys_tr else np.array([], dtype=np.int32)
    x_te = np.concatenate(xs_te) if xs_te else e0
    y_te = np.concatenate(ys_te) if ys_te else np.array([], dtype=np.int32)
    print(f"  -> {len(x_tr)} train, {len(x_te)} test")
    return x_tr, y_tr, x_te, y_te


def load_homehar(window_size=100, n_stft_bins=8):
    print("Loading HomeHAR...")
    labels = {"drink": 0, "eat": 1, "empty": 2, "sleep": 3, "smoke": 4, "watch": 5, "work": 6}
    n_out = 4 + n_stft_bins
    xs, ys = [], []
    for sess in ["data1", "data2"]:
        for name, lid in labels.items():
            for fn in [f"{sess}/{name}.csv", f"{sess}_{name}.csv", f"{name}_{sess}.csv"]:
                try:
                    raw = open(_hf("gadgadgad/HomeHAR", fn)).read()
                    X, y = process(raw, lid, window_size, n_stft_bins)
                    if len(X): xs.append(X); ys.append(y); print(f"  {fn}: {len(X)}")
                    break
                except Exception: continue
    e0 = np.empty((0, window_size, n_out), dtype=np.float32)
    pub_x = np.concatenate(xs) if xs else e0
    pub_y = np.concatenate(ys) if ys else np.array([], dtype=np.int32)
    print(f"  -> {len(pub_x)} public")
    return pub_x, pub_y


def prepare_datasets(window_size=100, n_stft_bins=8, n_classes=3):
    x_tr, y_tr, x_te, y_te = load_homeoccupancy(window_size, n_stft_bins)
    pub_x, pub_y = load_homehar(window_size, n_stft_bins)
    n_features = 4 + n_stft_bins
    scaler = MinMaxScaler()
    x_tr = scaler.fit_transform(x_tr.reshape(-1, n_features)).reshape(x_tr.shape)
    x_te = scaler.transform(x_te.reshape(-1, n_features)).reshape(x_te.shape)
    pub_x = scaler.transform(pub_x.reshape(-1, n_features)).reshape(pub_x.shape)
    y_te_cat = tf.keras.utils.to_categorical(y_te, num_classes=n_classes)
    idx = np.random.permutation(len(x_tr))
    return x_tr[idx], y_tr[idx], x_te, y_te_cat, pub_x, pub_y


def split_dataset(x, y, samples_per_class, n_models, include_classes,
                  to_categorical_flag=False, n_classes_total=None):
    n_cls = n_classes_total or len(np.unique(y))
    datasets, labels = [None]*n_models, [None]*n_models
    combined_idx = np.array([], dtype=np.int32)
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        n_needed = samples_per_class * n_models
        idx = np.random.choice(idx, max(n_needed, len(idx)), replace=True)[:n_needed]
        combined_idx = np.r_[combined_idx, idx]
        for i in range(n_models):
            if include_classes != "all" and label not in include_classes[i]:
                continue
            cx = x[idx[i*samples_per_class:(i+1)*samples_per_class]]
            cy = y[idx[i*samples_per_class:(i+1)*samples_per_class]]
            if datasets[i] is None: datasets[i], labels[i] = [cx], [cy]
            else: datasets[i].append(cx); labels[i].append(cy)
    for i in range(n_models):
        if datasets[i]:
            datasets[i] = np.concatenate(datasets[i]); labels[i] = np.concatenate(labels[i])
        else:
            datasets[i] = np.empty((0,)+x.shape[1:]); labels[i] = np.array([])
    if to_categorical_flag:
        labels = [tf.keras.utils.to_categorical(l, n_cls) if len(l) else l for l in labels]
    return datasets, labels, x[combined_idx], y[combined_idx]


# ── MODEL CONSTRUCTION ────────────────────────────────────────────

def _tiered_backbone_1d(inp, tier):
    """1-D backbone for CSI time-series.  Returns the pre-head feature tensor.

    Tier hierarchy (weakest → strongest):
        small  — Conv1D stack          (baseline IoT device)
        medium — Conv1D + BiLSTM       (mid-range device)
        large  — Conv1D + BiLSTM + Transformer encoder  (server / rich device)
    """
    if tier == "small":
        x = Conv1D(128, 3, activation="relu", padding="same")(inp)
        x = MaxPooling1D(2)(x)
        x = Conv1D(256, 3, activation="relu", padding="same")(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(256, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)  # avoids huge Flatten→Dense weight matrix
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.30)(x)
    elif tier == "medium":
        x = Conv1D(64, 5, activation="relu", padding="same")(inp)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation="relu", padding="same")(x)
        x = MaxPooling1D(2)(x)
        x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=False))(x)
        x = Dropout(0.30)(x)
    else:  # large
        x = Conv1D(64, 5, activation="relu", padding="same")(inp)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation="relu", padding="same")(x)
        x = MaxPooling1D(2)(x)
        x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=True))(x)
        # Transformer encoder block
        attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn)
        ffn = Dense(256, activation="relu")(x)
        ffn = Dense(256)(ffn)
        x = tf.keras.layers.LayerNormalization()(x + ffn)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dropout(0.30)(x)
    return x


def _tiered_backbone_2d(inp, tier):
    """2-D backbone for image datasets."""
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    if tier == "small":
        x = Conv2D(32, 3, activation="relu", padding="same")(inp)
        x = MaxPooling2D(2)(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.25)(x)
    elif tier == "medium":
        x = Conv2D(64, 3, activation="relu", padding="same")(inp)
        x = MaxPooling2D(2)(x)
        x = Conv2D(128, 3, activation="relu", padding="same")(x)
        x = MaxPooling2D(2)(x)
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.30)(x)
    else:
        x = Conv2D(64, 3, activation="relu", padding="same")(inp)
        x = MaxPooling2D(2)(x)
        x = Conv2D(128, 3, activation="relu", padding="same")(x)
        x = MaxPooling2D(2)(x)
        x = Conv2D(256, 3, activation="relu", padding="same")(x)
        x = MaxPooling2D(2)(x)
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.40)(x)
    return x


def build_tiered_model(tier, input_shape, n_classes, feature_dim=64):
    """Build a single model with two output heads that share one backbone.

    Head 1 — classification : Dense(n_classes) + softmax  → trained with CE on private data
    Head 2 — distillation   : Dense(feature_dim) + relu   → trained with MSE on public data;
                               its predictions are the KD carrier signal shared across clients.

    Returns (clf_model, feat_model) — two Keras sub-models that share all backbone
    weights.  Training either model updates the shared backbone.

    input_shape:
        (T, F)    → Conv1D / BiLSTM / Transformer  (CSI time-series)
        (H, W, C) → Conv2D                          (images)
    """
    is_image = len(input_shape) == 3
    inp = tf.keras.layers.Input(input_shape)

    backbone = _tiered_backbone_2d(inp, tier) if is_image else _tiered_backbone_1d(inp, tier)

    clf_out  = Dense(n_classes,   activation="softmax", name="clf_out")(backbone)
    feat_out = Dense(feature_dim, activation="relu",    name="feat_out")(backbone)

    clf_model  = tf.keras.Model(inputs=inp, outputs=clf_out,  name=f"{tier}_clf")
    feat_model = tf.keras.Model(inputs=inp, outputs=feat_out, name=f"{tier}_feat")

    clf_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="categorical_crossentropy", metrics=["accuracy"])
    feat_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    return clf_model, feat_model


# ── NODE ──────────────────────────────────────────────────────────

class Node:
    """Federated-learning client node.

    Holds a (clf_model, feat_model) pair that share one backbone.
    Set use_mixup=True to apply FedAKD-style Mixup augmentation on
    public data before producing the KD carrier signal (feat_model output).

    model[0] — classification model  (CE on private data)
    model[1] — distillation model    (MSE on public data; outputs KD carrier)
    """

    def __init__(self, model, local_dataset, shared_public,
                 target_validation_gen, use_mixup=False):
        self.model = model
        self.local_target_dataset = local_dataset
        self.shared_public_dataset = shared_public
        self.target_validation_gen = target_validation_gen
        self.use_mixup = use_mixup
        self.seed, self.alpha = 0, 1.0
        self.target_validation_acc, self.target_validation_loss = [], []

    def _mix(self, x):
        """Mixup: convex combination of x with a randomly shuffled copy.

        Uses in-place ops to keep peak allocation at 2× x instead of 4×.
        """
        np.random.seed(self.seed)
        idx = np.random.permutation(len(x))
        shuffled = x[idx]                   # 1 unavoidable copy (fancy index)
        shuffled *= (1.0 - self.alpha)      # in-place scale
        out = x * self.alpha                # 1 copy
        out += shuffled                     # in-place accumulate
        del shuffled
        return out

    def _public_x(self):
        x = self.shared_public_dataset[0]
        return self._mix(x) if self.use_mixup else x

    def get_training_metadata(self, seed, alpha):
        self.seed, self.alpha = seed, alpha
        return self.get_carrier_scores(), self.evaluate_on_validation_set(save=False)[1]

    def get_carrier_scores(self):
        return self.model[1].predict(self._public_x(), batch_size=32, verbose=0)

    def receive_training_metadata(self, metadata):
        self.shared_public_dataset = (self.shared_public_dataset[0], metadata)

    def evaluate_on_validation_set(self, save=True):
        loss, acc = self.model[0].evaluate(*self.target_validation_gen, batch_size=32, verbose=0)
        if save:
            self.target_validation_acc.append(acc)
            self.target_validation_loss.append(loss)
        return loss, acc

    def train_on_target(self, epochs=1, verbose=False, logger_file=None, evaluate=False):
        cbs = [CSVLogger(logger_file, append=True)] if logger_file else []
        kw = dict(validation_data=self.target_validation_gen) if evaluate else {}
        self.model[0].fit(self.local_target_dataset[0], self.local_target_dataset[1],
                          epochs=epochs, callbacks=cbs, verbose=verbose, **kw)

    def train_on_public(self, epochs=1, verbose=False, logger_file=None):
        cbs = [CSVLogger(logger_file, append=True)] if logger_file else []
        self.model[1].fit(self._public_x(), self.shared_public_dataset[1],
                          epochs=epochs, callbacks=cbs, verbose=verbose)


# ── FEDPROX TRAINING ──────────────────────────────────────────────

def fedprox_train(model, x, y, global_weights, mu=0.01, epochs=1,
                  batch_size=32, validation_data=None, csv_path=None):
    """Local SGD with proximal term: L_ce + (mu/2)||w - w_global||^2."""
    w_ref = [tf.constant(w, dtype=tf.float32) for w in global_weights]
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x)).batch(batch_size)
    rows = []
    for epoch in range(epochs):
        ep_loss, correct, total, nb = 0.0, 0, 0, 0
        for xb, yb in dataset:
            with tf.GradientTape() as tape:
                pred = model(xb, training=True)
                ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(yb, pred))
                prox = tf.add_n([tf.reduce_sum(tf.square(w - r))
                                 for w, r in zip(model.trainable_weights, w_ref)])
                loss = ce + (mu / 2.0) * prox
            model.optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_weights), model.trainable_weights))
            ep_loss += float(loss); nb += 1
            correct += int(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(yb,1)), tf.int32)))
            total += len(yb)
        vl, va = (model.evaluate(*validation_data, batch_size=batch_size, verbose=0)
                  if validation_data else (0.0, 0.0))
        rows.append({"epoch": epoch, "accuracy": correct/max(total,1),
                     "loss": ep_loss/max(nb,1), "val_accuracy": va, "val_loss": vl})
    if csv_path:
        pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    return rows


# ── WEIGHT AGGREGATION ────────────────────────────────────────────

def fedavg_aggregate(weights_list, n_samples_list):
    total = sum(n_samples_list)
    return [sum(w[l]*n/total for w, n in zip(weights_list, n_samples_list))
            for l in range(len(weights_list[0]))]


def aggregate_by_cluster(client_ids, weights_list, n_samples_list):
    """Per-cluster FedAvg; cluster_id = client_id % 10."""
    clusters = {}
    for cid, w, n in zip(client_ids, weights_list, n_samples_list):
        c = cid % 10
        clusters.setdefault(c, {"w": [], "n": []})
        clusters[c]["w"].append(w); clusters[c]["n"].append(n)
    return {c: fedavg_aggregate(d["w"], d["n"]) for c, d in clusters.items()}


def aggregate_soft_labels(soft_labels_list, accuracies):
    return np.average(soft_labels_list, weights=np.array(accuracies, dtype=np.float64), axis=0)


# ── PLOTTING ──────────────────────────────────────────────────────

def plot_data_partitioning(pri_y_iid, pri_y_noniid, class_names, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, pri_y, title in zip(axes, [pri_y_iid, pri_y_noniid], ["IID", "non-IID"]):
        n = len(pri_y)
        counts = np.zeros((n, len(class_names)), dtype=int)
        for i, y in enumerate(pri_y):
            if not len(y): continue
            lbl = np.argmax(y, axis=1) if y.ndim == 2 else y.astype(int)
            for j in range(len(class_names)): counts[i, j] = int(np.sum(lbl == j))
        fracs = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
        bottom = np.zeros(n)
        for j, name in enumerate(class_names):
            ax.bar(np.arange(n), fracs[:, j], bottom=bottom, label=name,
                   color=COLORS[j], edgecolor="white", linewidth=0.5)
            bottom += fracs[:, j]
        ax.set(title=f"Data Partitioning — {title}", xlabel="Client ID",
               ylabel="Class fraction", ylim=(0, 1.05))
        ax.set_xticks(np.arange(n)); ax.set_xticklabels(np.arange(n), fontsize=8)
        ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "data_partitioning.png"), dpi=150)
    plt.close(fig); print("  Saved data_partitioning.png")


def plot_training_curves(all_stats, save_path):
    """all_stats: {algo: {setting: {avg/min/max: [...]}}}"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ax, setting, st in zip(axes, ["iid", "noniid"], ["IID", "non-IID"]):
        for k, (algo, stats) in enumerate(all_stats.items()):
            if setting not in stats: continue
            d = stats[setting]
            rounds = np.arange(1, len(d["avg"])+1)
            col = COLORS[k % len(COLORS)]
            ax.plot(rounds, d["avg"], label=algo, color=col, lw=2)
            ax.fill_between(rounds, d["min"], d["max"], alpha=0.12, color=col)
        ax.set(title=f"Training Accuracy — {st}", xlabel="Round", ylabel="Avg accuracy",
               xlim=(1, None), ylim=(0, 1))
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


def plot_final_comparison(final_accs, save_path):
    """final_accs: {algo: {iid: float, noniid: float}}"""
    algos = list(final_accs.keys())
    iid_v = [final_accs[a]["iid"] for a in algos]
    noniid_v = [final_accs[a]["noniid"] for a in algos]
    x = np.arange(len(algos)); w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, iid_v, w, label="IID",
                color=[COLORS[i%len(COLORS)] for i in range(len(algos))], alpha=0.85)
    b2 = ax.bar(x + w/2, noniid_v, w, label="non-IID",
                color=[COLORS[i%len(COLORS)] for i in range(len(algos))], alpha=0.45, hatch="//")
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set(xticks=x, ylabel="Final avg accuracy (last 5 rounds)",
           title="Algorithm Comparison — Final Accuracy", ylim=(0, 1))
    ax.set_xticklabels(algos, fontsize=11); ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


def plot_per_client_accuracy(per_client_final, algo_names, save_path):
    """per_client_final: {algo: {iid: [acc,...], noniid: [acc,...]}}"""
    n = len(algo_names)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3*n), sharex=True, sharey=True)
    if n == 1: axes = [axes]
    for row, algo in enumerate(algo_names):
        for col, (setting, st) in enumerate([("iid","IID"),("noniid","non-IID")]):
            ax = axes[row][col]
            accs = per_client_final.get(algo, {}).get(setting, [])
            if accs:
                ax.bar(np.arange(len(accs)), accs, color=COLORS[row%len(COLORS)], alpha=0.8)
                ax.axhline(np.mean(accs), color="red", ls="--", lw=1.5,
                           label=f"mean={np.mean(accs):.3f}")
                ax.legend(fontsize=8)
            ax.set(title=f"{algo} — {st}", ylim=(0, 1))
            if col == 0: ax.set_ylabel("Val accuracy")
            ax.grid(axis="y", alpha=0.3)
    for ax in axes[-1]: ax.set_xlabel("Client ID")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


def plot_final_comparison_ci(all_stats_ms, save_path):
    """Bar chart with confidence intervals from multi-seed runs.

    all_stats_ms: {algo: {setting: {"mean": float, "std": float}}}
    """
    algos = list(all_stats_ms.keys())
    n = len(algos)
    x = np.arange(n)
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 6))
    for k, (setting, label, alph) in enumerate([("iid", "IID", 0.90), ("noniid", "non-IID", 0.55)]):
        means = [all_stats_ms[a].get(setting, {}).get("mean", 0.0) for a in algos]
        stds  = [all_stats_ms[a].get(setting, {}).get("std",  0.0) for a in algos]
        bars = ax.bar(x + (k - 0.5) * w, means, w, yerr=stds, capsize=5,
                      alpha=alph, label=label,
                      color=[COLORS[i % len(COLORS)] for i in range(n)],
                      error_kw={"elinewidth": 1.5, "ecolor": "black"})
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.015,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set(xticks=x, ylim=(0, 1.18),
           ylabel="Final accuracy (mean ± std across seeds)",
           title="Algorithm Comparison — Final Accuracy with Confidence Intervals")
    ax.set_xticklabels(algos, fontsize=11)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


def plot_tier_distribution(tiers, save_path):
    """Visualise client-tier assignment for a heterogeneity setting."""
    tier_colors = {"small": COLORS[0], "medium": COLORS[1], "large": COLORS[2]}
    colors = [tier_colors[t] for t in tiers]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(np.arange(len(tiers)), [1] * len(tiers), color=colors, edgecolor="white")
    handles = [plt.Rectangle((0, 0), 1, 1, color=tier_colors[t]) for t in ["small", "medium", "large"]]
    ax.legend(handles, ["Small (poor)", "Medium (mid)", "Large (high)"], fontsize=10)
    ax.set(title="Client Tier Distribution", xlabel="Client ID",
           yticks=[], xlim=(-0.5, len(tiers) - 0.5))
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


# ── IMAGE DATASET LOADERS ──────────────────────────────────────────

def load_mnist():
    print("Loading MNIST...")
    from tensorflow.keras.datasets import mnist as _mnist
    (x_tr, y_tr), (x_te, y_te) = _mnist.load_data()
    x_tr = (x_tr.astype(np.float32) / 255.0)[..., np.newaxis]
    x_te = (x_te.astype(np.float32) / 255.0)[..., np.newaxis]
    print(f"  -> {len(x_tr)} train, {len(x_te)} test")
    return x_tr, y_tr, x_te, y_te


def load_cifar10():
    print("Loading CIFAR-10...")
    from tensorflow.keras.datasets import cifar10 as _cifar10
    (x_tr, y_tr), (x_te, y_te) = _cifar10.load_data()
    x_tr = x_tr.astype(np.float32) / 255.0
    x_te = x_te.astype(np.float32) / 255.0
    y_tr, y_te = y_tr.ravel(), y_te.ravel()
    print(f"  -> {len(x_tr)} train, {len(x_te)} test")
    return x_tr, y_tr, x_te, y_te


def resize_images(x, target_shape):
    """Resize/convert image array to target_shape (H, W, C).

    Handles:  spatial resize  +  grayscale ↔ RGB channel conversion.
    Uses tf.image for GPU-friendly batched resize.
    """
    th, tw, tc = target_shape
    sh = x.shape[1], x.shape[2], x.shape[3]
    if sh == target_shape:
        return x
    out = x
    if sh[0] != th or sh[1] != tw:
        out = tf.image.resize(out, [th, tw]).numpy().astype(np.float32)
    if sh[2] == 1 and tc == 3:
        out = np.repeat(out, 3, axis=-1)
    elif sh[2] == 3 and tc == 1:
        out = np.mean(out, axis=-1, keepdims=True).astype(np.float32)
    return out


# ── DATA PARTITIONING ──────────────────────────────────────────────

def iid_partition(x, y, n_parties, n_samples_per_class, n_classes_total=None,
                  to_categorical_flag=False):
    """Uniform IID partition: each client gets n_samples_per_class per class."""
    return split_dataset(x, y, n_samples_per_class, n_parties, "all",
                         to_categorical_flag=to_categorical_flag,
                         n_classes_total=n_classes_total)[:2]


def dirichlet_partition(x, y, n_parties, alpha, n_classes_total=None,
                        to_categorical_flag=False):
    """Dirichlet-based non-IID partition into n_parties clients."""
    n_cls = n_classes_total if n_classes_total is not None else int(y.max()) + 1
    client_idxs = [[] for _ in range(n_parties)]
    for c in range(n_cls):
        idxs = np.where(y == c)[0]
        if len(idxs) == 0:
            continue
        np.random.shuffle(idxs)
        props = np.random.dirichlet(np.repeat(alpha, n_parties))
        splits = (np.cumsum(props[:-1]) * len(idxs)).astype(int)
        for i, chunk in enumerate(np.split(idxs, splits)):
            client_idxs[i].extend(chunk.tolist())
    pri_x = [x[ix] if ix else x[:0] for ix in client_idxs]
    pri_y = [y[ix] if ix else y[:0] for ix in client_idxs]
    if to_categorical_flag:
        pri_y = [tf.keras.utils.to_categorical(yy, n_cls) if len(yy) else yy
                 for yy in pri_y]
    return pri_x, pri_y


# ── CLIENT TIER SYSTEM ─────────────────────────────────────────────

_TIER_DISTS = {
    "all_small": [1.00, 0.00, 0.00],
    "uniform":   [0.35, 0.35, 0.30],
    "skewed":    [0.60, 0.30, 0.10],
}

TIER_NAMES = ["small", "medium", "large"]


def assign_client_tiers(n_parties, heterogeneity, custom_dist=None):
    """Return list of tier labels for each of n_parties clients.

    heterogeneity: 'all_small' | 'uniform' | 'skewed'
    custom_dist:   optional [frac_small, frac_med, frac_large] override.
    """
    fracs = custom_dist if custom_dist else _TIER_DISTS.get(heterogeneity, _TIER_DISTS["uniform"])
    counts = [round(f * n_parties) for f in fracs]
    diff = n_parties - sum(counts)
    counts[0] += diff
    tiers = []
    for tier, cnt in zip(TIER_NAMES, counts):
        tiers.extend([tier] * max(0, cnt))
    return tiers[:n_parties]


# ── TIER-BASED AGGREGATION ─────────────────────────────────────────

def aggregate_by_tier(tiers, weights_list, n_samples_list):
    """FedAvg within each tier cluster.  Returns {tier: avg_weights}."""
    groups = {}
    for tier, w, n in zip(tiers, weights_list, n_samples_list):
        groups.setdefault(tier, {"w": [], "n": []})
        groups[tier]["w"].append(w)
        groups[tier]["n"].append(n)
    return {t: fedavg_aggregate(d["w"], d["n"]) for t, d in groups.items()}
