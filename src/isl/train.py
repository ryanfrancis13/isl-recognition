from isl import core  # must import first to set TF log env vars

import json
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

tf.get_logger().setLevel("ERROR")


def clip_to_windows(clip, n_wins=8):
    # Slice stream clips (>35 frames) into n_wins evenly spaced 20-frame chunks resampled to SEQ_LEN
    if len(clip) > 35:
        win_len = min(20, len(clip) // 2)
        starts = np.linspace(0, len(clip) - win_len, n_wins, dtype=int)
        return [core.to_window(clip[s : s + win_len]) for s in starts]
    return [core.to_window(clip)]


def load_dataset():
    sessions = sorted(p.name for p in core.RAW_DIR.glob("session_*") if p.is_dir())
    signs = sorted({p.name for s in sessions for p in (core.RAW_DIR / s).iterdir() if p.is_dir() and any(p.glob("*.npy"))})
    label_map = {name: i for i, name in enumerate(signs)}

    tr_X, tr_y, val_X, val_y, te_X, te_y = [], [], [], [], [], []
    multi = len(sessions) >= 2

    for s in sessions:
        for sign in signs:
            wins = []
            for f in sorted((core.RAW_DIR / s / sign).glob("*.npy")):
                wins.extend(clip_to_windows(np.load(f)))
            lbl = label_map[sign]
            if multi and s == sessions[-1]:
                te_X.extend(wins)
                te_y.extend([lbl] * len(wins))
            else:
                # Hold out the last window(s) of each sign in train sessions for non-overlapping val
                n_val = max(1, len(wins) // 5)
                tr_X.extend(wins[:-n_val])
                tr_y.extend([lbl] * (len(wins) - n_val))
                val_X.extend(wins[-n_val:])
                val_y.extend([lbl] * n_val)

    if not tr_X:
        raise SystemExit("No .npy recordings found in data/raw/session_*/")

    X_tr, y_tr = np.array(tr_X, dtype=np.float32), np.array(tr_y, dtype=np.int32)
    X_val, y_val = np.array(val_X, dtype=np.float32), np.array(val_y, dtype=np.int32)

    if multi:
        return X_tr, y_tr, X_val, y_val, np.array(te_X, dtype=np.float32), np.array(te_y, dtype=np.int32), signs, f"held-out {sessions[-1]}"

    # shortcut: fallback when only 1 session exists, record >=2 sessions for leak-free test
    X_tr, X_te, y_tr, y_te = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)
    return X_tr, y_tr, X_val, y_val, X_te, y_te, signs, "single-session fallback (record >=2 sessions!)"


def augment(x):
    mask = (x != 0.0).astype(np.float32)
    out = x * np.random.uniform(0.9, 1.1) + np.random.normal(0.0, 0.012, x.shape).astype(np.float32) * mask
    shift = random.randint(-2, 2)
    if shift > 0:
        out = np.pad(out, ((shift, 0), (0, 0)), mode="edge")[: core.SEQ_LEN]
    elif shift < 0:
        out = np.pad(out, ((0, -shift), (0, 0)), mode="edge")[-core.SEQ_LEN :]
    return out


def make_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(core.SEQ_LEN, core.FEAT_DIM)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.35),
        layers.LSTM(64),
        layers.Dropout(0.35),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    X_tr, y_tr, X_val, y_val, X_te, y_te, signs, split_info = load_dataset()
    print(f"Split: {split_info} | Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)} | Classes: {signs}")

    X_aug = np.concatenate([X_tr, np.array([augment(x) for x in X_tr]), np.array([augment(x) for x in X_tr])])
    y_aug = np.tile(y_tr, 3)

    model = make_model(len(signs))
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]
    model.fit(X_aug, y_aug, validation_data=(X_val, y_val), batch_size=16, epochs=80, callbacks=cb, verbose=1)

    preds = np.argmax(model.predict(X_te, verbose=0), axis=1)
    print("\n" + classification_report(y_te, preds, target_names=signs, digits=3))

    core.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(core.ARTIFACTS_DIR / "model.keras")
    (core.ARTIFACTS_DIR / "labels.json").write_text(json.dumps(signs, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_te, preds)
    fig, ax = plt.subplots(figsize=(max(6, len(signs) * 0.5), max(5, len(signs) * 0.45)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(signs)), signs, rotation=45, ha="right")
    ax.set_yticks(range(len(signs)), signs)
    for i in range(len(signs)):
        for j in range(len(signs)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(core.ARTIFACTS_DIR / "confusion_matrix.png")
    print(f"Saved model, labels.json, and confusion_matrix.png to {core.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
