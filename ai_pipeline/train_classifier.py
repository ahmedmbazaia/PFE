"""
train_classifier.py — PFE TUNSA
Classification de sources astronomiques par interférométrie

Deux modèles entraînés et comparés :
  1. Random Forest   — sur les 32 features extraits (baseline classique)
  2. 1D CNN          — sur le signal brut 1024 pts (deep learning)

Export final : best_model.tflite (pour déploiement RPi 2)
"""

import numpy as np
import json
import os
import pickle
import time
from pathlib import Path

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_DIR  = Path("dataset")
OUTPUT_DIR   = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)

CLASSES = {
    0: "Étoile ponctuelle",
    1: "Binaire serrée",
    2: "Binaire large",
    3: "Nébuleuse étendue",
    4: "Disque circumstellaire",
    5: "Objet compact",
    6: "Sources multiples"
}
N_CLASSES   = len(CLASSES)
CLASS_NAMES = list(CLASSES.values())

# CNN hyperparams
CNN_EPOCHS      = 40
CNN_BATCH_SIZE  = 64
CNN_LR          = 1e-3

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─────────────────────────────────────────────
# CHARGEMENT DONNÉES
# ─────────────────────────────────────────────

def load_dataset():
    print("Chargement du dataset...")
    data = {}
    for split in ["train", "val", "test"]:
        data[split] = {
            "signals" : np.load(DATASET_DIR / f"X_signals_{split}.npy"),
            "features": np.load(DATASET_DIR / f"X_features_{split}.npy"),
            "labels"  : np.load(DATASET_DIR / f"y_{split}.npy"),
        }
        n = len(data[split]["labels"])
        dist = np.bincount(data[split]["labels"], minlength=N_CLASSES)
        print(f"  {split:6s}: {n} samples | dist: {dist}")

    with open(DATASET_DIR / "config.json") as f:
        config = json.load(f)

    return data, config


# ─────────────────────────────────────────────
# MODEL 1 — RANDOM FOREST
# ─────────────────────────────────────────────

def train_random_forest(data):
    print("\n" + "=" * 55)
    print("  [1/2] RANDOM FOREST — 32 features interférométriques")
    print("=" * 55)

    X_train = data["train"]["features"]
    y_train = data["train"]["labels"]
    X_val   = data["val"]["features"]
    y_val   = data["val"]["labels"]
    X_test  = data["test"]["features"]
    y_test  = data["test"]["labels"]

    # Normalisation
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Entraînement
    print("  Entraînement RF (300 arbres)...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
        verbose=0
    )
    rf.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"  Entraînement terminé en {t_train:.1f}s")

    # Évaluation
    y_pred_val  = rf.predict(X_val)
    y_pred_test = rf.predict(X_test)
    acc_val     = accuracy_score(y_val,  y_pred_val)
    acc_test    = accuracy_score(y_test, y_pred_test)

    print(f"\n  Accuracy Val  : {acc_val*100:.2f}%")
    print(f"  Accuracy Test : {acc_test*100:.2f}%")
    print(f"\n  Rapport classification (test) :")
    print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES))

    # Feature importance
    importances = rf.feature_importances_
    feat_names  = [
        "DC", "RMS", "Amplitude", "IQR", "ZeroCross", "VarMoy", "DevMax", "Puissance",
        "FreqDom", "PwrPic", "PwrDC", "RatioAC_DC", "EnergieSpec", "Largeur",
        "Harm2", "RatioH2", "Centroide", "PicPSD", "MoyPSD", "StdPSD",
        "Visibilité", "SNR_dB", "Cohérence", "FreqTh", "ErrFreq", "DemiPeriode",
        "DériveVis", "Entropie", "Baseline", "AngleIMU", "DistVL53", "FreqSpatiale"
    ]

    # Sauvegarde modèle
    with open(OUTPUT_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump({"model": rf, "scaler": scaler}, f)
    print(f"\n  Modèle sauvegardé : models/random_forest.pkl")

    return {
        "model"       : rf,
        "scaler"      : scaler,
        "acc_val"     : acc_val,
        "acc_test"    : acc_test,
        "y_test"      : y_test,
        "y_pred"      : y_pred_test,
        "importances" : importances,
        "feat_names"  : feat_names,
    }


# ─────────────────────────────────────────────
# MODEL 2 — 1D CNN
# ─────────────────────────────────────────────

def build_1d_cnn(input_length: int, n_classes: int) -> keras.Model:
    """
    Architecture 1D CNN pour classification de signaux interférométriques
    
    Input : signal brut 1024 points BPW34
    
    Couches Conv1D extraient automatiquement :
    - Fréquence de frange (couches basses)
    - Modulation de visibilité (couches moyennes)
    - Patterns complexes multi-sources (couches hautes)
    """
    inp = keras.Input(shape=(input_length, 1), name="signal_brut")

    # Bloc 1 — détection fréquence frange (grand kernel)
    x = layers.Conv1D(32, kernel_size=64, padding="same",
                      activation="relu", name="conv1")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)                      # 1024 → 256
    x = layers.Dropout(0.1)(x)

    # Bloc 2 — extraction motifs périodiques
    x = layers.Conv1D(64, kernel_size=32, padding="same",
                      activation="relu", name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)                      # 256 → 64
    x = layers.Dropout(0.2)(x)

    # Bloc 3 — caractéristiques fines
    x = layers.Conv1D(128, kernel_size=16, padding="same",
                      activation="relu", name="conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)                      # 64 → 16
    x = layers.Dropout(0.2)(x)

    # Bloc 4 — représentation compacte
    x = layers.Conv1D(256, kernel_size=8, padding="same",
                      activation="relu", name="conv4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)             # 16 → 256 (vecteur)
    x = layers.Dropout(0.3)(x)

    # Tête classification
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64,  activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="interferometry_1DCNN")
    return model


def train_cnn(data):
    print("\n" + "=" * 55)
    print("  [2/2] 1D CNN — signal brut BPW34 (1024 points)")
    print("=" * 55)

    # Prépare les données (CNN veut shape [N, 1024, 1])
    X_train = data["train"]["signals"][..., np.newaxis]
    y_train = tf.keras.utils.to_categorical(data["train"]["labels"], N_CLASSES)
    X_val   = data["val"]["signals"][..., np.newaxis]
    y_val   = tf.keras.utils.to_categorical(data["val"]["labels"],   N_CLASSES)
    X_test  = data["test"]["signals"][..., np.newaxis]
    y_test  = data["test"]["labels"]

    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}")

    model = build_1d_cnn(input_length=1024, n_classes=N_CLASSES)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(CNN_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cbs = [
        callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                monitor="val_accuracy", verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=4,
                                    monitor="val_loss", verbose=1),
        callbacks.ModelCheckpoint(str(OUTPUT_DIR / "cnn_best.keras"),
                                  save_best_only=True, monitor="val_accuracy",
                                  verbose=0)
    ]

    print(f"\n  Entraînement ({CNN_EPOCHS} epochs max, batch={CNN_BATCH_SIZE})...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        callbacks=cbs,
        verbose=1
    )
    t_train = time.time() - t0
    print(f"\n  Entraînement terminé en {t_train:.1f}s")

    # Évaluation
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    acc_test    = accuracy_score(y_test, y_pred)
    _, acc_val  = model.evaluate(X_val, y_val, verbose=0)

    print(f"\n  Accuracy Val  : {acc_val*100:.2f}%")
    print(f"  Accuracy Test : {acc_test*100:.2f}%")
    print(f"\n  Rapport classification (test) :")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Export TFLite (pour RPi 2)
    print("\n  Export TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path  = OUTPUT_DIR / "cnn_interferometry.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"  TFLite sauvegardé : {tflite_path} ({size_kb:.1f} KB)")

    return {
        "model"    : model,
        "history"  : history.history,
        "acc_val"  : acc_val,
        "acc_test" : acc_test,
        "y_test"   : y_test,
        "y_pred"   : y_pred,
        "y_pred_prob": y_pred_prob,
    }


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────

def plot_all(rf_results, cnn_results):
    print("\n  Génération des visualisations...")

    fig = plt.figure(figsize=(22, 18), facecolor="#0d0d1a")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

    # ── Confusion Matrix RF ───────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    cm_rf = confusion_matrix(rf_results["y_test"], rf_results["y_pred"])
    cm_rf_norm = cm_rf.astype(float) / cm_rf.sum(axis=1, keepdims=True)
    short_names = ["Étoile", "Bin.S", "Bin.L", "Nébul.", "Disque", "Compact", "Multi"]
    sns.heatmap(cm_rf_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=short_names, yticklabels=short_names,
                ax=ax1, cbar=True,
                annot_kws={"size": 9})
    ax1.set_title(f"Random Forest — Matrice de confusion\nTest acc: {rf_results['acc_test']*100:.2f}%",
                  color="white", fontsize=11, pad=8)
    ax1.set_xlabel("Prédit", color="#aaa", fontsize=9)
    ax1.set_ylabel("Réel", color="#aaa", fontsize=9)
    ax1.tick_params(colors="#ccc", labelsize=8)
    ax1.set_facecolor("#111122")

    # ── Confusion Matrix CNN ──────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    cm_cnn = confusion_matrix(cnn_results["y_test"], cnn_results["y_pred"])
    cm_cnn_norm = cm_cnn.astype(float) / cm_cnn.sum(axis=1, keepdims=True)
    sns.heatmap(cm_cnn_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=short_names, yticklabels=short_names,
                ax=ax2, cbar=True,
                annot_kws={"size": 9})
    ax2.set_title(f"1D CNN — Matrice de confusion\nTest acc: {cnn_results['acc_test']*100:.2f}%",
                  color="white", fontsize=11, pad=8)
    ax2.set_xlabel("Prédit", color="#aaa", fontsize=9)
    ax2.set_ylabel("Réel", color="#aaa", fontsize=9)
    ax2.tick_params(colors="#ccc", labelsize=8)
    ax2.set_facecolor("#111122")

    # ── CNN Learning Curves ───────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    h = cnn_results["history"]
    epochs = range(1, len(h["accuracy"]) + 1)
    ax3.plot(epochs, [v * 100 for v in h["accuracy"]],
             color="#00d4ff", lw=2, label="Train acc")
    ax3.plot(epochs, [v * 100 for v in h["val_accuracy"]],
             color="#ff6b6b", lw=2, linestyle="--", label="Val acc")
    ax3.set_facecolor("#111122")
    ax3.set_title("CNN — Courbes d'apprentissage", color="white", fontsize=11)
    ax3.set_xlabel("Époque", color="#aaa", fontsize=9)
    ax3.set_ylabel("Accuracy (%)", color="#aaa", fontsize=9)
    ax3.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    ax3.tick_params(colors="#888", labelsize=8)
    for sp in ax3.spines.values():
        sp.set_color("#333")

    # ── CNN Loss Curves ───────────────────────
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.plot(epochs, h["loss"],     color="#ffd93d", lw=2, label="Train loss")
    ax4.plot(epochs, h["val_loss"], color="#c77dff", lw=2, linestyle="--", label="Val loss")
    ax4.set_facecolor("#111122")
    ax4.set_title("CNN — Courbes de perte", color="white", fontsize=11)
    ax4.set_xlabel("Époque", color="#aaa", fontsize=9)
    ax4.set_ylabel("Loss", color="#aaa", fontsize=9)
    ax4.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    ax4.tick_params(colors="#888", labelsize=8)
    for sp in ax4.spines.values():
        sp.set_color("#333")

    # ── Feature Importances RF (top 15) ──────
    ax5 = fig.add_subplot(gs[2, :2])
    imp  = rf_results["importances"]
    names = rf_results["feat_names"]
    top15_idx = np.argsort(imp)[-15:]
    colors_bar = plt.cm.plasma(np.linspace(0.3, 1.0, 15))
    ax5.barh(range(15), imp[top15_idx], color=colors_bar, edgecolor="#222")
    ax5.set_yticks(range(15))
    ax5.set_yticklabels([names[i] for i in top15_idx], fontsize=8, color="#ccc")
    ax5.set_facecolor("#111122")
    ax5.set_title("Top 15 Features — Random Forest\n(importance relative)",
                  color="white", fontsize=11)
    ax5.set_xlabel("Importance", color="#aaa", fontsize=9)
    ax5.tick_params(colors="#888", labelsize=8)
    for sp in ax5.spines.values():
        sp.set_color("#333")

    # ── Comparaison RF vs CNN ─────────────────
    ax6 = fig.add_subplot(gs[2, 2:])
    rf_per_class  = cm_rf.diagonal()  / cm_rf.sum(axis=1)
    cnn_per_class = cm_cnn.diagonal() / cm_cnn.sum(axis=1)
    x = np.arange(N_CLASSES)
    w = 0.35
    ax6.bar(x - w/2, rf_per_class  * 100, w, label="Random Forest",
            color="#00d4ff", alpha=0.85, edgecolor="#333")
    ax6.bar(x + w/2, cnn_per_class * 100, w, label="1D CNN",
            color="#ff6b6b", alpha=0.85, edgecolor="#333")
    ax6.set_xticks(x)
    ax6.set_xticklabels(short_names, fontsize=8, color="#ccc", rotation=20)
    ax6.set_facecolor("#111122")
    ax6.set_title("Accuracy par classe — RF vs CNN", color="white", fontsize=11)
    ax6.set_ylabel("Accuracy (%)", color="#aaa", fontsize=9)
    ax6.set_ylim(0, 110)
    ax6.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    ax6.tick_params(colors="#888", labelsize=8)
    for sp in ax6.spines.values():
        sp.set_color("#333")

    fig.suptitle("Résultats Classification Interférométrique — PFE TUNSA\n"
                 "Random Forest (32 features) vs 1D CNN (signal brut 1024 pts)",
                 color="white", fontsize=13, y=0.98)

    out_path = OUTPUT_DIR / "plots" / "classification_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Résultats sauvegardés : {out_path}")


# ─────────────────────────────────────────────
# RÉSUMÉ FINAL
# ─────────────────────────────────────────────

def print_summary(rf_results, cnn_results):
    print("\n" + "=" * 55)
    print("  RÉSUMÉ FINAL")
    print("=" * 55)
    print(f"  Random Forest  — Test acc : {rf_results['acc_test']*100:.2f}%")
    print(f"  1D CNN         — Test acc : {cnn_results['acc_test']*100:.2f}%")
    winner = "CNN" if cnn_results['acc_test'] > rf_results['acc_test'] else "RF"
    print(f"\n  Meilleur modèle : {winner}")
    print(f"\n  Fichiers générés :")
    print(f"    models/random_forest.pkl       — RF + scaler")
    print(f"    models/cnn_best.keras          — CNN Keras")
    print(f"    models/cnn_interferometry.tflite — déploiement RPi 2")
    print(f"    models/plots/classification_results.png")
    print("\n  Prochaine étape → train_fringe_analyzer.py (1D CNN + LSTM)")
    print("=" * 55 + "\n")

    # Sauvegarde JSON résumé (pour rapport)
    summary = {
        "random_forest": {
            "acc_val" : round(rf_results["acc_val"]  * 100, 2),
            "acc_test": round(rf_results["acc_test"] * 100, 2),
            "n_trees" : 300,
            "n_features": 32
        },
        "cnn_1d": {
            "acc_val" : round(cnn_results["acc_val"]  * 100, 2),
            "acc_test": round(cnn_results["acc_test"] * 100, 2),
            "epochs_run": len(cnn_results["history"]["accuracy"]),
            "architecture": "Conv1D×4 + GAP + Dense×2"
        },
        "best_model": winner,
        "classes": CLASSES,
        "tflite_export": "models/cnn_interferometry.tflite"
    }
    with open(OUTPUT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Classifier Entraînement — PFE TUNSA")
    print("  Sources astronomiques × 7 classes")
    print("=" * 55 + "\n")

    data, config = load_dataset()

    rf_results  = train_random_forest(data)
    cnn_results = train_cnn(data)

    plot_all(rf_results, cnn_results)
    print_summary(rf_results, cnn_results)
