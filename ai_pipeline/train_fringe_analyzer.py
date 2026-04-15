"""
train_fringe_analyzer.py — PFE TUNSA
Analyse de franges interférométriques

Objectif : extraire la visibilité complexe V = |V|·exp(iφ) depuis le signal brut BPW34
C'est le cœur de l'interférométrie — ce que font VLTI et CHARA en temps réel

Architecture : 1D CNN (extraction features) + BiLSTM (dépendances temporelles)
Input  : signal BPW34 brut 1024 points
Output : [|V|, φ, SNR] — visibilité complexe + qualité signal
"""

import numpy as np
import json
import os
import time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_DIR = Path("dataset")
OUTPUT_DIR  = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)

LAMBDA_NM      = 625e-9
SAMPLE_RATE    = 1000
N_SAMPLES      = 1024
EPOCHS         = 50
BATCH_SIZE     = 64
LR             = 1e-3

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─────────────────────────────────────────────
# GÉNÉRATION DATASET SPÉCIALISÉ FRANGES
# (labels continus : |V|, φ, SNR)
# ─────────────────────────────────────────────

def generate_fringe_dataset(n_samples: int = 10000):
    """
    Génère un dataset dédié à l'estimation de visibilité.
    Labels continus (régression) : [|V|, φ normalisé, SNR normalisé]
    
    Couvre tout l'espace des visibilités possibles :
    |V| ∈ [0.0, 1.0]  — de totalement incohérent à parfaitement cohérent
    φ  ∈ [-π, π]       — phase complète
    """
    from scipy.fft import fft, fftfreq

    print(f"Génération dataset franges ({n_samples} samples)...")

    X = np.zeros((n_samples, N_SAMPLES), dtype=np.float32)
    y = np.zeros((n_samples, 3),         dtype=np.float32)  # |V|, φ, SNR_norm

    for i in range(n_samples):
        # Paramètres aléatoires
        V_mod      = np.random.uniform(0.0, 1.0)        # visibilité cible
        phi        = np.random.uniform(-np.pi, np.pi)   # phase cible
        baseline   = np.random.uniform(0.05, 0.50)
        imu_angle  = np.random.uniform(-30, 30)
        I1         = np.random.uniform(0.3, 1.0)
        I2         = np.random.uniform(0.3, 1.0)
        snr_db     = np.random.uniform(5, 40)

        # Signal interférométrique physique
        t      = np.linspace(0, N_SAMPLES / SAMPLE_RATE, N_SAMPLES)
        B_proj = baseline * np.cos(np.radians(imu_angle))
        f_f    = (B_proj / LAMBDA_NM) * 2e-6

        signal = (I1 + I2
                  + 2 * np.sqrt(I1 * I2) * V_mod
                  * np.cos(2 * np.pi * f_f * t + phi))

        # Bruit réaliste
        snr_lin  = 10 ** (snr_db / 20)
        rms      = np.sqrt(np.mean(signal ** 2))
        noise    = np.random.normal(0, rms / (snr_lin + 1e-10), N_SAMPLES)
        shot     = np.random.normal(0, 0.01 * np.sqrt(I1 + I2), N_SAMPLES)
        signal   = signal + noise + shot

        # Normalisation ADC
        signal = np.clip(signal, 0, None)
        s_max  = signal.max() + 1e-10
        signal = signal / s_max

        X[i] = signal.astype(np.float32)
        y[i] = [V_mod,
                phi / np.pi,              # normalisé [-1, 1]
                snr_db / 40.0]            # normalisé [0, 1]

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n_samples} done")

    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]

    # Split 80/10/10
    n_tr = int(0.8 * n_samples)
    n_va = int(0.9 * n_samples)
    splits = {
        "train": (X[:n_tr],       y[:n_tr]),
        "val":   (X[n_tr:n_va],   y[n_tr:n_va]),
        "test":  (X[n_va:],       y[n_va:])
    }

    # Sauvegarde
    for name, (xs, ys) in splits.items():
        np.save(DATASET_DIR / f"fringe_X_{name}.npy", xs)
        np.save(DATASET_DIR / f"fringe_y_{name}.npy", ys)
        print(f"  {name}: {len(ys)} samples")

    print("Dataset franges sauvegardé.\n")
    return splits


# ─────────────────────────────────────────────
# ARCHITECTURE : CNN + BiLSTM
# ─────────────────────────────────────────────

def build_fringe_analyzer(input_length: int) -> keras.Model:
    """
    CNN + BiLSTM pour extraction de visibilité complexe
    
    Pourquoi BiLSTM ?
    - Le signal de frange est une série temporelle
    - LSTM capture les dépendances long-terme (variation de phase)
    - Bi-directionnel = contexte passé ET futur
    
    Pourquoi CNN en premier ?
    - Conv1D extrait les features locaux (fréquence de frange)
    - Réduit la dimensionnalité avant LSTM (plus rapide)
    - Hiérarchie : fréquence → modulation → visibilité globale
    """
    inp = keras.Input(shape=(input_length, 1), name="signal_bpw34")

    # ── Bloc CNN : extraction features temporels locaux ──
    x = layers.Conv1D(32, kernel_size=64, padding="same",
                      activation="relu", name="conv_local")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)          # 1024 → 256
    x = layers.Dropout(0.1)(x)

    x = layers.Conv1D(64, kernel_size=32, padding="same",
                      activation="relu", name="conv_pattern")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)          # 256 → 64
    x = layers.Dropout(0.15)(x)

    x = layers.Conv1D(128, kernel_size=16, padding="same",
                      activation="relu", name="conv_global")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)          # 64 → 32
    x = layers.Dropout(0.2)(x)

    # ── Bloc BiLSTM : dépendances temporelles globales ──
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True), name="bilstm_1"
    )(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False), name="bilstm_2"
    )(x)
    x = layers.Dropout(0.2)(x)

    # ── Tête de régression ──
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)

    # 3 sorties : |V|, φ/π, SNR_norm
    out_V   = layers.Dense(1, activation="sigmoid", name="visibility")(x)  # [0,1]
    out_phi = layers.Dense(1, activation="tanh",    name="phase")(x)        # [-1,1]
    out_snr = layers.Dense(1, activation="sigmoid", name="snr")(x)          # [0,1]

    out = layers.Concatenate(name="output")([out_V, out_phi, out_snr])

    model = keras.Model(inputs=inp, outputs=out, name="FringeAnalyzer_CNN_BiLSTM")
    return model


# ─────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────

def train_fringe_analyzer(splits):
    print("=" * 55)
    print("  FRINGE ANALYZER — CNN + BiLSTM")
    print("  Régression : |V|, φ, SNR depuis signal BPW34")
    print("=" * 55)

    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    # Reshape pour CNN : (N, 1024, 1)
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[...,   np.newaxis]
    X_test  = X_test[...,  np.newaxis]

    print(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")

    model = build_fringe_analyzer(N_SAMPLES)
    model.summary()

    # Loss pondérée : visibilité plus importante que SNR
    def weighted_mse(y_true, y_pred):
        w = tf.constant([2.0, 1.5, 0.5])   # poids [|V|, φ, SNR]
        return tf.reduce_mean(w * tf.square(y_true - y_pred))

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=weighted_mse,
        metrics=["mae"]
    )

    cbs = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                monitor="val_loss", verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5,
                                    monitor="val_loss", verbose=1),
        callbacks.ModelCheckpoint(str(OUTPUT_DIR / "fringe_analyzer_best.keras"),
                                  save_best_only=True, monitor="val_loss",
                                  verbose=0)
    ]

    print(f"\n  Entraînement ({EPOCHS} epochs max, batch={BATCH_SIZE})...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1
    )
    print(f"\n  Entraîné en {time.time()-t0:.1f}s")

    return model, history.history, X_test, y_test


# ─────────────────────────────────────────────
# ÉVALUATION
# ─────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    print("\n  Évaluation sur test set...")

    y_pred = model.predict(X_test[..., np.newaxis]
                           if X_test.ndim == 2 else X_test,
                           verbose=0)

    # Dénormalisation
    V_true  = y_test[:, 0];         V_pred  = np.clip(y_pred[:, 0], 0, 1)
    phi_true = y_test[:, 1] * np.pi; phi_pred = y_pred[:, 1] * np.pi
    snr_true = y_test[:, 2] * 40;    snr_pred = np.clip(y_pred[:, 2], 0, 1) * 40

    mae_V   = mean_absolute_error(V_true,   V_pred)
    mae_phi = mean_absolute_error(phi_true, phi_pred)
    mae_snr = mean_absolute_error(snr_true, snr_pred)
    r2_V    = r2_score(V_true, V_pred)
    r2_phi  = r2_score(phi_true, phi_pred)

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Résultats Fringe Analyzer           │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │  Visibilité |V|  MAE = {mae_V:.4f}  R²={r2_V:.3f} │")
    print(f"  │  Phase φ        MAE = {mae_phi:.4f} rad       │")
    print(f"  │  SNR            MAE = {mae_snr:.2f} dB         │")
    print(f"  └─────────────────────────────────────┘")

    return {
        "V_true": V_true,   "V_pred": V_pred,
        "phi_true": phi_true, "phi_pred": phi_pred,
        "snr_true": snr_true, "snr_pred": snr_pred,
        "mae_V": mae_V, "mae_phi": mae_phi,
        "r2_V": r2_V,   "r2_phi": r2_phi
    }


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────

def plot_results(history, eval_results):
    print("\n  Génération visualisations...")

    fig = plt.figure(figsize=(22, 16), facecolor="#0d0d1a")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Learning curves ───────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ep = range(1, len(history["loss"]) + 1)
    ax1.plot(ep, history["loss"],     color="#00d4ff", lw=2, label="Train loss")
    ax1.plot(ep, history["val_loss"], color="#ff6b6b", lw=2, ls="--", label="Val loss")
    ax1.set_facecolor("#111122")
    ax1.set_title("Fringe Analyzer — Courbes d'apprentissage\n(CNN + BiLSTM)",
                  color="white", fontsize=11)
    ax1.set_xlabel("Époque", color="#aaa", fontsize=9)
    ax1.set_ylabel("Loss pondérée", color="#aaa", fontsize=9)
    ax1.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
    ax1.tick_params(colors="#888", labelsize=8)
    for sp in ax1.spines.values(): sp.set_color("#333")

    # ── MAE curve ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(ep, history["mae"],     color="#ffd93d", lw=2, label="Train MAE")
    ax2.plot(ep, history["val_mae"], color="#c77dff", lw=2, ls="--", label="Val MAE")
    ax2.set_facecolor("#111122")
    ax2.set_title("MAE au cours\nde l'entraînement", color="white", fontsize=11)
    ax2.set_xlabel("Époque", color="#aaa", fontsize=9)
    ax2.set_ylabel("MAE", color="#aaa", fontsize=9)
    ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    ax2.tick_params(colors="#888", labelsize=8)
    for sp in ax2.spines.values(): sp.set_color("#333")

    # ── Scatter |V| vrai vs prédit ────────────
    ax3 = fig.add_subplot(gs[1, 0])
    idx = np.random.choice(len(eval_results["V_true"]), 500, replace=False)
    sc  = ax3.scatter(eval_results["V_true"][idx],
                      eval_results["V_pred"][idx],
                      c=eval_results["snr_true"][idx],
                      cmap="plasma", alpha=0.6, s=15)
    ax3.plot([0,1],[0,1], "w--", lw=1, alpha=0.5)
    ax3.set_facecolor("#111122")
    ax3.set_title(f"Visibilité |V|\nVrai vs Prédit  (MAE={eval_results['mae_V']:.4f}  R²={eval_results['r2_V']:.3f})",
                  color="white", fontsize=9)
    ax3.set_xlabel("|V| vrai", color="#aaa", fontsize=8)
    ax3.set_ylabel("|V| prédit", color="#aaa", fontsize=8)
    ax3.tick_params(colors="#888", labelsize=7)
    for sp in ax3.spines.values(): sp.set_color("#333")
    plt.colorbar(sc, ax=ax3).set_label("SNR (dB)", color="#aaa", fontsize=7)

    # ── Scatter φ vrai vs prédit ──────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(eval_results["phi_true"][idx],
                eval_results["phi_pred"][idx],
                c=eval_results["V_true"][idx],
                cmap="cool", alpha=0.6, s=15)
    ax4.plot([-np.pi, np.pi], [-np.pi, np.pi], "w--", lw=1, alpha=0.5)
    ax4.set_facecolor("#111122")
    ax4.set_title(f"Phase φ\nVrai vs Prédit  (MAE={eval_results['mae_phi']:.4f} rad)",
                  color="white", fontsize=9)
    ax4.set_xlabel("φ vrai (rad)", color="#aaa", fontsize=8)
    ax4.set_ylabel("φ prédit (rad)", color="#aaa", fontsize=8)
    ax4.tick_params(colors="#888", labelsize=7)
    for sp in ax4.spines.values(): sp.set_color("#333")

    # ── Distribution erreur |V| ───────────────
    ax5 = fig.add_subplot(gs[1, 2])
    err_V = eval_results["V_pred"] - eval_results["V_true"]
    ax5.hist(err_V, bins=50, color="#00d4ff", alpha=0.8, edgecolor="#333")
    ax5.axvline(0, color="white", lw=1.5, ls="--")
    ax5.axvline(eval_results["mae_V"], color="#ff6b6b", lw=1.5, ls=":", label=f"MAE={eval_results['mae_V']:.4f}")
    ax5.axvline(-eval_results["mae_V"], color="#ff6b6b", lw=1.5, ls=":")
    ax5.set_facecolor("#111122")
    ax5.set_title("Distribution erreur |V|\n(prédit − vrai)", color="white", fontsize=9)
    ax5.set_xlabel("Erreur", color="#aaa", fontsize=8)
    ax5.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")
    ax5.tick_params(colors="#888", labelsize=7)
    for sp in ax5.spines.values(): sp.set_color("#333")

    # ── Polar plot : visibilité complexe ──────
    ax6 = fig.add_subplot(gs[2, 0], projection="polar")
    ax6.set_facecolor("#111122")
    n_show = 200
    r_true = eval_results["V_true"][:n_show]
    t_true = eval_results["phi_true"][:n_show]
    r_pred = eval_results["V_pred"][:n_show]
    t_pred = eval_results["phi_pred"][:n_show]
    ax6.scatter(t_true, r_true, c="#00d4ff", s=8, alpha=0.6, label="Vrai")
    ax6.scatter(t_pred, r_pred, c="#ff6b6b", s=8, alpha=0.6, label="Prédit")
    ax6.set_title("Plan complexe V\n(polaire : |V|, φ)", color="white",
                  fontsize=9, pad=15)
    ax6.tick_params(colors="#888", labelsize=7)
    ax6.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white",
               loc="upper right")

    # ── Résumé performance ────────────────────
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis("off")
    ax7.set_facecolor("#111122")
    summary_text = (
        f"FRINGE ANALYZER — RÉSULTATS\n\n"
        f"Architecture : 1D CNN (3 couches) + BiLSTM (2 couches)\n"
        f"Input        : Signal BPW34 brut — 1024 points @ 1kHz\n"
        f"Output       : Visibilité complexe V = |V|·exp(iφ)\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"  Visibilité |V|   MAE = {eval_results['mae_V']:.4f}   R² = {eval_results['r2_V']:.4f}\n"
        f"  Phase φ          MAE = {eval_results['mae_phi']:.4f} rad\n"
        f"  SNR              MAE = {eval_results['mae_snr']:.2f} dB\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"  Fichier exporté : models/fringe_analyzer.tflite\n"
        f"  Utilisation     : déploiement RPi 2 + intégration pipeline"
    )
    ax7.text(0.05, 0.95, summary_text,
             transform=ax7.transAxes,
             color="white", fontsize=9.5,
             verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a1a3e",
                       edgecolor="#00d4ff", linewidth=1.5))

    fig.suptitle("Fringe Analyzer CNN+BiLSTM — Extraction Visibilité Interférométrique\nPFE TUNSA — Démonstration Éducative Interférométrie Spatiale",
                 color="white", fontsize=12, y=0.98)

    out = OUTPUT_DIR / "plots" / "fringe_analyzer_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Visualisation : {out}")


# ─────────────────────────────────────────────
# EXPORT TFLITE
# ─────────────────────────────────────────────

def export_tflite(model):
    print("\n  Export TFLite (RPi 2)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    path = OUTPUT_DIR / "fringe_analyzer.tflite"
    with open(path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Sauvegardé : {path} ({size_kb:.1f} KB)")
    return path


# ─────────────────────────────────────────────
# INFÉRENCE TEMPS RÉEL (utilisé par station/main.py)
# ─────────────────────────────────────────────

INFERENCE_CODE = '''
# ─── Intégration dans station/main.py ────────────────────────────────────────
# Ajouter ces fonctions pour utiliser le Fringe Analyzer en temps réel

import numpy as np
import tflite_runtime.interpreter as tflite
# ou : import tensorflow.lite as tflite (si tensorflow installé)

class FringeAnalyzer:
    """
    Wrapper temps réel pour l'analyse de franges sur RPi 2.
    Reçoit le buffer BPW34 et retourne la visibilité complexe.
    """
    def __init__(self, model_path: str = "models/fringe_analyzer.tflite"):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_idx  = self.interpreter.get_input_details()[0]["index"]
        self.output_idx = self.interpreter.get_output_details()[0]["index"]
        print(f"FringeAnalyzer chargé depuis {model_path}")

    def analyze(self, signal_buffer: np.ndarray) -> dict:
        """
        Analyse un buffer de signal BPW34.
        
        Args:
            signal_buffer : np.ndarray shape (1024,) — signal brut normalisé
        
        Returns:
            dict avec visibility, phase_rad, snr_db
        """
        # Normalisation
        s = signal_buffer.astype(np.float32)
        s = np.clip(s, 0, None)
        s = s / (s.max() + 1e-10)

        # Inférence TFLite
        inp = s.reshape(1, 1024, 1)
        self.interpreter.set_tensor(self.input_idx, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_idx)[0]

        return {
            "visibility" : float(np.clip(out[0], 0, 1)),
            "phase_rad"  : float(out[1] * np.pi),
            "snr_db"     : float(np.clip(out[2], 0, 1) * 40),
        }

# Usage dans la boucle principale :
# analyzer = FringeAnalyzer("models/fringe_analyzer.tflite")
# result = analyzer.analyze(bpw34_buffer)
# print(f"|V| = {result['visibility']:.3f}  φ = {result['phase_rad']:.3f} rad")
'''


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Fringe Analyzer — PFE TUNSA")
    print("  CNN + BiLSTM — Extraction Visibilité Complexe")
    print("=" * 55 + "\n")

    # Génère dataset spécialisé si absent
    fringe_train_path = DATASET_DIR / "fringe_X_train.npy"
    if not fringe_train_path.exists():
        splits = generate_fringe_dataset(n_samples=10000)
    else:
        print("Dataset franges existant chargé.")
        splits = {}
        for name in ["train", "val", "test"]:
            splits[name] = (
                np.load(DATASET_DIR / f"fringe_X_{name}.npy"),
                np.load(DATASET_DIR / f"fringe_y_{name}.npy")
            )
            print(f"  {name}: {len(splits[name][1])} samples")

    # Entraînement
    model, history, X_test, y_test = train_fringe_analyzer(splits)

    # Évaluation
    eval_results = evaluate(model, X_test, y_test)

    # Visualisations
    plot_results(history, eval_results)

    # Export TFLite
    export_tflite(model)

    # Sauvegarde code d'intégration
    with open(OUTPUT_DIR / "fringe_analyzer_integration.py", "w") as f:
        f.write(INFERENCE_CODE)
    print(f"  Code intégration : models/fringe_analyzer_integration.py")

    # Résumé JSON
    summary = {
        "architecture" : "1D CNN (×3) + BiLSTM (×2) + Dense regression",
        "input"        : "BPW34 signal 1024pts @ 1kHz",
        "output"       : ["|V| visibilité [0,1]", "φ phase [-π,π]", "SNR [dB]"],
        "metrics"      : {
            "mae_visibility": round(eval_results["mae_V"], 4),
            "mae_phase_rad" : round(eval_results["mae_phi"], 4),
            "mae_snr_db"    : round(eval_results["mae_snr"], 2),
            "r2_visibility" : round(eval_results["r2_V"], 4),
        },
        "tflite"       : "models/fringe_analyzer.tflite"
    }
    with open(OUTPUT_DIR / "fringe_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("  Fringe Analyzer entraîné ✓")
    print("  Prochaine étape → train_unet_reconstructor.py")
    print("=" * 55 + "\n")
