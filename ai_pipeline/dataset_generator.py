"""
Interferometry Dataset Generator — PFE TUNSA
Génère un dataset synthétique basé sur la physique réelle de l'interférométrie
Capteurs simulés : BPW34 × 2, MPU9250 × 2, VL53L0X × 2

Signal physique : I(t) = I1 + I2 + 2*sqrt(I1*I2)*|V|*cos(2π*B*θ/λ + φ + noise)
"""

import numpy as np
import os
import json
import pickle
from pathlib import Path
from scipy.fft import fft, fftfreq
from scipy.signal import welch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# CONFIG GLOBALE
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = Path("dataset")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)

# Paramètres hardware réels
LAMBDA_NM       = 625e-9        # longueur d'onde LED rouge (m)
BASELINE_MIN    = 0.05          # baseline min (m) — distance min entre télescopes
BASELINE_MAX    = 0.50          # baseline max (m)
SAMPLE_RATE     = 1000          # Hz — fréquence d'échantillonnage BPW34
N_SAMPLES       = 1024          # points par signal temporel
SHOT_NOISE      = 0.01          # bruit shot BPW34
THERMAL_NOISE   = 0.005         # bruit thermique
ADC_BITS        = 12            # résolution ADC ESP8266/TTGO

# 7 classes
CLASSES = {
    0: "etoile_ponctuelle",
    1: "binaire_serree",
    2: "binaire_large",
    3: "nebuleuse_etendue",
    4: "disque_circumstellaire",
    5: "objet_compact",
    6: "sources_multiples"
}

N_SYNTHETIC = 8000   # total échantillons synthétiques
N_PER_CLASS = N_SYNTHETIC // len(CLASSES)   # ~1142 par classe


# ─────────────────────────────────────────────
# PHYSIQUE INTERFÉROMÉTRIQUE
# ─────────────────────────────────────────────

def compute_visibility(source_type: int, baseline: float, params: dict) -> complex:
    """
    Calcule la visibilité complexe V(u,v) via le théorème de Van Cittert-Zernike
    V = FT2D[I(θ)] évaluée à la fréquence spatiale u = B/λ
    
    Retourne V complexe = |V| * exp(iφ)
    """
    u = baseline / LAMBDA_NM  # fréquence spatiale (rad⁻¹)

    if source_type == 0:  # Étoile ponctuelle — source delta de Dirac
        # FT d'un Dirac = constante → V = 1 (visibilité parfaite)
        V_mod = 1.0
        phi   = 0.0

    elif source_type == 1:  # Binaire serrée — 2 sources ponctuelles proches
        # I(θ) = δ(θ) + r·δ(θ - θ_sep) avec θ_sep petit
        theta_sep = params.get("theta_sep", 0.5e-6)  # rad — séparation angulaire
        r         = params.get("flux_ratio", 0.8)      # rapport de flux
        # V = (1 + r·exp(2πi·u·θ_sep)) / (1 + r)
        V_complex = (1 + r * np.exp(2j * np.pi * u * theta_sep)) / (1 + r)
        V_mod     = np.abs(V_complex)
        phi       = np.angle(V_complex)

    elif source_type == 2:  # Binaire large — séparation angulaire grande
        theta_sep = params.get("theta_sep", 3e-6)   # rad — plus grand que binaire serrée
        r         = params.get("flux_ratio", 0.6)
        V_complex = (1 + r * np.exp(2j * np.pi * u * theta_sep)) / (1 + r)
        V_mod     = np.abs(V_complex)
        phi       = np.angle(V_complex)

    elif source_type == 3:  # Nébuleuse étendue — disque uniforme
        # FT d'un disque uniforme = 2*J1(x)/x (fonction de Bessel)
        from scipy.special import j1
        theta_disk = params.get("theta_disk", 2e-6)  # rayon angulaire
        x = np.pi * u * theta_disk
        V_mod = np.abs(2 * j1(x + 1e-10) / (x + 1e-10)) if x > 0 else 1.0
        phi   = 0.0

    elif source_type == 4:  # Disque circumstellaire — anneau mince
        # FT d'un anneau = J0(2π·u·θ_ring)
        from scipy.special import j0
        theta_ring = params.get("theta_ring", 1.5e-6)
        V_mod = np.abs(j0(2 * np.pi * u * theta_ring))
        phi   = np.pi if V_mod < 0 else 0.0
        V_mod = np.abs(V_mod)

    elif source_type == 5:  # Objet compact — source quasi-ponctuelle très brillante
        # Comme étoile ponctuelle mais flux très élevé, très haute cohérence
        V_mod = params.get("coherence", 0.98)
        phi   = params.get("phase_offset", 0.05)

    elif source_type == 6:  # Sources multiples — N composantes
        # Somme de N étoiles ponctuelles à positions différentes
        n_sources = params.get("n_sources", 3)
        positions = params.get("positions", np.random.uniform(-3e-6, 3e-6, n_sources))
        fluxes    = params.get("fluxes", np.random.dirichlet(np.ones(n_sources)))
        V_complex = sum(f * np.exp(2j * np.pi * u * p) for f, p in zip(fluxes, positions))
        V_mod     = np.abs(V_complex)
        phi       = np.angle(V_complex)

    else:
        V_mod = 1.0
        phi   = 0.0

    return V_mod * np.exp(1j * phi)


def simulate_bpw34_signal(V_complex: complex,
                          baseline: float,
                          I1: float,
                          I2: float,
                          imu_angle: float,
                          snr_db: float) -> np.ndarray:
    """
    Simule le signal temporel du photodiode BPW34 :
    I(t) = I1 + I2 + 2√(I1·I2)·|V|·cos(2π·f_fringe·t + φ)
    
    La rotation IMU change la fréquence de frange via :
    f_fringe = B·cos(imu_angle) / λ · (dθ/dt)
    """
    t        = np.linspace(0, N_SAMPLES / SAMPLE_RATE, N_SAMPLES)
    V_mod    = np.abs(V_complex)
    phi      = np.angle(V_complex)

    # Fréquence de frange effective (tenant compte de l'orientation IMU)
    B_proj   = baseline * np.cos(np.radians(imu_angle))
    f_fringe = (B_proj / LAMBDA_NM) * 2e-6  # normalisé pour rester audible

    # Signal interférométrique
    I_signal = (I1 + I2
                + 2 * np.sqrt(I1 * I2) * V_mod
                * np.cos(2 * np.pi * f_fringe * t + phi))

    # Bruit réaliste
    snr_linear  = 10 ** (snr_db / 20)
    signal_rms  = np.sqrt(np.mean(I_signal ** 2))
    noise_std   = signal_rms / (snr_linear + 1e-10)
    shot        = np.random.normal(0, SHOT_NOISE * np.sqrt(I1 + I2), N_SAMPLES)
    thermal     = np.random.normal(0, THERMAL_NOISE, N_SAMPLES)
    white       = np.random.normal(0, noise_std, N_SAMPLES)

    I_noisy = I_signal + shot + thermal + white

    # Quantification ADC 12-bit
    adc_max = 2 ** ADC_BITS - 1
    I_noisy = np.clip(I_noisy, 0, None)
    I_noisy = np.round(I_noisy / (I_noisy.max() + 1e-10) * adc_max) / adc_max

    return I_noisy.astype(np.float32)


# ─────────────────────────────────────────────
# EXTRACTION DE FEATURES
# ─────────────────────────────────────────────

def extract_features(signal: np.ndarray,
                     baseline: float,
                     imu_angle: float,
                     distance: float) -> np.ndarray:
    """
    Extrait 32 features physiquement significatifs depuis le signal brut + capteurs.
    
    Features :
    - Temporelles (8) : stats signal brut
    - Fréquentielles (12) : FFT, pic dominant, largeur, harmoniques
    - Interférométriques (8) : visibilité estimée, SNR, cohérence
    - Capteurs (4) : baseline mesurée, angle IMU, distance VL53L0X, température simulée
    """
    features = []

    # ── Features temporelles ──────────────────
    features.append(float(np.mean(signal)))                        # f01 : DC moyen
    features.append(float(np.std(signal)))                         # f02 : RMS fluctuations
    features.append(float(np.max(signal) - np.min(signal)))        # f03 : amplitude frange
    features.append(float(np.percentile(signal, 75)
                         - np.percentile(signal, 25)))              # f04 : IQR
    features.append(float(np.sum(np.diff(np.sign(
        signal - np.mean(signal))) != 0)))                          # f05 : zero crossings
    features.append(float(np.mean(np.abs(np.diff(signal)))))       # f06 : variation moyenne
    features.append(float(np.max(np.abs(signal - np.mean(signal)))))  # f07 : déviation max
    features.append(float(np.sum(signal ** 2) / N_SAMPLES))        # f08 : puissance signal

    # ── Features fréquentielles ───────────────
    freqs    = fftfreq(N_SAMPLES, 1 / SAMPLE_RATE)
    spectrum = np.abs(fft(signal))[:N_SAMPLES // 2]
    freqs_p  = freqs[:N_SAMPLES // 2]

    peak_idx   = np.argmax(spectrum[1:]) + 1  # évite DC
    peak_freq  = float(freqs_p[peak_idx])
    peak_power = float(spectrum[peak_idx])
    dc_power   = float(spectrum[0])

    features.append(peak_freq)                                      # f09 : fréq dominante
    features.append(peak_power)                                     # f10 : puissance pic
    features.append(dc_power)                                       # f11 : composante DC
    features.append(float(peak_power / (dc_power + 1e-10)))        # f12 : ratio AC/DC
    features.append(float(np.sum(spectrum ** 2)))                   # f13 : énergie spectrale

    # Largeur spectrale (bandwidth du pic)
    half_max = peak_power / 2
    above    = np.where(spectrum > half_max)[0]
    bw       = float(freqs_p[above[-1]] - freqs_p[above[0]]) if len(above) > 1 else 0.0
    features.append(bw)                                             # f14 : largeur pic

    # 2ème harmonique
    harm2_idx = min(peak_idx * 2, len(spectrum) - 1)
    features.append(float(spectrum[harm2_idx]))                     # f15 : 2ème harmonique
    features.append(float(spectrum[harm2_idx] / (peak_power + 1e-10)))  # f16 : ratio h2/h1

    # Centroïde spectral
    spec_centroid = float(np.sum(freqs_p * spectrum) / (np.sum(spectrum) + 1e-10))
    features.append(spec_centroid)                                  # f17 : centroïde spectral

    # PSD via Welch
    _, psd = welch(signal, fs=SAMPLE_RATE, nperseg=256)
    features.append(float(np.max(psd)))                             # f18 : pic PSD
    features.append(float(np.mean(psd)))                            # f19 : PSD moyen
    features.append(float(np.std(psd)))                             # f20 : PSD std

    # ── Features interférométriques ───────────
    # Visibilité estimée : V = (Imax - Imin) / (Imax + Imin)
    I_max    = float(np.max(signal))
    I_min    = float(np.min(signal))
    V_est    = (I_max - I_min) / (I_max + I_min + 1e-10)
    features.append(V_est)                                          # f21 : visibilité estimée

    # SNR estimé
    ac_power   = peak_power ** 2
    noise_floor = float(np.median(spectrum) ** 2)
    snr_est    = float(10 * np.log10(ac_power / (noise_floor + 1e-10) + 1e-10))
    features.append(snr_est)                                        # f22 : SNR estimé (dB)

    # Degré de cohérence (normalisé)
    coherence = V_est ** 2
    features.append(coherence)                                      # f23 : cohérence

    # Fréquence de frange théorique
    B_proj       = baseline * np.cos(np.radians(imu_angle))
    f_th         = (B_proj / LAMBDA_NM) * 2e-6
    f_err        = abs(peak_freq - f_th) / (f_th + 1e-10)
    features.append(float(f_th))                                    # f24 : fréq théorique
    features.append(float(f_err))                                   # f25 : erreur fréq relative

    # Phase relative (depuis autocorrélation)
    autocorr   = np.correlate(signal - np.mean(signal),
                              signal - np.mean(signal), mode='full')
    autocorr   = autocorr[N_SAMPLES - 1:]
    first_zero = np.argmax(autocorr < 0) if np.any(autocorr < 0) else N_SAMPLES
    features.append(float(first_zero / SAMPLE_RATE))                # f26 : demi-période

    # Variation de visibilité (glissement temporel)
    mid      = N_SAMPLES // 2
    V_first  = (np.max(signal[:mid]) - np.min(signal[:mid])) / \
               (np.max(signal[:mid]) + np.min(signal[:mid]) + 1e-10)
    V_second = (np.max(signal[mid:]) - np.min(signal[mid:])) / \
               (np.max(signal[mid:]) + np.min(signal[mid:]) + 1e-10)
    features.append(float(V_second - V_first))                      # f27 : dérive visibilité

    # Entropie spectrale
    spec_norm = spectrum / (np.sum(spectrum) + 1e-10)
    entropy   = float(-np.sum(spec_norm * np.log(spec_norm + 1e-10)))
    features.append(entropy)                                         # f28 : entropie spectrale

    # ── Features capteurs hardware ────────────
    features.append(float(baseline))                                 # f29 : baseline B (m)
    features.append(float(imu_angle))                                # f30 : angle IMU (deg)
    features.append(float(distance))                                 # f31 : distance VL53L0X (m)
    features.append(float(B_proj / LAMBDA_NM))                       # f32 : fréq spatiale u

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────
# GÉNÉRATEUR PRINCIPAL
# ─────────────────────────────────────────────

def generate_sample(class_id: int):
    """Génère un échantillon complet pour une classe donnée."""

    # Randomisation paramètres hardware réels
    baseline   = np.random.uniform(BASELINE_MIN, BASELINE_MAX)
    imu_angle  = np.random.uniform(-30, 30)    # ±30° autour de la cible
    distance   = np.random.uniform(0.5, 3.0)   # VL53L0X : 0.5 → 3m
    I1         = np.random.uniform(0.3, 1.0)   # flux photodiode 1
    I2         = np.random.uniform(0.3, 1.0)   # flux photodiode 2
    snr_db     = np.random.uniform(10, 35)      # SNR 10-35dB

    # Paramètres spécifiques à la source
    params = {}

    if class_id == 1:  # binaire serrée
        params["theta_sep"]  = np.random.uniform(0.2e-6, 1.0e-6)
        params["flux_ratio"] = np.random.uniform(0.5, 1.0)

    elif class_id == 2:  # binaire large
        params["theta_sep"]  = np.random.uniform(2e-6, 6e-6)
        params["flux_ratio"] = np.random.uniform(0.3, 0.9)

    elif class_id == 3:  # nébuleuse
        params["theta_disk"] = np.random.uniform(1e-6, 5e-6)

    elif class_id == 4:  # disque circumstellaire
        params["theta_ring"] = np.random.uniform(0.8e-6, 3e-6)

    elif class_id == 5:  # objet compact
        params["coherence"]     = np.random.uniform(0.90, 1.0)
        params["phase_offset"]  = np.random.uniform(0, 0.1)

    elif class_id == 6:  # sources multiples
        n = np.random.randint(3, 6)
        params["n_sources"] = n
        params["positions"] = np.random.uniform(-4e-6, 4e-6, n)
        params["fluxes"]    = np.random.dirichlet(np.ones(n))

    # Calcul physique
    V_complex = compute_visibility(class_id, baseline, params)
    signal    = simulate_bpw34_signal(V_complex, baseline, I1, I2, imu_angle, snr_db)
    features  = extract_features(signal, baseline, imu_angle, distance)

    return {
        "class_id"   : class_id,
        "class_name" : CLASSES[class_id],
        "signal"     : signal,          # 1024 points BPW34
        "features"   : features,        # 32 features
        "V_mod"      : float(np.abs(V_complex)),
        "V_phase"    : float(np.angle(V_complex)),
        "baseline"   : baseline,
        "imu_angle"  : imu_angle,
        "distance"   : distance,
        "snr_db"     : snr_db,
        "params"     : {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in params.items()}
    }


# ─────────────────────────────────────────────
# GÉNÉRATION & SAUVEGARDE
# ─────────────────────────────────────────────

def generate_dataset():
    print("=" * 60)
    print("  Génération dataset interférométrique — PFE TUNSA")
    print("=" * 60)

    all_signals  = []
    all_features = []
    all_labels   = []
    metadata     = []

    for class_id in range(len(CLASSES)):
        print(f"\n[Classe {class_id}] {CLASSES[class_id]} — {N_PER_CLASS} échantillons...")

        for i in range(N_PER_CLASS):
            sample = generate_sample(class_id)
            all_signals.append(sample["signal"])
            all_features.append(sample["features"])
            all_labels.append(class_id)
            metadata.append({k: v for k, v in sample.items()
                              if k not in ("signal", "features")})

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{N_PER_CLASS} done")

    # Arrays numpy
    X_signals  = np.array(all_signals,  dtype=np.float32)   # (N, 1024)
    X_features = np.array(all_features, dtype=np.float32)   # (N, 32)
    y          = np.array(all_labels,   dtype=np.int32)      # (N,)

    # Shuffle
    idx = np.random.permutation(len(y))
    X_signals, X_features, y = X_signals[idx], X_features[idx], y[idx]

    # Train/Val/Test split : 80/10/10
    n      = len(y)
    n_train = int(0.8 * n)
    n_val   = int(0.9 * n)

    splits = {
        "train": (X_signals[:n_train],  X_features[:n_train],  y[:n_train]),
        "val":   (X_signals[n_train:n_val], X_features[n_train:n_val], y[n_train:n_val]),
        "test":  (X_signals[n_val:],    X_features[n_val:],    y[n_val:])
    }

    print("\nSauvegarde...")
    for split_name, (sig, feat, labels) in splits.items():
        np.save(OUTPUT_DIR / f"X_signals_{split_name}.npy",  sig)
        np.save(OUTPUT_DIR / f"X_features_{split_name}.npy", feat)
        np.save(OUTPUT_DIR / f"y_{split_name}.npy",          labels)
        print(f"  {split_name}: {len(labels)} échantillons — {np.bincount(labels)}")

    # Métadonnées + config
    config = {
        "classes"       : CLASSES,
        "n_total"       : int(n),
        "n_features"    : 32,
        "n_signal_pts"  : N_SAMPLES,
        "sample_rate"   : SAMPLE_RATE,
        "wavelength_nm" : LAMBDA_NM * 1e9,
        "baseline_range": [BASELINE_MIN, BASELINE_MAX],
        "splits"        : {k: int(len(v[2])) for k, v in splits.items()}
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDataset sauvegardé dans : {OUTPUT_DIR.absolute()}")
    return splits, config


# ─────────────────────────────────────────────
# VISUALISATION — 1 EXEMPLE PAR CLASSE
# ─────────────────────────────────────────────

def visualize_dataset():
    print("\nGénération des visualisations...")
    fig = plt.figure(figsize=(20, 20), facecolor="#0d0d1a")
    gs  = gridspec.GridSpec(7, 3, figure=fig, hspace=0.5, wspace=0.35)

    colors = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77",
              "#c77dff", "#ff9a3c", "#ff61a6"]

    for class_id in range(7):
        sample = generate_sample(class_id)
        signal = sample["signal"]
        t      = np.linspace(0, N_SAMPLES / SAMPLE_RATE, N_SAMPLES)
        freqs  = fftfreq(N_SAMPLES, 1 / SAMPLE_RATE)
        spectrum = np.abs(fft(signal))[:N_SAMPLES // 2]

        c = colors[class_id]

        # Signal temporel
        ax1 = fig.add_subplot(gs[class_id, 0])
        ax1.plot(t * 1000, signal, color=c, linewidth=0.8, alpha=0.9)
        ax1.set_facecolor("#111122")
        ax1.set_title(f"[{class_id}] {CLASSES[class_id]}\nSignal BPW34",
                      color="white", fontsize=7, pad=3)
        ax1.set_xlabel("t (ms)", color="#888", fontsize=6)
        ax1.tick_params(colors="#666", labelsize=6)
        for sp in ax1.spines.values():
            sp.set_color("#333")

        # Spectre FFT
        ax2 = fig.add_subplot(gs[class_id, 1])
        ax2.plot(freqs[:N_SAMPLES // 2], spectrum, color=c, linewidth=0.8)
        ax2.set_facecolor("#111122")
        ax2.set_title("Spectre FFT", color="white", fontsize=7, pad=3)
        ax2.set_xlabel("Fréquence (Hz)", color="#888", fontsize=6)
        ax2.tick_params(colors="#666", labelsize=6)
        for sp in ax2.spines.values():
            sp.set_color("#333")

        # Features radar (top 8)
        ax3 = fig.add_subplot(gs[class_id, 2])
        feat = sample["features"]
        feat_labels = ["DC", "RMS", "Ampl", "IQR", "ZCR",
                       "Var", "Dév", "Pwr"]
        feat_vals = feat[:8]
        feat_norm = (feat_vals - feat_vals.min()) / ((feat_vals.max() - feat_vals.min()) + 1e-10)
        ax3.bar(range(8), feat_norm, color=c, alpha=0.75, edgecolor="#333")
        ax3.set_xticks(range(8))
        ax3.set_xticklabels(feat_labels, fontsize=5, rotation=45, color="#aaa")
        ax3.set_facecolor("#111122")
        ax3.set_title(f"Features (V={sample['V_mod']:.3f}  φ={sample['V_phase']:.2f})",
                      color="white", fontsize=7, pad=3)
        ax3.tick_params(colors="#666", labelsize=6)
        for sp in ax3.spines.values():
            sp.set_color("#333")

    fig.suptitle("Dataset Interférométrique — PFE TUNSA\n"
                 "Simulation physique BPW34 + MPU9250 + VL53L0X",
                 color="white", fontsize=13, y=0.98)

    plt.savefig(OUTPUT_DIR / "plots" / "dataset_overview.png",
                dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Visualisation : {OUTPUT_DIR}/plots/dataset_overview.png")


# ─────────────────────────────────────────────
# STATS DATASET
# ─────────────────────────────────────────────

def print_stats(splits, config):
    print("\n" + "=" * 60)
    print("  RÉSUMÉ DATASET")
    print("=" * 60)
    print(f"  Total échantillons  : {config['n_total']}")
    print(f"  Features par sample : {config['n_features']}")
    print(f"  Points signal       : {config['n_signal_pts']}")
    print(f"  Fréquence échant.   : {config['sample_rate']} Hz")
    print(f"  Longueur d'onde     : {config['wavelength_nm']} nm")
    print(f"\n  Classes :")
    for cid, cname in config["classes"].items():
        print(f"    [{cid}] {cname}")
    print(f"\n  Splits :")
    for split, count in config["splits"].items():
        print(f"    {split:6s} : {count} échantillons")
    print("=" * 60)
    print("\n  Prochaine étape → train_classifier.py")
    print("  Prochaine étape → train_fringe_analyzer.py")
    print("  Prochaine étape → train_unet_reconstructor.py\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    splits, config = generate_dataset()
    visualize_dataset()
    print_stats(splits, config)
