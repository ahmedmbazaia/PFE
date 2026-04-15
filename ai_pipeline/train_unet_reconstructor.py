"""
train_unet_reconstructor.py — PFE TUNSA
Reconstruction d'image astronomique par interférométrie

Objectif : reconstruire la distribution d'intensité sky I(θ) depuis
           les mesures de visibilité dans le plan UV

Pipeline complet :
  1. CLEAN algorithm   — méthode classique (baseline, utilisée par VLTI/ALMA)
  2. U-Net             — raffinement deep learning de l'image CLEAN
  3. Comparaison       — FFT naïve vs CLEAN vs U-Net

Architecture : U-Net 64×64 avec skip connections
Input  : dirty image 64×64 (FT inverse des visibilités éparses)
Output : sky image reconstruite 64×64
"""

import numpy as np
import json
import os
import time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_DIR = Path("dataset")
OUTPUT_DIR  = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)

IMG_SIZE    = 64       # image sky 64×64
N_SAMPLES   = 6000     # dataset reconstruction
EPOCHS      = 40
BATCH_SIZE  = 32
LR          = 1e-3

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─────────────────────────────────────────────
# GÉNÉRATION DATASET UV → SKY
# ─────────────────────────────────────────────

def generate_sky_source(source_type: int, img_size: int = 64) -> np.ndarray:
    """
    Génère une distribution d'intensité sky I(θ) réaliste.
    Retourne une image normalisée [0,1] de taille img_size×img_size.
    """
    sky = np.zeros((img_size, img_size), dtype=np.float32)
    cx, cy = img_size // 2, img_size // 2

    if source_type == 0:  # Étoile ponctuelle
        x = cx + np.random.randint(-10, 10)
        y = cy + np.random.randint(-10, 10)
        sky[y, x] = 1.0
        # Léger étalement PSF
        from scipy.ndimage import gaussian_filter
        sky = gaussian_filter(sky, sigma=0.8)

    elif source_type == 1:  # Système binaire
        sep = np.random.randint(5, 20)
        angle = np.random.uniform(0, 2 * np.pi)
        x1 = int(cx + sep / 2 * np.cos(angle))
        y1 = int(cy + sep / 2 * np.sin(angle))
        x2 = int(cx - sep / 2 * np.cos(angle))
        y2 = int(cy - sep / 2 * np.sin(angle))
        r  = np.random.uniform(0.5, 1.0)
        for x, y, flux in [(x1, y1, 1.0), (x2, y2, r)]:
            x = np.clip(x, 1, img_size - 2)
            y = np.clip(y, 1, img_size - 2)
            sky[y, x] = flux
        from scipy.ndimage import gaussian_filter
        sky = gaussian_filter(sky, sigma=0.8)

    elif source_type == 2:  # Nébuleuse — disque gaussien
        sigma = np.random.uniform(3, 12)
        dx = np.random.randint(-8, 8)
        dy = np.random.randint(-8, 8)
        yy, xx = np.mgrid[:img_size, :img_size]
        sky = np.exp(-((xx - cx - dx) ** 2 + (yy - cy - dy) ** 2) / (2 * sigma ** 2))

    elif source_type == 3:  # Disque circumstellaire — anneau
        r_ring  = np.random.uniform(8, 20)
        w_ring  = np.random.uniform(1.5, 4)
        dx = np.random.randint(-5, 5)
        dy = np.random.randint(-5, 5)
        yy, xx = np.mgrid[:img_size, :img_size]
        r = np.sqrt((xx - cx - dx) ** 2 + (yy - cy - dy) ** 2)
        sky = np.exp(-((r - r_ring) ** 2) / (2 * w_ring ** 2))

    elif source_type == 4:  # Objet compact — point très brillant + halo faible
        sky[cy, cx] = 1.0
        sigma_halo = np.random.uniform(4, 8)
        yy, xx = np.mgrid[:img_size, :img_size]
        halo = 0.15 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma_halo ** 2))
        sky  = sky + halo
        from scipy.ndimage import gaussian_filter
        sky[cy, cx] = 0
        sky = gaussian_filter(sky, sigma=0.5) + halo
        sky[cy, cx] += 0.85

    elif source_type == 5:  # Sources multiples
        n = np.random.randint(3, 6)
        for _ in range(n):
            x = np.random.randint(8, img_size - 8)
            y = np.random.randint(8, img_size - 8)
            flux = np.random.uniform(0.3, 1.0)
            sky[y, x] = flux
        from scipy.ndimage import gaussian_filter
        sky = gaussian_filter(sky, sigma=0.8)

    elif source_type == 6:  # Galaxie spirale simplifiée
        yy, xx = np.mgrid[:img_size, :img_size]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        theta = np.arctan2(yy - cy, xx - cx)
        arm_width = 2.0
        n_arms = 2
        for i in range(n_arms):
            spiral = np.exp(-((theta - 0.3 * r + i * np.pi) % (2 * np.pi) - np.pi) ** 2 / arm_width)
            sky += spiral * np.exp(-r / 20)
        sky[cy, cx] += 0.5  # noyau central

    # Normalisation
    sky = np.clip(sky, 0, None)
    if sky.max() > 0:
        sky = sky / sky.max()
    return sky.astype(np.float32)


def simulate_uv_sampling(sky: np.ndarray,
                         n_baselines: int = None,
                         add_noise: bool = True) -> tuple:
    """
    Simule l'échantillonnage UV réaliste d'un interféromètre à 2 télescopes.

    1. Calcule la FT 2D du sky (visibilités parfaites)
    2. Échantillonne seulement les points UV correspondant aux baselines mesurées
       (rotation terrestre + orientations IMU)
    3. Reconstruit la dirty image par FT inverse (aliasing + artefacts)

    Retourne : (dirty_image, uv_mask, visibility_sampled)
    """
    img_size = sky.shape[0]
    if n_baselines is None:
        n_baselines = np.random.randint(15, 60)

    # Visibilités parfaites (FT du sky)
    V_true = fft2(sky)

    # Masque UV — simule les baselines mesurées
    uv_mask = np.zeros((img_size, img_size), dtype=bool)

    # Baselines selon rotation terrestre (simulation réaliste)
    for _ in range(n_baselines):
        hour_angle = np.random.uniform(-np.pi, np.pi)
        dec        = np.random.uniform(0.1, np.pi / 2)
        B_len      = np.random.uniform(0.05, 0.50)
        B_angle    = np.random.uniform(0, np.pi)

        # Projection baseline dans plan UV
        u = B_len * (np.sin(B_angle) * np.sin(hour_angle)
                     + np.cos(B_angle) * np.cos(hour_angle) * np.sin(dec))
        v = B_len * (np.sin(B_angle) * np.cos(hour_angle)
                     - np.cos(B_angle) * np.sin(hour_angle) * np.sin(dec))

        # Convertit en pixels
        ui = int((u / 0.5) * (img_size // 4) + img_size // 2) % img_size
        vi = int((v / 0.5) * (img_size // 4) + img_size // 2) % img_size
        uv_mask[vi, ui] = True
        uv_mask[img_size - vi - 1, img_size - ui - 1] = True  # conjugué

    # Visibilités éparses
    V_sparse = np.where(uv_mask, V_true, 0 + 0j)

    # Bruit sur les visibilités
    if add_noise:
        snr = np.random.uniform(10, 40)
        noise_level = np.abs(V_sparse).max() / (10 ** (snr / 20) + 1e-10)
        V_sparse += (np.random.normal(0, noise_level, V_sparse.shape)
                     + 1j * np.random.normal(0, noise_level, V_sparse.shape))

    # Dirty image = FT inverse des visibilités éparses
    dirty = np.real(fftshift(ifft2(ifftshift(V_sparse))))
    dirty = np.clip(dirty, 0, None)
    if dirty.max() > 0:
        dirty = dirty / dirty.max()

    return dirty.astype(np.float32), uv_mask, V_sparse


# ─────────────────────────────────────────────
# ALGORITHME CLEAN (baseline classique)
# ─────────────────────────────────────────────

def clean_algorithm(dirty: np.ndarray,
                    psf: np.ndarray = None,
                    n_iter: int = 200,
                    gain: float = 0.1,
                    threshold: float = 0.01) -> np.ndarray:
    """
    Algorithme CLEAN de Högbom (1974) — utilisé par VLTI, ALMA, VLA.

    Principe :
    1. Trouve le pixel de maximum dans la dirty image
    2. Soustrait gain × PSF centrée sur ce pixel
    3. Ajoute le composant propre (delta)
    4. Répète jusqu'à convergence

    C'est la méthode de déconvolution standard en radioastronomie.
    """
    img_size = dirty.shape[0]
    cx, cy   = img_size // 2, img_size // 2

    # PSF par défaut : dirty beam (sinc 2D)
    if psf is None:
        yy, xx = np.mgrid[:img_size, :img_size]
        psf    = np.sinc((xx - cx) / 8) * np.sinc((yy - cy) / 8)
        psf    = psf.astype(np.float32)
        psf   /= psf.max()

    residual   = dirty.copy()
    clean_comp = np.zeros_like(dirty)

    for _ in range(n_iter):
        # Trouve le maximum
        max_val = residual.max()
        if max_val < threshold:
            break
        y_max, x_max = np.unravel_index(np.argmax(residual), residual.shape)

        # Soustrait PSF
        delta = gain * max_val
        clean_comp[y_max, x_max] += delta

        # Décalage PSF
        y_shift = y_max - cy
        x_shift = x_max - cx
        psf_shifted = np.roll(np.roll(psf, y_shift, axis=0), x_shift, axis=1)
        residual   -= delta * psf_shifted
        residual    = np.clip(residual, 0, None)

    # Convolution finale avec Gaussian restoring beam
    from scipy.ndimage import gaussian_filter
    clean_image = gaussian_filter(clean_comp, sigma=1.2) + 0.05 * residual
    if clean_image.max() > 0:
        clean_image = clean_image / clean_image.max()

    return clean_image.astype(np.float32)


# ─────────────────────────────────────────────
# GÉNÉRATION DATASET U-Net
# ─────────────────────────────────────────────

def generate_unet_dataset(n_samples: int = N_SAMPLES):
    print(f"Génération dataset UV reconstruction ({n_samples} samples)...")
    n_classes = 7

    X_dirty = np.zeros((n_samples, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    y_sky   = np.zeros((n_samples, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    X_clean = np.zeros((n_samples, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    for i in range(n_samples):
        source_type = i % n_classes
        sky         = generate_sky_source(source_type, IMG_SIZE)
        dirty, _, _ = simulate_uv_sampling(sky)
        clean       = clean_algorithm(dirty)

        X_dirty[i, :, :, 0] = dirty
        y_sky[i,   :, :, 0] = sky
        X_clean[i, :, :, 0] = clean

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n_samples} done")

    # Shuffle
    idx     = np.random.permutation(n_samples)
    X_dirty = X_dirty[idx]
    y_sky   = y_sky[idx]
    X_clean = X_clean[idx]

    # Split 80/10/10
    n_tr = int(0.8 * n_samples)
    n_va = int(0.9 * n_samples)

    splits = {
        "train": (X_dirty[:n_tr],       X_clean[:n_tr],       y_sky[:n_tr]),
        "val":   (X_dirty[n_tr:n_va],   X_clean[n_tr:n_va],   y_sky[n_tr:n_va]),
        "test":  (X_dirty[n_va:],       X_clean[n_va:],       y_sky[n_va:])
    }

    for name, (xd, xc, ys) in splits.items():
        np.save(DATASET_DIR / f"unet_dirty_{name}.npy",  xd)
        np.save(DATASET_DIR / f"unet_clean_{name}.npy",  xc)
        np.save(DATASET_DIR / f"unet_sky_{name}.npy",    ys)
        print(f"  {name}: {len(ys)} samples")

    print("Dataset UV sauvegardé.\n")
    return splits


# ─────────────────────────────────────────────
# ARCHITECTURE U-Net
# ─────────────────────────────────────────────

def conv_block(x, filters, name):
    """Bloc Conv2D × 2 avec BN + ReLU."""
    x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                      name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                      name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    return x


def build_unet(img_size: int = 64) -> keras.Model:
    """
    U-Net pour reconstruction d'images interférométriques.

    Input  : dirty image 64×64×1 (FT inverse des visibilités éparses)
    Output : sky image reconstruite 64×64×1

    Architecture :
    Encoder : 64→128→256→512 (avec MaxPool)
    Bottleneck : 1024
    Decoder : 512→256→128→64 (avec UpSampling + skip connections)

    Les skip connections transmettent les détails fins de l'encoder
    au decoder → préserve les sources ponctuelles.
    """
    inp = keras.Input(shape=(img_size, img_size, 1), name="dirty_image")

    # ── Encoder ──────────────────────────────
    e1 = conv_block(inp, 64,  "enc1")          # 64×64×64
    p1 = layers.MaxPooling2D(2)(e1)            # 32×32×64

    e2 = conv_block(p1,  128, "enc2")          # 32×32×128
    p2 = layers.MaxPooling2D(2)(e2)            # 16×16×128

    e3 = conv_block(p2,  256, "enc3")          # 16×16×256
    p3 = layers.MaxPooling2D(2)(e3)            # 8×8×256

    e4 = conv_block(p3,  512, "enc4")          # 8×8×512
    p4 = layers.MaxPooling2D(2)(e4)            # 4×4×512

    # ── Bottleneck ───────────────────────────
    b  = conv_block(p4, 1024, "bottleneck")    # 4×4×1024
    b  = layers.Dropout(0.3)(b)

    # ── Decoder ──────────────────────────────
    u4 = layers.UpSampling2D(2)(b)             # 8×8×1024
    u4 = layers.Concatenate()([u4, e4])        # skip → 8×8×1536
    d4 = conv_block(u4, 512, "dec4")           # 8×8×512

    u3 = layers.UpSampling2D(2)(d4)            # 16×16×512
    u3 = layers.Concatenate()([u3, e3])        # skip → 16×16×768
    d3 = conv_block(u3, 256, "dec3")           # 16×16×256

    u2 = layers.UpSampling2D(2)(d3)            # 32×32×256
    u2 = layers.Concatenate()([u2, e2])        # skip → 32×32×384
    d2 = conv_block(u2, 128, "dec2")           # 32×32×128

    u1 = layers.UpSampling2D(2)(d2)            # 64×64×128
    u1 = layers.Concatenate()([u1, e1])        # skip → 64×64×192
    d1 = conv_block(u1, 64,  "dec1")           # 64×64×64

    # Sortie
    out = layers.Conv2D(1, 1, activation="sigmoid", name="sky_output")(d1)

    model = keras.Model(inputs=inp, outputs=out, name="UNet_SkyReconstructor")
    return model


# ─────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────

def train_unet(splits):
    print("=" * 55)
    print("  U-Net SKY RECONSTRUCTOR")
    print("  Dirty image 64×64  →  Sky image 64×64")
    print("=" * 55)

    X_train, _, y_train = splits["train"]
    X_val,   _, y_val   = splits["val"]
    X_test,  X_clean_test, y_test = splits["test"]

    print(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")

    model = build_unet(IMG_SIZE)
    model.summary()
    print(f"\n  Paramètres : {model.count_params():,}")

    # Loss combinée : MSE + SSIM structurel
    def combined_loss(y_true, y_pred):
        mse  = tf.reduce_mean(tf.square(y_true - y_pred))
        ssim = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        return 0.7 * mse + 0.3 * ssim

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=combined_loss,
        metrics=["mae", keras.metrics.RootMeanSquaredError(name="rmse")]
    )

    cbs = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                      monitor="val_loss", verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4,
                                          monitor="val_loss", verbose=1),
        keras.callbacks.ModelCheckpoint(str(OUTPUT_DIR / "unet_best.keras"),
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

    return model, history.history, X_test, X_clean_test, y_test


# ─────────────────────────────────────────────
# ÉVALUATION ET COMPARAISON
# ─────────────────────────────────────────────

def evaluate_and_compare(model, X_test, X_clean_test, y_test):
    print("\n  Évaluation comparative...")

    y_pred  = model.predict(X_test, verbose=0)

    # Métriques sur tout le test set
    mse_dirty = mean_squared_error(y_test.flatten(), X_test.flatten())
    mse_clean = mean_squared_error(y_test.flatten(), X_clean_test.flatten())
    mse_unet  = mean_squared_error(y_test.flatten(), y_pred.flatten())

    # SSIM moyen (sur échantillon)
    n_eval = min(100, len(y_test))

    def mean_ssim(a, b):
        vals = []
        for i in range(n_eval):
            img_a = a[i, :, :, 0]
            img_b = b[i, :, :, 0]
            # Simple SSIM approximation
            mu_a, mu_b   = img_a.mean(), img_b.mean()
            sig_a, sig_b = img_a.std(), img_b.std()
            sig_ab       = ((img_a - mu_a) * (img_b - mu_b)).mean()
            c1, c2       = 0.01 ** 2, 0.03 ** 2
            ssim = ((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) / \
                   ((mu_a ** 2 + mu_b ** 2 + c1) * (sig_a ** 2 + sig_b ** 2 + c2))
            vals.append(ssim)
        return float(np.mean(vals))

    ssim_dirty = mean_ssim(y_test, X_test)
    ssim_clean = mean_ssim(y_test, X_clean_test)
    ssim_unet  = mean_ssim(y_test, y_pred)

    print(f"\n  ┌───────────────────────────────────────────┐")
    print(f"  │  Comparaison reconstruction               │")
    print(f"  ├─────────────────┬──────────┬─────────────┤")
    print(f"  │  Méthode        │   MSE    │    SSIM     │")
    print(f"  ├─────────────────┼──────────┼─────────────┤")
    print(f"  │  Dirty (FT⁻¹)  │ {mse_dirty:.5f}  │  {ssim_dirty:.4f}      │")
    print(f"  │  CLEAN          │ {mse_clean:.5f}  │  {ssim_clean:.4f}      │")
    print(f"  │  U-Net          │ {mse_unet:.5f}  │  {ssim_unet:.4f}      │")
    print(f"  └─────────────────┴──────────┴─────────────┘")

    improvement = ((mse_clean - mse_unet) / (mse_clean + 1e-10)) * 100
    print(f"\n  Amélioration U-Net vs CLEAN : {improvement:.1f}%")

    return {
        "y_pred"     : y_pred,
        "mse_dirty"  : mse_dirty,
        "mse_clean"  : mse_clean,
        "mse_unet"   : mse_unet,
        "ssim_dirty" : ssim_dirty,
        "ssim_clean" : ssim_clean,
        "ssim_unet"  : ssim_unet,
        "improvement": improvement
    }


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────

def plot_results(history, X_test, X_clean_test, y_test, eval_results):
    print("\n  Génération visualisations...")

    fig = plt.figure(figsize=(24, 20), facecolor="#0d0d1a")
    gs  = gridspec.GridSpec(4, 6, figure=fig, hspace=0.45, wspace=0.3)

    # ── 3 exemples : Dirty | CLEAN | U-Net | Vérité ──
    for row in range(3):
        idx = np.random.randint(len(y_test))
        dirty  = X_test[idx, :, :, 0]
        clean  = X_clean_test[idx, :, :, 0]
        pred   = eval_results["y_pred"][idx, :, :, 0]
        truth  = y_test[idx, :, :, 0]

        titles = ["Dirty Image\n(FT⁻¹ éparses)", "CLEAN\n(Högbom 1974)",
                  "U-Net\n(deep learning)", "Vérité terrain\n(sky réel)"]
        imgs   = [dirty, clean, pred, truth]
        cmaps  = ["hot", "inferno", "plasma", "gray"]

        for col, (title, img, cmap) in enumerate(zip(titles, imgs, cmaps)):
            ax = fig.add_subplot(gs[row, col + 1])
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1, aspect="equal")
            ax.set_title(title if row == 0 else "", color="white", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color("#444")
            if col == 0:
                ax.set_ylabel(f"Exemple {row+1}", color="#aaa", fontsize=8)

        # Profil 1D (coupe centrale)
        ax_p = fig.add_subplot(gs[row, 0])
        mid  = IMG_SIZE // 2
        ax_p.plot(dirty[mid],  color="#888",   lw=1.2, alpha=0.7, label="Dirty")
        ax_p.plot(clean[mid],  color="#ffd93d", lw=1.5, alpha=0.8, label="CLEAN")
        ax_p.plot(pred[mid],   color="#00d4ff", lw=1.5, alpha=0.9, label="U-Net")
        ax_p.plot(truth[mid],  color="white",  lw=1.0, ls="--",    label="Vérité")
        ax_p.set_facecolor("#111122")
        ax_p.set_title("Profil central", color="white", fontsize=7)
        ax_p.tick_params(colors="#666", labelsize=6)
        for sp in ax_p.spines.values(): sp.set_color("#333")
        if row == 0:
            ax_p.legend(fontsize=6, facecolor="#1a1a2e", labelcolor="white")

    # ── Learning curves ───────────────────────
    ax_lc = fig.add_subplot(gs[3, :3])
    ep = range(1, len(history["loss"]) + 1)
    ax_lc.plot(ep, history["loss"],     color="#00d4ff", lw=2, label="Train loss")
    ax_lc.plot(ep, history["val_loss"], color="#ff6b6b", lw=2, ls="--", label="Val loss")
    ax_lc.set_facecolor("#111122")
    ax_lc.set_title("U-Net — Courbes d'apprentissage (MSE + SSIM)", color="white", fontsize=10)
    ax_lc.set_xlabel("Époque", color="#aaa", fontsize=9)
    ax_lc.set_ylabel("Loss combinée", color="#aaa", fontsize=9)
    ax_lc.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
    ax_lc.tick_params(colors="#888", labelsize=8)
    for sp in ax_lc.spines.values(): sp.set_color("#333")

    # ── Barres comparaison ────────────────────
    ax_bar = fig.add_subplot(gs[3, 3:])
    methods = ["Dirty\n(FT⁻¹)", "CLEAN\n(classique)", "U-Net\n(IA)"]
    mse_vals  = [eval_results["mse_dirty"],  eval_results["mse_clean"],  eval_results["mse_unet"]]
    ssim_vals = [eval_results["ssim_dirty"], eval_results["ssim_clean"], eval_results["ssim_unet"]]
    x = np.arange(3)
    w = 0.35
    colors_mse  = ["#888", "#ffd93d", "#00d4ff"]
    colors_ssim = ["#666", "#ff9a3c", "#6bcb77"]
    b1 = ax_bar.bar(x - w/2, mse_vals,  w, label="MSE (↓)",  color=colors_mse,  alpha=0.85, edgecolor="#333")
    ax2 = ax_bar.twinx()
    b2 = ax2.bar(x + w/2, ssim_vals, w, label="SSIM (↑)", color=colors_ssim, alpha=0.85, edgecolor="#333")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(methods, color="#ccc", fontsize=9)
    ax_bar.set_facecolor("#111122")
    ax_bar.set_title(f"Comparaison Dirty vs CLEAN vs U-Net\nAmélioration U-Net: {eval_results['improvement']:.1f}%",
                     color="white", fontsize=10)
    ax_bar.set_ylabel("MSE (↓ meilleur)", color="#00d4ff", fontsize=9)
    ax2.set_ylabel("SSIM (↑ meilleur)", color="#6bcb77", fontsize=9)
    ax_bar.tick_params(colors="#888", labelsize=8)
    ax2.tick_params(colors="#888", labelsize=8)
    for sp in ax_bar.spines.values(): sp.set_color("#333")
    lines = [b1, b2]
    ax_bar.legend(lines, ["MSE (↓)", "SSIM (↑)"], fontsize=8,
                  facecolor="#1a1a2e", labelcolor="white")

    fig.suptitle(
        "U-Net Sky Reconstructor — Reconstruction Image Interférométrique\n"
        "PFE TUNSA | Dirty Image → CLEAN (Högbom) → U-Net Refinement",
        color="white", fontsize=12, y=0.99
    )

    out = OUTPUT_DIR / "plots" / "unet_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  Visualisation : {out}")


# ─────────────────────────────────────────────
# EXPORT TFLite
# ─────────────────────────────────────────────

def export_tflite(model):
    print("\n  Export TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # U-Net n'a pas de LSTM → export standard
    tflite_model = converter.convert()
    path = OUTPUT_DIR / "unet_reconstructor.tflite"
    with open(path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Sauvegardé : {path} ({size_mb:.1f} MB)")
    return path


# ─────────────────────────────────────────────
# INFÉRENCE TEMPS RÉEL
# ─────────────────────────────────────────────

INFERENCE_CODE = '''
# ─── Intégration dans station/main.py ────────────────────────────────────────
# Pipeline complet : visibilités → dirty image → CLEAN → U-Net → sky map

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import tensorflow as tf  # ou tflite_runtime

class SkyReconstructor:
    """
    Reconstruit l\'image sky depuis les mesures de visibilité.
    Utilisé par la station centrale (RPi 4 / RPi 2).
    """
    def __init__(self, unet_path: str = "models/unet_reconstructor.tflite",
                       img_size: int = 64):
        self.img_size    = img_size
        self.interpreter = tf.lite.Interpreter(model_path=unet_path)
        self.interpreter.allocate_tensors()
        self.input_idx   = self.interpreter.get_input_details()[0]["index"]
        self.output_idx  = self.interpreter.get_output_details()[0]["index"]
        print(f"SkyReconstructor chargé depuis {unet_path}")

    def visibilities_to_dirty(self, visibilities: dict) -> np.ndarray:
        """
        Construit la dirty image depuis les mesures de visibilité.
        visibilities : {(u,v) : V_complex}
        """
        V_grid = np.zeros((self.img_size, self.img_size), dtype=complex)
        for (u, v), V in visibilities.items():
            ui = int(u % self.img_size)
            vi = int(v % self.img_size)
            V_grid[vi, ui] = V
            V_grid[self.img_size - vi - 1, self.img_size - ui - 1] = np.conj(V)

        dirty = np.real(fftshift(ifft2(ifftshift(V_grid))))
        dirty = np.clip(dirty, 0, None)
        if dirty.max() > 0:
            dirty = dirty / dirty.max()
        return dirty.astype(np.float32)

    def reconstruct(self, dirty_image: np.ndarray) -> np.ndarray:
        """
        Reconstruit le sky depuis la dirty image via U-Net TFLite.
        """
        inp = dirty_image.reshape(1, self.img_size, self.img_size, 1)
        self.interpreter.set_tensor(self.input_idx, inp)
        self.interpreter.invoke()
        sky = self.interpreter.get_tensor(self.output_idx)[0, :, :, 0]
        return sky

# Usage :
# reconstructor = SkyReconstructor("models/unet_reconstructor.tflite")
# dirty = reconstructor.visibilities_to_dirty(visibility_measurements)
# sky   = reconstructor.reconstruct(dirty)
# # sky est une image 64×64 normalisée [0,1]
'''


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  U-Net Sky Reconstructor — PFE TUNSA")
    print("  Reconstruction interférométrique : UV → Sky")
    print("=" * 55 + "\n")

    # Dataset
    dirty_path = DATASET_DIR / "unet_dirty_train.npy"
    if not dirty_path.exists():
        splits = generate_unet_dataset(N_SAMPLES)
    else:
        print("Dataset UV existant chargé.")
        splits = {}
        for name in ["train", "val", "test"]:
            splits[name] = (
                np.load(DATASET_DIR / f"unet_dirty_{name}.npy"),
                np.load(DATASET_DIR / f"unet_clean_{name}.npy"),
                np.load(DATASET_DIR / f"unet_sky_{name}.npy")
            )
            print(f"  {name}: {len(splits[name][2])} samples")

    # Entraînement
    model, history, X_test, X_clean_test, y_test = train_unet(splits)

    # Évaluation comparative
    eval_results = evaluate_and_compare(model, X_test, X_clean_test, y_test)

    # Visualisations
    plot_results(history, X_test, X_clean_test, y_test, eval_results)

    # Export TFLite
    export_tflite(model)

    # Code intégration
    with open(OUTPUT_DIR / "sky_reconstructor_integration.py", "w") as f:
        f.write(INFERENCE_CODE)

    # Résumé JSON
    summary = {
        "architecture" : "U-Net 64×64 (Enc 64→128→256→512→1024, Dec sym.)",
        "input"        : "dirty image 64×64×1 (FT⁻¹ visibilités éparses)",
        "output"       : "sky image reconstruite 64×64×1",
        "loss"         : "0.7*MSE + 0.3*(1-SSIM)",
        "clean_baseline": "Algorithme CLEAN de Högbom (1974)",
        "metrics"      : {
            "mse_dirty"  : round(eval_results["mse_dirty"],  6),
            "mse_clean"  : round(eval_results["mse_clean"],  6),
            "mse_unet"   : round(eval_results["mse_unet"],   6),
            "ssim_unet"  : round(eval_results["ssim_unet"],  4),
            "improvement": round(eval_results["improvement"], 1)
        },
        "tflite": "models/unet_reconstructor.tflite"
    }
    with open(OUTPUT_DIR / "unet_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("  U-Net Reconstructor entraîné ✓")
    print("\n  Pipeline AI complet :")
    print("  ① dataset_generator.py     — données synthétiques")
    print("  ② train_classifier.py      — RF + 1D CNN (7 classes)")
    print("  ③ train_fringe_analyzer.py — CNN+BiLSTM (|V|, φ)")
    print("  ④ train_unet_reconstructor.py — U-Net (sky image)")
    print("=" * 55 + "\n")
