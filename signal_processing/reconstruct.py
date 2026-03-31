"""
Sky reconstruction module.
Uses visibility and phase data to reconstruct a sky brightness map
via inverse Fourier transform (simplified aperture synthesis).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VISIBILITY_CSV = os.path.join(DATA_DIR, "visibility.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "plots")
RECONSTRUCTION_IMG = os.path.join(OUTPUT_DIR, "sky_reconstruction.png")

GRID_SIZE = 128


def reconstruct_sky(visibility_df, grid_size=GRID_SIZE):
    """
    Reconstruct a sky brightness map from visibility measurements.

    Uses the Van Cittert-Zernike theorem: the sky brightness distribution
    is the inverse Fourier transform of the complex visibility function.
    """
    uv_grid = np.zeros((grid_size, grid_size), dtype=np.complex128)
    center = grid_size // 2

    for _, row in visibility_df.iterrows():
        freq = row["spatial_freq"]
        angle_rad = np.radians(row["fringe_angle_deg"])
        vis = row["visibility"]
        phase = row["phase_rad"]

        # Map to UV coordinates
        u = int(round(freq * np.cos(angle_rad) * grid_size * 4))
        v = int(round(freq * np.sin(angle_rad) * grid_size * 4))

        complex_vis = vis * np.exp(1j * phase)

        # Place visibility and its conjugate (Hermitian symmetry)
        u_idx = center + u
        v_idx = center + v
        if 0 <= u_idx < grid_size and 0 <= v_idx < grid_size:
            uv_grid[v_idx, u_idx] = complex_vis
            # Conjugate
            u_conj = center - u
            v_conj = center - v
            if 0 <= u_conj < grid_size and 0 <= v_conj < grid_size:
                uv_grid[v_conj, u_conj] = np.conj(complex_vis)

    # Inverse FFT to get sky image
    sky = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(uv_grid)))
    sky_real = np.abs(sky)

    # Normalize to 0-255
    if sky_real.max() > 0:
        sky_norm = (sky_real / sky_real.max() * 255).astype(np.uint8)
    else:
        sky_norm = np.zeros_like(sky_real, dtype=np.uint8)

    return sky_norm, uv_grid


def main():
    print("=== Sky Reconstruction ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(VISIBILITY_CSV):
        print(f"Visibility data not found at {VISIBILITY_CSV}")
        print("Run visibility.py first.")
        return

    df = pd.read_csv(VISIBILITY_CSV)
    print(f"Loaded {len(df)} visibility measurements")

    sky_img, uv_grid = reconstruct_sky(df)

    # Create figure with UV coverage and reconstruction
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # UV coverage
    uv_mag = np.abs(uv_grid)
    axes[0].imshow(np.log1p(uv_mag), cmap="hot", origin="lower")
    axes[0].set_title("UV Coverage (log scale)")
    axes[0].axis("off")

    # Dirty beam (PSF)
    psf_grid = (np.abs(uv_grid) > 0).astype(np.complex128)
    psf = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(psf_grid))))
    if psf.max() > 0:
        psf = psf / psf.max()
    axes[1].imshow(psf, cmap="inferno", origin="lower")
    axes[1].set_title("Dirty Beam (PSF)")
    axes[1].axis("off")

    # Reconstructed sky
    axes[2].imshow(sky_img, cmap="gray", origin="lower")
    axes[2].set_title("Reconstructed Sky")
    axes[2].axis("off")

    fig.suptitle("Aperture Synthesis Reconstruction", fontsize=13)
    fig.tight_layout()
    fig.savefig(RECONSTRUCTION_IMG, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Reconstruction saved to {os.path.abspath(RECONSTRUCTION_IMG)}")
    print(f"Sky image shape: {sky_img.shape}, max pixel: {sky_img.max()}")


if __name__ == "__main__":
    main()
