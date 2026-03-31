"""
Fringe visibility and phase computation module.
Analyzes fringe pattern images to extract visibility (contrast)
and phase information for each baseline.
"""

import os
import glob
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_CSV = os.path.join(DATA_DIR, "visibility.csv")
OUTPUT_PLOT_DIR = os.path.join(DATA_DIR, "plots")


def compute_visibility(image):
    """
    Compute fringe visibility and phase from a fringe image.

    Visibility = (I_max - I_min) / (I_max + I_min)
    Phase is estimated via FFT peak detection along dominant frequency.
    """
    img = image.astype(np.float64)

    # Visibility from intensity statistics
    i_max = np.percentile(img, 99)
    i_min = np.percentile(img, 1)
    if (i_max + i_min) > 0:
        visibility = (i_max - i_min) / (i_max + i_min)
    else:
        visibility = 0.0

    # Phase and spatial frequency from 2D FFT
    fft2 = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft2)
    magnitude = np.abs(fft_shifted)

    cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    # Zero out DC component
    magnitude[cy-2:cy+3, cx-2:cx+3] = 0

    peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    freq_y = (peak_idx[0] - cy) / magnitude.shape[0]
    freq_x = (peak_idx[1] - cx) / magnitude.shape[1]
    spatial_freq = np.sqrt(freq_x**2 + freq_y**2)
    angle = np.arctan2(freq_y, freq_x)

    phase = np.angle(fft_shifted[peak_idx[0], peak_idx[1]])

    return {
        "visibility": round(float(visibility), 4),
        "phase_rad": round(float(phase), 4),
        "spatial_freq": round(float(spatial_freq), 6),
        "fringe_angle_deg": round(float(np.degrees(angle)), 2),
    }


def process_fringe_images():
    """Process all fringe images and return a DataFrame of measurements."""
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    fringe_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "fringe_*.png")))
    if not fringe_files:
        print("No fringe images found. Run synthetic_data.py first.")
        return pd.DataFrame()

    results = []
    for fpath in fringe_files:
        fname = os.path.basename(fpath)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        meas = compute_visibility(img)
        meas["file"] = fname
        results.append(meas)
        print(f"  {fname}: V={meas['visibility']:.3f}, phase={meas['phase_rad']:.3f} rad")

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Visibility data saved to {OUTPUT_CSV}")

        # Plot visibility vs spatial frequency
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(range(len(df)), df["visibility"], color="steelblue")
        axes[0].set_xlabel("Baseline index")
        axes[0].set_ylabel("Visibility")
        axes[0].set_title("Fringe Visibility per Baseline")

        axes[1].scatter(df["spatial_freq"], df["visibility"], c=df["phase_rad"], cmap="hsv", s=60)
        axes[1].set_xlabel("Spatial Frequency")
        axes[1].set_ylabel("Visibility")
        axes[1].set_title("Visibility vs Frequency (color=phase)")
        plt.colorbar(axes[1].collections[0], ax=axes[1], label="Phase (rad)")

        fig.tight_layout()
        plot_path = os.path.join(OUTPUT_PLOT_DIR, "visibility_summary.png")
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Summary plot saved to {plot_path}")

    return df


def main():
    print("=== Fringe Visibility & Phase ===")
    df = process_fringe_images()
    if not df.empty:
        print(f"\nSummary:\n{df[['visibility', 'phase_rad', 'spatial_freq']].describe()}")


if __name__ == "__main__":
    main()
