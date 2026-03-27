#!/usr/bin/env python3
"""
Sky Reconstruction Module
=========================
Takes detected stars with visibility and phase data and produces
a 2D reconstructed sky map. Saves the result as a PNG image.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")

# ─── Reconstruction parameters ───────────────────────────────
SKY_MAP_SIZE = 256       # output image size in pixels
COLORMAP = "hot"         # colormap for intensity display


def reconstruct_sky(star_results, size=SKY_MAP_SIZE):
    """
    Build a 2D sky brightness map from detected stars with visibility.

    Each star is placed as a Gaussian weighted by its visibility.
    Higher visibility → sharper, brighter point in the reconstruction.

    Args:
        star_results: list of dicts from visibility.compute_batch()
                      (must have x, y, visibility, detection_intensity)
        size: output image dimension (pixels)

    Returns:
        2D NumPy array (sky map), normalized to [0, 1].
    """
    sky = np.zeros((size, size), dtype=np.float64)

    if not star_results:
        return sky

    x_coords = np.arange(size)
    y_coords = np.arange(size)
    xx, yy = np.meshgrid(x_coords, y_coords)

    for star in star_results:
        cx = star["x"]
        cy = star["y"]
        vis = star["visibility"]
        intensity = star["detection_intensity"]

        # Visibility controls the sharpness: high V → tight Gaussian
        # Low visibility → broader, dimmer blob (unresolved)
        sigma = max(2.0, 15.0 * (1.0 - vis))  # 2–15 px
        amplitude = intensity * (0.3 + 0.7 * vis)  # floor at 30%

        gaussian = amplitude * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)
        )
        sky += gaussian

    # Normalize to [0, 1]
    sky_max = sky.max()
    if sky_max > 0:
        sky /= sky_max

    return sky


def save_sky_map(sky, star_results=None, filename=None):
    """
    Render and save the sky map as a PNG with annotations.

    Args:
        sky: 2D array from reconstruct_sky()
        star_results: optional list of star dicts for annotation
        filename: output path (auto-generated if None)

    Returns:
        Path to the saved image.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(IMAGE_DIR, f"reconstruction_{ts}.png")

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.imshow(sky, cmap=COLORMAP, origin="lower", vmin=0, vmax=1)
    ax.set_title("Reconstructed Sky Map", fontsize=12)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    # Annotate detected stars
    if star_results:
        for i, star in enumerate(star_results):
            cx = star["x"]
            cy = star["y"]
            vis = star["visibility"]

            # Circle size proportional to visibility
            radius = 8 + 12 * vis
            circle = Circle((cx, cy), radius, fill=False,
                            edgecolor="cyan", linewidth=1.5, linestyle="--")
            ax.add_patch(circle)
            ax.annotate(
                f"V={vis:.2f}",
                (cx, cy), textcoords="offset points",
                xytext=(10, 10), fontsize=8, color="cyan",
            )

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    print(f"[reconstruct] Saved sky map → {filename}")
    return filename


def reconstruct_and_save(star_results, filename=None):
    """
    Full pipeline: reconstruct sky from star results, save image.
    Returns (sky_array, image_path).
    """
    sky = reconstruct_sky(star_results)
    path = save_sky_map(sky, star_results, filename)
    return sky, path


# ─── Standalone usage ────────────────────────────────────────
if __name__ == "__main__":
    # Demo with fake star data
    demo_stars = [
        {"x": 80, "y": 120, "visibility": 0.9, "detection_intensity": 0.95,
         "phase_rad": 1.2, "phase_deg": 68.75, "baseline_mm": 300,
         "t1_intensity": 3000, "t2_intensity": 2800},
        {"x": 180, "y": 60, "visibility": 0.5, "detection_intensity": 0.6,
         "phase_rad": 2.5, "phase_deg": 143.24, "baseline_mm": 300,
         "t1_intensity": 2000, "t2_intensity": 1000},
        {"x": 150, "y": 200, "visibility": 0.2, "detection_intensity": 0.3,
         "phase_rad": 0.8, "phase_deg": 45.84, "baseline_mm": 300,
         "t1_intensity": 1500, "t2_intensity": 1100},
    ]
    _, path = reconstruct_and_save(demo_stars)
    print(f"Demo reconstruction: {path}")
