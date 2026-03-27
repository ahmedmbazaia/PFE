#!/usr/bin/env python3
"""
Synthetic Data Generator
========================
Generates fake interferometric fringe images with 2-3 simulated point
sources (stars) at random positions. Adds Gaussian noise and varies
brightness. Saves images to data/images/ and metadata to a CSV.
"""

import csv
import os
import random
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
METADATA_CSV = os.path.join(DATA_DIR, "synthetic_metadata.csv")

# ─── Image parameters ────────────────────────────────────────
IMAGE_SIZE = 256          # pixels (square)
FRINGE_FREQ_RANGE = (5, 20)    # cycles across image
NOISE_STD_RANGE = (0.02, 0.15)
NUM_STARS_RANGE = (2, 3)
STAR_BRIGHTNESS_RANGE = (0.3, 1.0)
STAR_FWHM = 5            # pixels — Gaussian point spread


def generate_fringe_pattern(size, frequency, angle_deg=0):
    """
    Generate a 2D sinusoidal fringe pattern.
    Simulates the interference pattern from a two-telescope baseline.
    """
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)

    angle_rad = np.radians(angle_deg)
    phase = 2 * np.pi * frequency * (xx * np.cos(angle_rad) + yy * np.sin(angle_rad))
    return 0.5 * (1 + np.cos(phase))


def add_point_source(image, cx, cy, brightness, fwhm=STAR_FWHM):
    """
    Add a Gaussian point source (star) to the image at (cx, cy).
    """
    size = image.shape[0]
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)

    sigma = fwhm / 2.355  # FWHM to sigma
    gaussian = brightness * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return image + gaussian


def generate_image(num_stars=None):
    """
    Generate a single synthetic fringe image with point sources.
    Returns the image array and a list of star metadata dicts.
    """
    size = IMAGE_SIZE
    if num_stars is None:
        num_stars = random.randint(*NUM_STARS_RANGE)

    # Base fringe pattern
    freq = random.uniform(*FRINGE_FREQ_RANGE)
    angle = random.uniform(0, 180)
    image = generate_fringe_pattern(size, freq, angle) * 0.3  # dim background fringes

    # Add point sources
    stars = []
    margin = 20  # keep stars away from edges
    for i in range(num_stars):
        cx = random.randint(margin, size - margin)
        cy = random.randint(margin, size - margin)
        brightness = random.uniform(*STAR_BRIGHTNESS_RANGE)

        image = add_point_source(image, cx, cy, brightness)
        stars.append({
            "star_id": i,
            "x": cx,
            "y": cy,
            "brightness": round(brightness, 4),
        })

    # Add Gaussian noise
    noise_std = random.uniform(*NOISE_STD_RANGE)
    noise = np.random.normal(0, noise_std, (size, size))
    image = np.clip(image + noise, 0, 1)

    metadata = {
        "fringe_freq": round(freq, 2),
        "fringe_angle": round(angle, 2),
        "noise_std": round(noise_std, 4),
        "num_stars": num_stars,
        "stars": stars,
    }
    return image, metadata


def save_image(image, filename):
    """Save a grayscale image as PNG using Matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=64)
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    fig.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def generate_dataset(n_images=50):
    """
    Generate a full synthetic dataset: n_images fringe images + metadata CSV.
    Returns the path to the metadata CSV.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)

    rows = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(n_images):
        image, meta = generate_image()

        # Save image
        fname = f"synth_{timestamp}_{i:04d}.png"
        fpath = os.path.join(IMAGE_DIR, fname)
        save_image(image, fpath)

        # Flatten star data into CSV row
        for star in meta["stars"]:
            rows.append({
                "image_file": fname,
                "image_index": i,
                "fringe_freq": meta["fringe_freq"],
                "fringe_angle": meta["fringe_angle"],
                "noise_std": meta["noise_std"],
                "num_stars": meta["num_stars"],
                "star_id": star["star_id"],
                "star_x": star["x"],
                "star_y": star["y"],
                "star_brightness": star["brightness"],
            })

    # Write metadata CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {n_images} images → {IMAGE_DIR}")
    print(f"Metadata → {METADATA_CSV} ({len(rows)} star entries)")
    return METADATA_CSV


# ─── Standalone usage ────────────────────────────────────────
if __name__ == "__main__":
    generate_dataset(50)
