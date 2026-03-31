"""
Synthetic data generator for optical interferometry pipeline.
Generates fake star field images and fringe pattern images for testing
without any hardware connected.
"""

import os
import numpy as np
import cv2

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")
NUM_STAR_IMAGES = 10
NUM_FRINGE_IMAGES = 10
IMG_SIZE = 256


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_star_image(index, img_size=IMG_SIZE):
    """Generate a synthetic star field image with random point sources."""
    rng = np.random.RandomState(index)
    img = rng.normal(loc=10, scale=3, size=(img_size, img_size)).clip(0, 255).astype(np.float64)

    num_stars = rng.randint(3, 12)
    for _ in range(num_stars):
        x = rng.randint(20, img_size - 20)
        y = rng.randint(20, img_size - 20)
        brightness = rng.uniform(150, 255)
        sigma = rng.uniform(1.5, 4.0)

        yy, xx = np.mgrid[0:img_size, 0:img_size]
        gaussian = brightness * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        img += gaussian

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def generate_fringe_image(index, img_size=IMG_SIZE):
    """Generate a synthetic fringe (interference) pattern image."""
    rng = np.random.RandomState(1000 + index)

    freq = rng.uniform(5, 20)
    angle = rng.uniform(0, np.pi)
    visibility = rng.uniform(0.3, 1.0)
    phase = rng.uniform(0, 2 * np.pi)

    yy, xx = np.mgrid[0:img_size, 0:img_size]
    spatial = xx * np.cos(angle) + yy * np.sin(angle)
    fringe = 128 + 127 * visibility * np.cos(2 * np.pi * freq * spatial / img_size + phase)

    noise = rng.normal(0, 8, (img_size, img_size))
    fringe += noise

    fringe = np.clip(fringe, 0, 255).astype(np.uint8)
    return fringe


def main():
    ensure_output_dir()
    print(f"Generating synthetic images in: {os.path.abspath(OUTPUT_DIR)}")

    for i in range(NUM_STAR_IMAGES):
        img = generate_star_image(i)
        path = os.path.join(OUTPUT_DIR, f"star_{i:03d}.png")
        cv2.imwrite(path, img)
        print(f"  [star] {path}")

    for i in range(NUM_FRINGE_IMAGES):
        img = generate_fringe_image(i)
        path = os.path.join(OUTPUT_DIR, f"fringe_{i:03d}.png")
        cv2.imwrite(path, img)
        print(f"  [fringe] {path}")

    print(f"Done: {NUM_STAR_IMAGES} star + {NUM_FRINGE_IMAGES} fringe images generated.")


if __name__ == "__main__":
    main()
