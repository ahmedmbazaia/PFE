"""
Star detection module.
Detects point sources (stars) in star field images using blob detection
and Gaussian fitting.
"""

import os
import glob
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_CSV = os.path.join(DATA_DIR, "detections.csv")
OUTPUT_PLOT_DIR = os.path.join(DATA_DIR, "plots")


def detect_stars(image, threshold=40, min_area=4, max_area=500):
    """Detect star positions in a grayscale image using thresholding + contours."""
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                # Estimate brightness as peak pixel value in bounding box
                x, y, w, h = cv2.boundingRect(cnt)
                roi = image[y:y+h, x:x+w]
                peak = float(roi.max())
                detections.append({
                    "x": round(cx, 2),
                    "y": round(cy, 2),
                    "area": round(area, 1),
                    "peak_brightness": peak,
                })
    return detections


def process_star_images():
    """Process all star images and return a DataFrame of detections."""
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    star_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "star_*.png")))
    if not star_files:
        print("No star images found. Run synthetic_data.py first.")
        return pd.DataFrame()

    all_detections = []
    for fpath in star_files:
        fname = os.path.basename(fpath)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        dets = detect_stars(img)
        for d in dets:
            d["file"] = fname
        all_detections.extend(dets)

        # Save annotated plot
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(img, cmap="gray", origin="lower")
        for d in dets:
            ax.plot(d["x"], d["y"], "r+", markersize=10, markeredgewidth=1.5)
        ax.set_title(f"{fname} ({len(dets)} stars)")
        ax.axis("off")
        plot_path = os.path.join(OUTPUT_PLOT_DIR, fname.replace(".png", "_det.png"))
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}: {len(dets)} stars detected")

    df = pd.DataFrame(all_detections)
    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Detections saved to {OUTPUT_CSV} ({len(df)} total)")
    return df


def main():
    print("=== Star Detection ===")
    df = process_star_images()
    if not df.empty:
        print(f"\nSummary:\n{df.describe()}")
    print("Detection plots saved to:", os.path.abspath(OUTPUT_PLOT_DIR))


if __name__ == "__main__":
    main()
