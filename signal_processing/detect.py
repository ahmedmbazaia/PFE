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


def process_single_image(image_path):
    """Run detection on a single user-provided image."""
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    image_path = os.path.abspath(image_path)

    if not os.path.isfile(image_path):
        print(f"Error: file not found: {image_path}")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: could not read image: {image_path}")
        return

    fname = os.path.basename(image_path)
    print(f"=== Star Detection: {fname} ===")
    print(f"Image size: {img.shape[1]}x{img.shape[0]} px")

    dets = detect_stars(img)
    print(f"\nDetected {len(dets)} stars:\n")
    print(f"  {'#':<4} {'x':>8} {'y':>8} {'brightness':>12}")
    print(f"  {'-'*36}")
    for i, d in enumerate(dets, 1):
        print(f"  {i:<4} {d['x']:>8.1f} {d['y']:>8.1f} {d['peak_brightness']:>12.0f}")

    # Save labeled output image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, d in enumerate(dets, 1):
        cx, cy = int(round(d["x"])), int(round(d["y"]))
        cv2.drawMarker(img_color, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(img_color, str(i), (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    out_path = os.path.join(OUTPUT_PLOT_DIR, "labeled_sky.png")
    cv2.imwrite(out_path, img_color)
    print(f"\nLabeled image saved to: {os.path.abspath(out_path)}")


def main():
    import sys
    if len(sys.argv) > 1:
        process_single_image(sys.argv[1])
    else:
        print("=== Star Detection ===")
        df = process_star_images()
        if not df.empty:
            print(f"\nSummary:\n{df.describe()}")
        print("Detection plots saved to:", os.path.abspath(OUTPUT_PLOT_DIR))


if __name__ == "__main__":
    main()
