"""
Star detection module.
Detects point sources in star field images and classifies each source into
one of four astrophysical types with confidence score and Arabic star name.
"""

import os
import glob
import numpy as np
import cv2
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_CSV = os.path.join(DATA_DIR, "detections.csv")
OUTPUT_PLOT_DIR = os.path.join(DATA_DIR, "plots")

ARABIC_NAMES = [
    "Althurayya", "Rigel", "Sirius", "Vega",
    "Aldebaran", "Betelgeuse", "Deneb",
]

# BGR colors per source type
TYPE_COLORS = {
    "Étoile ponctuelle": (220, 80,  20),   # blue
    "Système binaire":   (0,  210, 255),   # yellow
    "Nébuleuse étendue": (30,  30, 220),   # red
    "Objet compact":     (230, 230, 230),  # white
}

TYPE_SHORT = {
    "Étoile ponctuelle": "E.PONCT",
    "Système binaire":   "BINAIRE",
    "Nébuleuse étendue": "NEBUL.",
    "Objet compact":     "COMPACT",
}


def detect_stars(image, threshold=40, min_area=4, max_area=500):
    """
    Detect bright point sources in a grayscale image.
    Returns list of dicts with: x, y, area, peak_brightness, circularity, compactness.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y + h, x:x + w]
        peak = float(roi.max())

        perimeter = cv2.arcLength(cnt, True)
        circularity = float(4 * np.pi * area / (perimeter ** 2 + 1e-10))
        compactness = float(peak / (area + 1e-10))

        detections.append({
            "x": round(cx, 2),
            "y": round(cy, 2),
            "area": round(area, 1),
            "peak_brightness": peak,
            "circularity": round(min(circularity, 1.0), 4),
            "compactness": round(compactness, 4),
        })
    return detections


def compute_neighbor_distances(detections):
    """
    Add 'neighbor_distance' (px to nearest other source) to each detection.
    """
    if len(detections) < 2:
        for d in detections:
            d["neighbor_distance"] = 9999.0
        return detections

    positions = np.array([[d["x"], d["y"]] for d in detections])
    for i, det in enumerate(detections):
        dists = np.sqrt(np.sum((positions - positions[i]) ** 2, axis=1))
        dists[i] = np.inf
        det["neighbor_distance"] = round(float(dists.min()), 2)
    return detections


def classify_source(det):
    """
    Rule-based classification into 4 astrophysical types.
    Returns (class_name, confidence_float).

    Rules:
      Objet compact     — very small (area<25), very bright (peak>200), high compactness
      Système binaire   — nearest neighbour within 20 px
      Nébuleuse étendue — large (area>100) or diffuse (low circularity + area>40)
      Étoile ponctuelle — all other cases
    """
    area = det["area"]
    peak = det["peak_brightness"]
    circ = det["circularity"]
    comp = det["compactness"]
    nd   = det["neighbor_distance"]

    if area < 25 and peak > 200 and comp > 8.0:
        conf = round(min(0.97, 0.72 + (peak - 200) / 200 * 0.25), 2)
        return "Objet compact", conf

    if nd < 20:
        conf = round(min(0.93, 0.65 + (20 - nd) / 20 * 0.28), 2)
        return "Système binaire", conf

    if area > 100 or (circ < 0.4 and area > 40):
        conf = round(min(0.91, 0.58 + min(area, 300) / 300 * 0.33), 2)
        return "Nébuleuse étendue", conf

    conf = round(min(0.95, 0.65 + circ * 0.30), 2)
    return "Étoile ponctuelle", conf


def classify_all(detections):
    """
    Classify each detection and assign an Arabic name to point stars.
    Mutates and returns the list.
    """
    arabic_idx = 0
    for det in detections:
        cls, conf = classify_source(det)
        det["classification"] = cls
        det["confidence"] = conf
        if cls == "Étoile ponctuelle":
            det["arabic_name"] = ARABIC_NAMES[arabic_idx % len(ARABIC_NAMES)]
            arabic_idx += 1
        else:
            det["arabic_name"] = ""
    return detections


def draw_labeled_image(img_gray, detections):
    """
    Build an annotated BGR image:
      - Colored circle (radius ∝ √area) per type
      - Label: source index, short type name, Arabic name, confidence %
    """
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for i, det in enumerate(detections, 1):
        cx    = int(round(det["x"]))
        cy    = int(round(det["y"]))
        cls   = det.get("classification", "Étoile ponctuelle")
        conf  = det.get("confidence", 0.0)
        name  = det.get("arabic_name", "")
        color = TYPE_COLORS.get(cls, (200, 200, 200))
        short = TYPE_SHORT.get(cls, cls[:7])

        radius = max(8, int(np.sqrt(det["area"] / np.pi) + 3))
        cv2.circle(img_color, (cx, cy), radius, color, 1, cv2.LINE_AA)
        cv2.drawMarker(img_color, (cx, cy), color, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)

        line1 = f"S{i:02d} {short}"
        line2 = f"{name}  {int(conf * 100)}%" if name else f"{int(conf * 100)}%"
        tx, ty = cx + radius + 3, cy - 3
        cv2.putText(img_color, line1, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        cv2.putText(img_color, line2, (tx, ty + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (170, 170, 170), 1, cv2.LINE_AA)

    return img_color


def process_star_images():
    """Process all synthetic star images, save detections.csv and annotated plots."""
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
        dets = compute_neighbor_distances(dets)
        dets = classify_all(dets)

        for d in dets:
            d["file"] = fname
        all_detections.extend(dets)

        img_ann = draw_labeled_image(img, dets)
        plot_path = os.path.join(OUTPUT_PLOT_DIR, fname.replace(".png", "_det.png"))
        cv2.imwrite(plot_path, img_ann)
        print(f"  {fname}: {len(dets)} sources detected")

    df = pd.DataFrame(all_detections)
    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Detections saved to {OUTPUT_CSV} ({len(df)} total)")
    return df


def process_single_image(image_path):
    """Run detection + classification on a single image and save labeled output."""
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
    dets = compute_neighbor_distances(dets)
    dets = classify_all(dets)

    print(f"\nDetected {len(dets)} sources:\n")
    print(f"  {'#':<4} {'x':>8} {'y':>8}  {'Type':<22} {'Name':<14} {'Conf':>6}  {'Bright':>8}")
    print(f"  {'-'*72}")
    for i, d in enumerate(dets, 1):
        name = d.get("arabic_name") or "—"
        print(f"  {i:<4} {d['x']:>8.1f} {d['y']:>8.1f}  "
              f"{d['classification']:<22} {name:<14} "
              f"{int(d['confidence'] * 100):>5}%  {d['peak_brightness']:>7.0f}")

    img_ann = draw_labeled_image(img, dets)
    out_path = os.path.join(OUTPUT_PLOT_DIR, "labeled_sky.png")
    cv2.imwrite(out_path, img_ann)
    print(f"\nLabeled image saved to: {os.path.abspath(out_path)}")

    df = pd.DataFrame(dets)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Detections CSV saved to: {os.path.abspath(OUTPUT_CSV)}")


def main():
    import sys
    if len(sys.argv) > 1:
        process_single_image(sys.argv[1])
    else:
        print("=== Star Detection ===")
        df = process_star_images()
        if not df.empty:
            print(f"\nClass breakdown:\n{df['classification'].value_counts().to_string()}")
        print("\nDetection plots saved to:", os.path.abspath(OUTPUT_PLOT_DIR))


if __name__ == "__main__":
    main()
