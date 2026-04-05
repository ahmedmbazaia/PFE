"""
Full AI pipeline for optical interferometry.
Runs all stages end-to-end:
  1. Generate synthetic data (star + fringe images)
  2. Detect and classify sources in star field images
  3. Compute fringe visibility and phase
  4. Reconstruct sky brightness map
  5. Train 4-class astrophysical source classifier
  6. Re-classify detections with trained model, annotate reconstruction,
     print summary table
"""

import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import synthetic_data
import detect
import visibility
import reconstruct
import classifier


def run_stage(name, func):
    print(f"\n{'=' * 60}")
    print(f"  STAGE: {name}")
    print(f"{'=' * 60}")
    t0 = time.time()
    result = func()
    elapsed = time.time() - t0
    print(f"  [{name}] completed in {elapsed:.2f}s")
    return result


def annotate_and_summarize():
    """
    Stage 6: Load detections, re-classify with trained model,
    annotate the reconstructed sky image, and print a summary table.
    """
    import cv2
    import pandas as pd
    import numpy as np

    DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
    plots_dir  = os.path.join(DATA_DIR, "plots")
    det_csv    = os.path.join(DATA_DIR, "detections.csv")
    model_path = os.path.join(DATA_DIR, "model.pkl")
    recon_path = os.path.join(plots_dir, "sky_reconstruction.png")

    # ── Load detections ───────────────────────────────────────────────────────
    if not os.path.isfile(det_csv):
        print("  No detections.csv — skipping annotation stage.")
        return

    df = pd.read_csv(det_csv)
    if df.empty:
        print("  detections.csv is empty — nothing to annotate.")
        return

    # ── Re-classify with trained model ────────────────────────────────────────
    if os.path.isfile(model_path):
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        detections = df.to_dict("records")
        detections = classifier.classify_detections(detections, model_data)
        df = pd.DataFrame(detections)
        df.to_csv(det_csv, index=False)
        print(f"  Re-classified {len(df)} detections with trained model.")
    else:
        print("  model.pkl not found — using rule-based classifications from detect stage.")

    # ── Annotate reconstruction image ─────────────────────────────────────────
    if os.path.isfile(recon_path):
        img_bgr = cv2.imread(recon_path)
        if img_bgr is not None:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            dets = detect.detect_stars(gray, threshold=15, min_area=2, max_area=300)
            dets = detect.compute_neighbor_distances(dets)
            dets = detect.classify_all(dets)

            # If model available, re-classify reconstruction sources too
            if os.path.isfile(model_path):
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                dets = classifier.classify_detections(dets, model_data)

            img_ann = detect.draw_labeled_image(gray, dets)

            # Composite: paste annotation over original reconstruction panels
            h_orig, w_orig = img_bgr.shape[:2]
            h_ann,  w_ann  = img_ann.shape[:2]
            if w_orig > w_ann:
                # Reconstruction is the rightmost third panel — replace it
                panel_w = w_orig // 3
                x_off   = w_orig - panel_w
                resized = cv2.resize(img_ann, (panel_w, h_orig))
                composite = img_bgr.copy()
                composite[:, x_off:] = resized
            else:
                composite = img_ann

            labeled_path = os.path.join(plots_dir, "sky_reconstruction_labeled.png")
            cv2.imwrite(labeled_path, composite)
            print(f"  Annotated reconstruction saved to {os.path.abspath(labeled_path)}")
    else:
        print("  sky_reconstruction.png not found — skipping image annotation.")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n  {'=' * 80}")
    print(f"  DETECTION SUMMARY  ({len(df)} sources total)")
    print(f"  {'=' * 80}")
    hdr = f"  {'#':<4}  {'Type':<24}  {'Name':<14}  {'Brightness':>10}  {'x':>7}  {'y':>7}  {'Conf':>5}"
    print(hdr)
    print(f"  {'-' * 76}")

    for i, row in enumerate(df.itertuples(index=False), 1):
        cls   = str(getattr(row, "classification", "—"))
        name  = str(getattr(row, "arabic_name", "") or "—")
        peak  = float(getattr(row, "peak_brightness", 0))
        x     = float(getattr(row, "x", 0))
        y     = float(getattr(row, "y", 0))
        conf  = float(getattr(row, "confidence", 0))
        print(f"  {i:<4}  {cls:<24}  {name:<14}  {peak:>10.0f}  "
              f"{x:>7.1f}  {y:>7.1f}  {int(conf * 100):>4}%")

    print(f"  {'-' * 76}")
    if "classification" in df.columns:
        for cls_name, cnt in df["classification"].value_counts().items():
            print(f"  {str(cls_name):<30}: {cnt} source(s)")
    print()


def main():
    print("=" * 60)
    print("  OPTICAL INTERFEROMETRY AI PIPELINE")
    print("  (Synthetic data mode - no hardware)")
    print("=" * 60)

    t_total = time.time()

    run_stage("1. Synthetic Data Generation", synthetic_data.main)
    run_stage("2. Source Detection & Classification", detect.main)
    run_stage("3. Visibility & Phase", visibility.main)
    run_stage("4. Sky Reconstruction", reconstruct.main)
    run_stage("5. 4-Class Source Classifier (train)", classifier.main)
    run_stage("6. Annotation & Summary", annotate_and_summarize)

    elapsed_total = time.time() - t_total

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — Total time: {elapsed_total:.2f}s")
    print(f"{'=' * 60}")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    print("\nGenerated outputs:")
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {os.path.relpath(fpath, data_dir):<42} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
