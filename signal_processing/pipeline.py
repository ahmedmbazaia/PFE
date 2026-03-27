#!/usr/bin/env python3
"""
Full Signal Processing Pipeline
================================
Runs the complete chain: load data → detect stars → compute visibility
→ reconstruct sky → classify signal quality → print summary.

Works standalone with synthetic data if no real mission data is available.
"""

import os
import sys
from datetime import datetime

import pandas as pd

import synthetic_data
import detect
import visibility
import reconstruct
import classifier

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MISSION_LOG = os.path.join(BASE_DIR, "..", "data", "logs", "mission_log.csv")
SYNTH_METADATA = os.path.join(BASE_DIR, "data", "synthetic_metadata.csv")
SYNTH_IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")


def load_mission_data():
    """
    Load the latest record from the station's mission_log.csv.
    Returns a dict with T1/T2 telemetry, or None if unavailable.
    """
    if not os.path.isfile(MISSION_LOG):
        return None

    try:
        df = pd.read_csv(MISSION_LOG)
        if df.empty:
            return None

        latest = df.iloc[-1].to_dict()
        # Convert NaN to 0 for numeric fields
        for key in latest:
            if pd.isna(latest[key]):
                latest[key] = 0
        print(f"[pipeline] Loaded mission data — {len(df)} records, using latest")
        return latest
    except Exception as e:
        print(f"[pipeline] Could not load mission log: {e}")
        return None


def load_synthetic_image():
    """
    Pick a synthetic fringe image for processing.
    Generates new data if none exists.
    Returns the image path.
    """
    if not os.path.isdir(SYNTH_IMAGE_DIR) or not os.listdir(SYNTH_IMAGE_DIR):
        print("[pipeline] No synthetic images found — generating dataset...")
        synthetic_data.generate_dataset(20)

    # Pick the first available PNG
    images = sorted(f for f in os.listdir(SYNTH_IMAGE_DIR) if f.endswith(".png"))
    if not images:
        print("[pipeline] ERROR: no images after generation")
        return None

    return os.path.join(SYNTH_IMAGE_DIR, images[0])


def run(use_synthetic=False):
    """
    Execute the full pipeline end to end.

    Args:
        use_synthetic: if True, skip mission data and use synthetic images.

    Returns:
        Summary dict with all results.
    """
    print("=" * 60)
    print("  Signal Processing Pipeline")
    print(f"  {datetime.now().isoformat()}")
    print("=" * 60)

    # ── Step 0: Load data ────────────────────────────────────
    mission = None if use_synthetic else load_mission_data()

    if mission:
        t1_intensity = float(mission.get("t1_light_intensity", 0))
        t2_intensity = float(mission.get("t2_light_intensity", 0))
        t1_timestamp = float(mission.get("t1_timestamp", 0))
        t2_timestamp = float(mission.get("t2_timestamp", 0))
        baseline = float(mission.get("t1_baseline_distance_mm", 300))
        print(f"  T1 intensity: {t1_intensity}")
        print(f"  T2 intensity: {t2_intensity}")
        print(f"  Baseline: {baseline} mm")
    else:
        print("[pipeline] No mission data — using synthetic defaults")
        t1_intensity = 2500.0
        t2_intensity = 2100.0
        t1_timestamp = 0
        t2_timestamp = 0.3
        baseline = 300.0

    # ── Step 1: Load / generate image ────────────────────────
    print("\n── Step 1: Load image ──")
    image_path = load_synthetic_image()
    if image_path is None:
        print("FATAL: no image available")
        return None
    print(f"  Image: {image_path}")

    # ── Step 2: Detect stars ─────────────────────────────────
    print("\n── Step 2: Detect stars ──")
    detections = detect.detect_from_file(image_path)
    print(f"  Detected {len(detections)} stars")
    for i, d in enumerate(detections):
        print(f"    Star {i}: ({d['x']}, {d['y']}) intensity={d['intensity']}")

    if not detections:
        print("  WARNING: no stars detected — using lower threshold")
        detections = detect.detect_from_file(image_path, threshold=120)
        print(f"  Re-detected {len(detections)} stars")

    # ── Step 3: Compute visibility ───────────────────────────
    print("\n── Step 3: Compute visibility ──")
    star_results = visibility.compute_batch(
        detections, t1_intensity, t2_intensity,
        t1_timestamp, t2_timestamp, baseline
    )
    for s in star_results:
        print(f"    ({s['x']}, {s['y']}): V={s['visibility']:.4f}  "
              f"φ={s['phase_deg']:.1f}°")

    # ── Step 4: Reconstruct sky ──────────────────────────────
    print("\n── Step 4: Reconstruct sky map ──")
    sky_array, recon_path = reconstruct.reconstruct_and_save(star_results)

    # ── Step 5: Classify signal quality ──────────────────────
    print("\n── Step 5: Classify signal quality ──")
    classified = classifier.classify_batch(star_results)
    for s in classified:
        print(f"    ({s['x']}, {s['y']}): {s['signal_quality']}  "
              f"(V={s['visibility']:.3f})")

    # ── Step 6: Summary report ───────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY REPORT")
    print("=" * 60)
    print(f"  Timestamp:     {datetime.now().isoformat()}")
    print(f"  Image:         {os.path.basename(image_path)}")
    print(f"  Stars found:   {len(detections)}")
    print(f"  Baseline:      {baseline} mm")
    print(f"  T1 intensity:  {t1_intensity}")
    print(f"  T2 intensity:  {t2_intensity}")

    if classified:
        avg_vis = sum(s["visibility"] for s in classified) / len(classified)
        qualities = [s["signal_quality"] for s in classified]
        # Overall quality is the worst individual quality
        if "LOST" in qualities:
            overall = "LOST"
        elif "DEGRADED" in qualities:
            overall = "DEGRADED"
        else:
            overall = "GOOD"
        print(f"  Avg visibility: {avg_vis:.4f}")
        print(f"  Overall signal: {overall}")
    else:
        print("  No stars classified")

    print(f"  Reconstruction: {recon_path}")
    print("=" * 60)

    return {
        "timestamp": datetime.now().isoformat(),
        "image": image_path,
        "detections": detections,
        "star_results": classified,
        "reconstruction": recon_path,
        "overall_quality": overall if classified else "UNKNOWN",
    }


# ─── Entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    synth = "--synthetic" in sys.argv or not os.path.isfile(MISSION_LOG)
    if synth and "--synthetic" not in sys.argv:
        print("[pipeline] No mission_log.csv found — running on synthetic data\n")
    run(use_synthetic=synth)
