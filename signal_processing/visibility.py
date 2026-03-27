#!/usr/bin/env python3
"""
Visibility Computation Module
==============================
Computes interferometric visibility and phase from the intensity
measurements of two telescopes (T1 and T2).

Visibility V = (Imax - Imin) / (Imax + Imin)
Phase φ estimated from timestamp difference and baseline geometry.
"""

import math


# ─── Constants ───────────────────────────────────────────────
WAVELENGTH_NM = 550.0          # observation wavelength (green light)
BASELINE_MM = 300.0            # default T1-T2 baseline distance
SPEED_OF_LIGHT = 3.0e8        # m/s


def compute_visibility(i1, i2):
    """
    Compute fringe visibility from two intensity measurements.

    V = (Imax - Imin) / (Imax + Imin)

    Args:
        i1: intensity from telescope 1 (0–4095 or 0.0–1.0)
        i2: intensity from telescope 2 (same scale)

    Returns:
        Visibility value in [0.0, 1.0], or 0.0 if both inputs are zero.
    """
    i_max = max(i1, i2)
    i_min = min(i1, i2)
    total = i_max + i_min

    if total == 0:
        return 0.0

    return (i_max - i_min) / total


def compute_phase(timestamp_t1, timestamp_t2, baseline_mm=BASELINE_MM):
    """
    Estimate phase difference from the time delay between T1 and T2.

    The geometric delay introduces a phase shift:
        φ = 2π · baseline · sin(θ) / λ

    For simplicity, we approximate the angular offset from the
    timestamp difference (assuming a known slew rate or sky rotation).

    Args:
        timestamp_t1: T1 timestamp (seconds or millis)
        timestamp_t2: T2 timestamp (seconds or millis)
        baseline_mm:  T1-T2 separation in mm

    Returns:
        Phase in radians [0, 2π).
    """
    dt = abs(timestamp_t1 - timestamp_t2)

    # Convert baseline to meters
    baseline_m = baseline_mm / 1000.0
    wavelength_m = WAVELENGTH_NM * 1e-9

    # Approximate angular offset from time delay
    # (simplified: assumes dt maps to a geometric path difference)
    if dt == 0:
        return 0.0

    path_diff = baseline_m * math.sin(dt * 0.01)  # scaled angle proxy
    phase = (2 * math.pi * path_diff / wavelength_m) % (2 * math.pi)
    return phase


def compute_for_star(star_detection, t1_intensity, t2_intensity,
                     t1_timestamp=0, t2_timestamp=0, baseline_mm=BASELINE_MM):
    """
    Compute visibility and phase for a single detected star.

    Args:
        star_detection: dict with "x", "y", "intensity" from detect.py
        t1_intensity:   raw light intensity from T1 photodiode
        t2_intensity:   raw light intensity from T2 photodiode
        t1_timestamp:   T1 measurement timestamp
        t2_timestamp:   T2 measurement timestamp
        baseline_mm:    telescope separation

    Returns:
        Dict with star info + visibility + phase.
    """
    vis = compute_visibility(t1_intensity, t2_intensity)
    phase = compute_phase(t1_timestamp, t2_timestamp, baseline_mm)

    return {
        "x": star_detection["x"],
        "y": star_detection["y"],
        "detection_intensity": star_detection["intensity"],
        "t1_intensity": t1_intensity,
        "t2_intensity": t2_intensity,
        "visibility": round(vis, 6),
        "phase_rad": round(phase, 6),
        "phase_deg": round(math.degrees(phase), 2),
        "baseline_mm": baseline_mm,
    }


def compute_batch(detections, t1_intensity, t2_intensity,
                  t1_timestamp=0, t2_timestamp=0, baseline_mm=BASELINE_MM):
    """
    Compute visibility and phase for all detected stars.

    Uses the same T1/T2 photodiode readings for all stars in a single
    frame (the photodiode measures total sky brightness, not per-star).

    Returns:
        List of result dicts from compute_for_star().
    """
    results = []
    for star in detections:
        result = compute_for_star(
            star, t1_intensity, t2_intensity,
            t1_timestamp, t2_timestamp, baseline_mm
        )
        results.append(result)
    return results


# ─── Standalone usage ────────────────────────────────────────
if __name__ == "__main__":
    # Demo with synthetic values
    print("Visibility computation demo:")
    print(f"  V(1000, 800) = {compute_visibility(1000, 800):.4f}")
    print(f"  V(500, 500)  = {compute_visibility(500, 500):.4f}")
    print(f"  V(4000, 100) = {compute_visibility(4000, 100):.4f}")
    print(f"  Phase(0, 0.5) = {compute_phase(0, 0.5):.4f} rad")
