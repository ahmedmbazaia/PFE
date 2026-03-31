"""
Full AI pipeline for optical interferometry.
Runs all stages end-to-end:
  1. Generate synthetic data (star + fringe images)
  2. Detect stars in star field images
  3. Compute fringe visibility and phase
  4. Reconstruct sky brightness map
  5. Train image classifier
"""

import time
import sys
import os

# Ensure the script can find sibling modules
sys.path.insert(0, os.path.dirname(__file__))

import synthetic_data
import detect
import visibility
import reconstruct
import classifier


def run_stage(name, func):
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = func()
    elapsed = time.time() - t0
    print(f"  [{name}] completed in {elapsed:.2f}s")
    return result


def main():
    print("=" * 60)
    print("  OPTICAL INTERFEROMETRY AI PIPELINE")
    print("  (Synthetic data mode - no hardware)")
    print("=" * 60)

    t_total = time.time()

    run_stage("1. Synthetic Data Generation", synthetic_data.main)
    run_stage("2. Star Detection", detect.main)
    run_stage("3. Visibility & Phase", visibility.main)
    run_stage("4. Sky Reconstruction", reconstruct.main)
    run_stage("5. Image Classifier", classifier.main)

    elapsed_total = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE - Total time: {elapsed_total:.2f}s")
    print(f"{'='*60}")

    # List outputs
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    print("\nGenerated outputs:")
    for root, dirs, files in os.walk(data_dir):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {os.path.relpath(fpath, data_dir):40s} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
