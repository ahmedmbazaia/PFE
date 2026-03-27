# Signal Processing Pipeline

AI-driven pipeline for interferometric fringe analysis: star detection, visibility computation, sky reconstruction, and signal quality classification.

## Modules

| File | Role |
|------|------|
| `synthetic_data.py` | Generate fake fringe images with point sources |
| `detect.py` | OpenCV star detection (threshold + contours) |
| `visibility.py` | Compute fringe visibility V and phase φ |
| `reconstruct.py` | Build 2D sky map from visibility data |
| `classifier.py` | Random Forest signal quality classifier |
| `pipeline.py` | End-to-end orchestrator |

## Pipeline Flow

```
mission_log.csv ──┐
                  ├──► detect → visibility → reconstruct → classify → report
synthetic data ───┘
```

1. Load latest telemetry from `data/logs/mission_log.csv` (or generate synthetic data)
2. Detect bright point sources in fringe image (OpenCV)
3. Compute visibility V = (Imax - Imin) / (Imax + Imin) and phase
4. Reconstruct 2D sky map (Matplotlib)
5. Classify signal quality: **GOOD** / **DEGRADED** / **LOST** (scikit-learn Random Forest)
6. Print summary report

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic data
python synthetic_data.py

# Train classifier
python classifier.py

# Run full pipeline (auto-uses synthetic if no real data)
python pipeline.py

# Force synthetic mode
python pipeline.py --synthetic
```

## Standalone Module Usage

```python
# Detect stars in any image
from detect import detect_from_file
stars = detect_from_file("path/to/image.png")

# Compute visibility
from visibility import compute_visibility
v = compute_visibility(t1_intensity=3000, t2_intensity=2500)

# Classify signal
from classifier import classify
quality = classify(visibility=0.8, phase_rad=1.2, baseline_mm=300,
                   t1_intensity=3000, t2_intensity=2500)
```

## Output Files

```
signal_processing/
├── data/
│   ├── images/             # Synthetic + reconstruction PNGs
│   └── synthetic_metadata.csv
├── model.pkl               # Trained Random Forest model
└── ...
```

## Dependencies

- Python 3.9+
- numpy, opencv-python, matplotlib, scikit-learn, pandas
