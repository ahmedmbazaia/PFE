import base64
import csv
import io
import logging
import os

from flask import Flask, jsonify, render_template

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STATION_DATA = os.path.join(PROJECT_ROOT, "station", "data")
CSV_PATH = os.path.join(STATION_DATA, "logs", "mission_log.csv")
LABELED_SKY_PATH = "/home/pfe2/PFE/data/plots/labeled_sky.png"
DETECTIONS_CSV = os.path.join(STATION_DATA, "detections.csv")

DEMO_VALUES = {
    "t1_pitch": "12.3", "t1_roll": "-4.1", "t1_yaw": "178.2", "t1_light_intensity": "342",
    "t2_pitch": "8.7",  "t2_roll": "2.3",  "t2_yaw": "182.5", "t2_light_intensity": "287",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("dashboard")

app = Flask(__name__)


def get_latest_row():
    """Read the most recent row from mission_log.csv."""
    if not os.path.isfile(CSV_PATH):
        logger.warning("CSV not found: %s", CSV_PATH)
        return None
    try:
        with open(CSV_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            row = rows[-1]
            logger.info("CSV row loaded — t1_pitch=%s t2_pitch=%s",
                        row.get("t1_pitch", ""), row.get("t2_pitch", ""))
            return row
    except Exception as e:
        logger.error("Error reading CSV: %s", e)
    return None


def get_latest_image():
    """Return labeled_sky.png resized to max 800px width as JPEG base64."""
    logger.info("Looking for labeled sky image at: %s", LABELED_SKY_PATH)
    if not os.path.isfile(LABELED_SKY_PATH):
        logger.warning("Labeled sky image not found: %s", LABELED_SKY_PATH)
        return None
    try:
        if PIL_AVAILABLE:
            img = Image.open(LABELED_SKY_PATH).convert("RGB")
            max_w = 800
            if img.width > max_w:
                ratio = max_w / img.width
                new_size = (max_w, int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            raw = buf.getvalue()
            logger.info("Image resized to %dx%d, %d bytes", img.width, img.height, len(raw))
            data = base64.b64encode(raw).decode("utf-8")
            return {"filename": "labeled_sky.jpg", "data": f"data:image/jpeg;base64,{data}"}
        else:
            # PIL not available — return raw PNG (may be large)
            logger.warning("Pillow not installed — returning raw PNG")
            with open(LABELED_SKY_PATH, "rb") as f:
                raw = f.read()
            data = base64.b64encode(raw).decode("utf-8")
            logger.info("Raw PNG loaded (%d bytes)", len(raw))
            return {"filename": "labeled_sky.png", "data": f"data:image/png;base64,{data}"}
    except Exception as e:
        logger.error("Error reading/resizing image: %s", e)
    return None


def get_ai_detections():
    """Read latest AI detection results from detections.csv."""
    if not os.path.isfile(DETECTIONS_CSV):
        return []
    try:
        with open(DETECTIONS_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        logger.error("Error reading detections CSV: %s", e)
    return []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/latest")
def latest():
    row = get_latest_row()
    if row is None:
        row = {}
    # Fill missing or zero fields with demo values
    for key, val in DEMO_VALUES.items():
        v = str(row.get(key, "")).strip()
        if v == "" or v == "0" or v == "0.0":
            row[key] = val
            logger.debug("Demo fallback applied: %s = %s", key, val)
    return jsonify({
        "status": "ok",
        "received_at": row.get("rpi_timestamp", None),
        "data": row,
    })


@app.route("/api/image")
def image():
    img = get_latest_image()
    if img is None:
        return jsonify({"status": "no_image"})
    return jsonify({"status": "ok", "filename": img["filename"], "src": img["data"]})


@app.route("/api/ai")
def ai_detections():
    rows = get_ai_detections()
    return jsonify({"status": "ok", "detections": rows})


if __name__ == "__main__":
    logger.info("Dashboard reading CSV: %s", CSV_PATH)
    logger.info("Dashboard labeled sky: %s", LABELED_SKY_PATH)
    logger.info("Pillow available: %s", PIL_AVAILABLE)
    app.run(host="0.0.0.0", port=5001, debug=True)
