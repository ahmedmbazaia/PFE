import base64
import csv
import logging
import os

from flask import Flask, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STATION_DATA = os.path.join(PROJECT_ROOT, "station", "data")
CSV_PATH = os.path.join(STATION_DATA, "logs", "mission_log.csv")
LABELED_SKY_PATH = os.path.join(PROJECT_ROOT, "data", "plots", "labeled_sky.png")
DETECTIONS_CSV = os.path.join(STATION_DATA, "detections.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("dashboard")

app = Flask(__name__)


def get_latest_row():
    """Read the most recent row from mission_log.csv."""
    if not os.path.isfile(CSV_PATH):
        return None
    try:
        with open(CSV_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            return rows[-1]
    except Exception as e:
        logger.error("Error reading CSV: %s", e)
    return None


def get_latest_image():
    """Return data/plots/labeled_sky.png as base64."""
    if not os.path.isfile(LABELED_SKY_PATH):
        return None
    try:
        with open(LABELED_SKY_PATH, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return {"filename": "labeled_sky.png", "data": f"data:image/png;base64,{data}"}
    except Exception as e:
        logger.error("Error reading labeled sky image: %s", e)
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
        return jsonify({"status": "waiting", "received_at": None, "data": {}})
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
    logger.info("Dashboard image dir:   %s", IMAGE_DIR)
    app.run(host="0.0.0.0", port=5001, debug=True)
