import csv
import logging
import os

from flask import Flask, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "logs", "mission_log.csv")

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
        "received_at": row.get("timestamp", None),
        "data": row,
    })


if __name__ == "__main__":
    logger.info("Dashboard reading from: %s", CSV_PATH)
    app.run(host="0.0.0.0", port=5001, debug=True)
