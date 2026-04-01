import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STATION_DIR = os.path.join(PROJECT_ROOT, "station")

if STATION_DIR not in sys.path:
    sys.path.insert(0, STATION_DIR)

import config  # type: ignore  # noqa: E402
import lora_receiver  # type: ignore  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("dashboard")

app = Flask(__name__)

_latest_lock = threading.Lock()
_latest_packet = None
_latest_received_at = None
_receiver_started = False


def _set_latest(packet):
    global _latest_packet, _latest_received_at
    with _latest_lock:
        _latest_packet = packet
        _latest_received_at = datetime.now(timezone.utc).isoformat()


def get_latest_snapshot():
    with _latest_lock:
        if _latest_packet is None:
            return {
                "status": "waiting",
                "received_at": None,
                "data": {},
            }

        return {
            "status": "ok",
            "received_at": _latest_received_at,
            "data": dict(_latest_packet),
        }


def receiver_loop():
    logger.info("Starting LoRa receiver thread on %.1f MHz", config.LORA_FREQUENCY / 1e6)

    ready = lora_receiver.setup()
    if not ready:
        logger.warning("LoRa receiver setup failed or is running in stub mode")

    while True:
        try:
            packet = lora_receiver.receive()
            if packet:
                logger.info("Received LoRa packet: %s", packet)
                _set_latest(packet)
            else:
                time.sleep(0.2)
        except Exception as exc:
            logger.exception("LoRa polling error: %s", exc)
            time.sleep(1.0)


def start_receiver_thread():
    global _receiver_started
    if _receiver_started:
        return

    thread = threading.Thread(target=receiver_loop, name="lora-receiver", daemon=True)
    thread.start()
    _receiver_started = True


@app.before_request
def ensure_receiver_started():
    start_receiver_thread()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/latest")
def latest():
    return jsonify(get_latest_snapshot())


if __name__ == "__main__":
    start_receiver_thread()
    app.run(host="0.0.0.0", port=5000, debug=True)
