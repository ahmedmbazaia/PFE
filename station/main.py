#!/usr/bin/env python3
# =============================================================
#  Central Station — Main Loop (Raspberry Pi)
#  Receives telemetry from T2 (WiFi HTTP POST + FSO),
#  T1 (LoRa), captures sky images, logs everything.
#  Also runs a Flask server on port 5000 for HTTP telemetry.
# =============================================================

import json
import logging
import signal
import sys
import time
import threading
from datetime import datetime

from flask import Flask, request, jsonify

import config
import lora_receiver
import fso_receiver
import camera
import gps_parser
import data_logger

# ─── Logging setup ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-12s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("station")

# ─── Flask server for WiFi telemetry ────────────────────────
app = Flask(__name__)
app.logger.setLevel(logging.WARNING)  # suppress Flask request logs

# Latest telemetry received via HTTP POST
latest_http_telemetry = None


@app.route("/telemetry", methods=["POST"])
def telemetry_endpoint():
    """Receive JSON telemetry from T2 via WiFi HTTP POST."""
    global latest_http_telemetry
    try:
        data = request.get_json(force=True)
        data["_source"] = "wifi"
        data["_received_at"] = datetime.now().isoformat()
        latest_http_telemetry = data
        logger.info("  HTTP RX: node=%s counter=%s",
                    data.get("node", "?"), data.get("counter", "?"))
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error("  HTTP RX error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 400


def get_http_telemetry():
    """Return and clear the latest HTTP telemetry packet."""
    global latest_http_telemetry
    data = latest_http_telemetry
    latest_http_telemetry = None
    return data


def start_flask():
    """Run Flask in a background thread."""
    app.run(host="0.0.0.0", port=5000, use_reloader=False, threaded=True)


# ─── Graceful shutdown ──────────────────────────────────────
running = True


def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =============================================================
#  SETUP — initialize all subsystems
# =============================================================
def setup():
    logger.info("=" * 50)
    logger.info("  Central Station — Boot")
    logger.info("=" * 50)

    # Start Flask server in background thread
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    logger.info("  Flask server started on port 5000")

    lora_ok = lora_receiver.setup()
    fso_ok  = fso_receiver.setup()
    cam_ok  = camera.setup()
    gps_ok  = gps_parser.setup()
    log_ok  = data_logger.setup()

    logger.info("-" * 50)
    logger.info("  HTTP:   OK (port 5000)")
    logger.info("  LoRa:   %s", "OK" if lora_ok else "DISABLED/FAILED")
    logger.info("  FSO:    %s", "OK" if fso_ok else "DISABLED/FAILED")
    logger.info("  Camera: %s", "OK" if cam_ok else "DISABLED/FAILED")
    logger.info("  GPS:    %s", "OK" if gps_ok else "DISABLED/FAILED")
    logger.info("  Logger: %s", "OK" if log_ok else "FAILED")
    logger.info("-" * 50)

    if not log_ok:
        logger.error("Data logger failed — cannot continue")
        sys.exit(1)


# =============================================================
#  MAIN LOOP
# =============================================================
def loop():
    cycle = 0

    while running:
        cycle += 1
        t_start = time.time()
        now = datetime.now().strftime("%H:%M:%S")

        logger.info("── Cycle %d (%s) ──", cycle, now)

        # 1. Receive T1 data via LoRa
        t1_data = None
        try:
            t1_data = lora_receiver.receive()
            if t1_data:
                logger.info("  T1: pitch=%.1f roll=%.1f yaw=%.1f light=%d",
                            t1_data.get("pitch", 0), t1_data.get("roll", 0),
                            t1_data.get("yaw", 0), t1_data.get("light_intensity", 0))
            else:
                logger.info("  T1: no data")
        except Exception as e:
            logger.error("  T1 error: %s", e)

        # 2. Receive T2 data via WiFi HTTP POST
        t2_http = None
        try:
            t2_http = get_http_telemetry()
            if t2_http:
                logger.info("  T2 (WiFi): node=%s counter=%s pitch=%.1f",
                            t2_http.get("node", "?"), t2_http.get("counter", "?"),
                            t2_http.get("pitch", 0))
            else:
                logger.info("  T2 (WiFi): no data")
        except Exception as e:
            logger.error("  T2 (WiFi) error: %s", e)

        # 3. Receive T2 data via FSO (fallback)
        t2_fso = None
        try:
            t2_fso = fso_receiver.receive()
            if t2_fso:
                logger.info("  T2 (FSO): pitch=%.1f roll=%.1f yaw=%.1f",
                            t2_fso.get("pitch", 0), t2_fso.get("roll", 0),
                            t2_fso.get("yaw", 0))
            else:
                logger.info("  T2 (FSO): no data")
        except Exception as e:
            logger.error("  T2 (FSO) error: %s", e)

        # Merge T2 sources — prefer WiFi, fallback to FSO
        t2_data = t2_http if t2_http else t2_fso

        # 4. Capture sky image
        image_path = None
        try:
            image_path = camera.capture()
            if image_path:
                logger.info("  CAM: %s", image_path)
            else:
                logger.info("  CAM: no capture")
        except Exception as e:
            logger.error("  CAM error: %s", e)

        # 5. Read station GPS
        gps_data = None
        try:
            gps_data = gps_parser.read()
            if gps_data:
                logger.info("  GPS: %.6f, %.6f alt=%.1fm",
                            gps_data.get("lat", 0), gps_data.get("lon", 0),
                            gps_data.get("altitude", 0))
            else:
                logger.info("  GPS: no fix")
        except Exception as e:
            logger.error("  GPS error: %s", e)

        # 6. Log everything to CSV
        try:
            record = data_logger.log(
                t1_data=t1_data,
                t2_data=t2_data,
                gps_data=gps_data,
                image_path=image_path
            )
        except Exception as e:
            logger.error("  LOG error: %s", e)
            record = None

        # 7. Send merged packet downstream via LoRa
        try:
            downlink = {
                "station": "RPi",
                "cycle": cycle,
                "timestamp": datetime.now().isoformat(),
                "t1": t1_data if t1_data else {},
                "t2": t2_data if t2_data else {},
                "gps": gps_data if gps_data else {},
                "image": image_path or ""
            }
            # Remove internal keys before transmitting
            for src in (downlink["t1"], downlink["t2"]):
                src.pop("_source", None)

            sent = lora_receiver.transmit(downlink)
            logger.info("  TX: %s", "sent" if sent else "skipped")
        except Exception as e:
            logger.error("  TX error: %s", e)

        # 8. Sleep remainder of interval
        elapsed = time.time() - t_start
        sleep_time = max(0, config.LOOP_INTERVAL_S - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


# =============================================================
#  ENTRY POINT
# =============================================================
def main():
    try:
        setup()
        loop()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        camera.shutdown()
        gps_parser.shutdown()
        data_logger.shutdown()
        logger.info("Goodbye.")


if __name__ == "__main__":
    main()
