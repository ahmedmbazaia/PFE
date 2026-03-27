#!/usr/bin/env python3
# =============================================================
#  Central Station — Main Loop (Raspberry Pi)
#  Receives telemetry from T1 (LoRa) and T2 (FSO),
#  captures sky images, logs everything, and sends data
#  downstream via LoRa to the ground laptop.
# =============================================================

import json
import logging
import signal
import sys
import time
from datetime import datetime

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

    lora_ok = lora_receiver.setup()
    fso_ok  = fso_receiver.setup()
    cam_ok  = camera.setup()
    gps_ok  = gps_parser.setup()
    log_ok  = data_logger.setup()

    logger.info("-" * 50)
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

        # 2. Receive T2 data via FSO
        t2_data = None
        try:
            t2_data = fso_receiver.receive()
            if t2_data:
                logger.info("  T2: pitch=%.1f roll=%.1f yaw=%.1f lat=%.6f lon=%.6f",
                            t2_data.get("pitch", 0), t2_data.get("roll", 0),
                            t2_data.get("yaw", 0), t2_data.get("lat", 0),
                            t2_data.get("lon", 0))
            else:
                logger.info("  T2: no data")
        except Exception as e:
            logger.error("  T2 error: %s", e)

        # 3. Capture sky image
        image_path = None
        try:
            image_path = camera.capture()
            if image_path:
                logger.info("  CAM: %s", image_path)
            else:
                logger.info("  CAM: no capture")
        except Exception as e:
            logger.error("  CAM error: %s", e)

        # 4. Read station GPS
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

        # 5. Log everything to CSV
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

        # 6. Send merged packet downstream via LoRa
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

        # 7. Sleep remainder of interval
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
