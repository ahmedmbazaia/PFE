# =============================================================
#  Data Logger — Merge all sources and log to CSV
#  Appends records to data/logs/mission_log.csv
# =============================================================

import csv
import logging
import os
from datetime import datetime

import config

logger = logging.getLogger(__name__)

# CSV column order
FIELDNAMES = [
    "rpi_timestamp",
    # T1 fields (from LoRa)
    "t1_timestamp", "t1_pitch", "t1_roll", "t1_yaw",
    "t1_light_intensity", "t1_sky_distance_mm", "t1_baseline_distance_mm",
    "t1_motor_angle",
    # T2 fields (from FSO)
    "t2_timestamp", "t2_pitch", "t2_roll", "t2_yaw",
    "t2_light_intensity", "t2_sky_distance_mm", "t2_baseline_distance_mm",
    "t2_motor_angle", "t2_lat", "t2_lon", "t2_altitude",
    # Station fields
    "gps_lat", "gps_lon", "gps_altitude", "gps_time",
    "image_path"
]

_writer = None
_file = None


def setup():
    """Create log directory and open CSV file with header."""
    global _writer, _file

    os.makedirs(config.LOG_DIR, exist_ok=True)

    file_exists = os.path.isfile(config.LOG_FILE) and os.path.getsize(config.LOG_FILE) > 0

    try:
        _file = open(config.LOG_FILE, "a", newline="", encoding="utf-8")
        _writer = csv.DictWriter(_file, fieldnames=FIELDNAMES, extrasaction="ignore")

        if not file_exists:
            _writer.writeheader()
            _file.flush()

        logger.info("Data logger ready — %s", config.LOG_FILE)
        return True

    except Exception as e:
        logger.error("Data logger init failed: %s", e)
        return False


def _prefix_dict(data, prefix):
    """Add a prefix to all keys in a dict (e.g., 't1_pitch')."""
    if data is None:
        return {}
    return {f"{prefix}{k}": v for k, v in data.items() if not k.startswith("_")}


def log(t1_data=None, t2_data=None, gps_data=None, image_path=None):
    """
    Merge data from all sources into a single CSV row.
    Any source can be None — missing fields will be empty.
    Returns the merged record dict.
    """
    if _writer is None:
        logger.warning("Logger not initialized — skipping")
        return None

    try:
        record = {"rpi_timestamp": datetime.now().isoformat()}

        # Telescope 1 data (from LoRa)
        if t1_data:
            record.update(_prefix_dict(t1_data, "t1_"))

        # Telescope 2 data (from FSO)
        if t2_data:
            record.update(_prefix_dict(t2_data, "t2_"))

        # Station GPS
        if gps_data:
            record["gps_lat"] = gps_data.get("lat", "")
            record["gps_lon"] = gps_data.get("lon", "")
            record["gps_altitude"] = gps_data.get("altitude", "")
            record["gps_time"] = gps_data.get("gps_time", "")

        # Image reference
        if image_path:
            record["image_path"] = image_path

        _writer.writerow(record)
        _file.flush()

        logger.debug("Logged record at %s", record["rpi_timestamp"])
        return record

    except Exception as e:
        logger.error("Logging error: %s", e)
        return None


def shutdown():
    """Close the CSV file."""
    global _writer, _file
    if _file is not None:
        try:
            _file.close()
        except Exception:
            pass
        _file = None
        _writer = None
        logger.info("Data logger closed")
