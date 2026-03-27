# =============================================================
#  GPS Parser — NMEA sentences from UART GPS module
#  Extracts lat, lon, altitude, and UTC timestamp
# =============================================================

import logging

import config

logger = logging.getLogger(__name__)

_serial = None


def setup():
    """Open the GPS serial port."""
    global _serial

    if not config.GPS_ENABLED:
        logger.info("GPS disabled in config")
        return False

    try:
        import serial

        _serial = serial.Serial(
            port=config.GPS_SERIAL_PORT,
            baudrate=config.GPS_BAUD_RATE,
            timeout=config.GPS_TIMEOUT
        )
        logger.info("GPS initialized on %s @ %d baud",
                     config.GPS_SERIAL_PORT, config.GPS_BAUD_RATE)
        return True

    except ImportError:
        logger.warning("pyserial not available — using stub mode")
        _serial = None
        return False
    except Exception as e:
        logger.error("GPS init failed: %s", e)
        _serial = None
        return False


def _parse_nmea_coord(raw, direction):
    """
    Convert NMEA coordinate (DDMM.MMMM or DDDMM.MMMM) to decimal degrees.
    Direction is 'N'/'S' or 'E'/'W'.
    """
    if not raw or not direction:
        return 0.0

    # Split into degrees and minutes
    dot = raw.index(".")
    degrees = int(raw[:dot - 2])
    minutes = float(raw[dot - 2:])
    decimal = degrees + minutes / 60.0

    if direction in ("S", "W"):
        decimal = -decimal

    return decimal


def _parse_gga(sentence):
    """
    Parse a GGA sentence for position and altitude.
    $GPGGA,time,lat,N/S,lon,E/W,fix,sats,hdop,alt,M,...
    """
    parts = sentence.split(",")
    if len(parts) < 10 or parts[6] == "0":
        return None  # no fix

    return {
        "gps_time": parts[1],
        "lat": _parse_nmea_coord(parts[2], parts[3]),
        "lon": _parse_nmea_coord(parts[4], parts[5]),
        "altitude": float(parts[9]) if parts[9] else 0.0,
        "satellites": int(parts[7]) if parts[7] else 0
    }


def read():
    """
    Read available NMEA sentences and return the latest fix.
    Returns dict with lat, lon, altitude, gps_time — or None if no fix.
    """
    if _serial is None:
        return None

    try:
        result = None

        # Read all available lines, keep the latest valid GGA
        while _serial.in_waiting > 0:
            line = _serial.readline().decode("ascii", errors="replace").strip()

            if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                parsed = _parse_gga(line)
                if parsed is not None:
                    result = parsed

        if result:
            logger.debug("GPS fix: %.6f, %.6f, alt=%.1fm",
                         result["lat"], result["lon"], result["altitude"])
        return result

    except Exception as e:
        logger.error("GPS read error: %s", e)
        return None


def shutdown():
    """Close the serial port."""
    global _serial
    if _serial is not None:
        try:
            _serial.close()
        except Exception:
            pass
        _serial = None
        logger.info("GPS serial closed")
