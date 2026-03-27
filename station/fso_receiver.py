# =============================================================
#  FSO Receiver — BPW34 photodiode via MCP3008 ADC
#  Decodes OOK signal at 1 kbps from Telescope 2
#  Protocol: [4x 0xAA preamble] [0x7E sync] [2B len] [payload] [XOR checksum]
# =============================================================

import json
import logging
import time

import config

logger = logging.getLogger(__name__)

_spi = None
_adc = None


def setup():
    """Initialize the MCP3008 ADC on SPI for BPW34 photodiode reading."""
    global _spi, _adc

    if not config.FSO_ENABLED:
        logger.info("FSO disabled in config")
        return False

    try:
        import spidev
        import RPi.GPIO as GPIO

        _spi = spidev.SpiDev()
        _spi.open(config.FSO_SPI_BUS, config.FSO_SPI_CS)
        _spi.max_speed_hz = 1000000  # 1 MHz SPI clock for MCP3008
        _spi.mode = 0

        logger.info("FSO receiver initialized — MCP3008 on SPI%d CS%d, ch%d",
                     config.FSO_SPI_BUS, config.FSO_SPI_CS, config.FSO_ADC_CHANNEL)
        return True

    except ImportError:
        logger.warning("spidev/RPi.GPIO not available — using stub mode")
        return False
    except Exception as e:
        logger.error("FSO init failed: %s", e)
        return False


def _read_adc():
    """Read a single sample from the MCP3008 ADC channel (0–1023)."""
    if _spi is None:
        return 0
    ch = config.FSO_ADC_CHANNEL
    cmd = [1, (8 + ch) << 4, 0]
    reply = _spi.xfer2(cmd)
    value = ((reply[1] & 0x03) << 8) | reply[2]
    return value


def _read_bit():
    """Sample one bit at the OOK bitrate. Returns 1 or 0."""
    value = _read_adc()
    return 1 if value >= config.FSO_THRESHOLD else 0


def _wait_for_preamble(timeout=2.0):
    """
    Wait for the preamble pattern (0xAA = 10101010 repeating).
    Returns True if preamble detected within timeout.
    """
    bit_period = 1.0 / config.FSO_BITRATE
    start = time.time()
    consecutive_toggles = 0
    last_bit = -1

    while time.time() - start < timeout:
        bit = _read_bit()
        if last_bit != -1 and bit != last_bit:
            consecutive_toggles += 1
        else:
            consecutive_toggles = 0
        last_bit = bit

        if consecutive_toggles >= 16:  # 2 bytes of alternating bits
            return True

        time.sleep(bit_period * 0.5)  # oversample at 2x

    return False


def _receive_byte():
    """
    Receive one framed byte: [start bit] [8 data bits MSB-first] [stop bit].
    Returns the byte value, or -1 on framing error.
    """
    bit_period = 1.0 / config.FSO_BITRATE

    # Wait for start bit (HIGH)
    timeout = time.time() + bit_period * 12
    while _read_bit() == 0:
        if time.time() > timeout:
            return -1
        time.sleep(bit_period * 0.25)

    # Sample at center of start bit
    time.sleep(bit_period * 0.5)

    # Read 8 data bits (MSB first)
    value = 0
    for _ in range(8):
        time.sleep(bit_period)
        bit = _read_bit()
        value = (value << 1) | bit

    # Stop bit (LOW) — just wait through it
    time.sleep(bit_period)

    return value


def receive():
    """
    Attempt to receive and decode a full FSO packet from T2.
    Returns parsed dict or None if no valid packet received.

    Non-blocking: returns None quickly if no preamble detected.
    """
    if _spi is None:
        return None

    try:
        # Look for preamble (short timeout so main loop isn't blocked)
        if not _wait_for_preamble(timeout=0.5):
            return None

        # Look for sync byte
        sync = _receive_byte()
        if sync != config.FSO_SYNC_BYTE:
            logger.debug("FSO: bad sync byte 0x%02X", sync if sync >= 0 else 0)
            return None

        # Read length (2 bytes, big-endian)
        len_hi = _receive_byte()
        len_lo = _receive_byte()
        if len_hi < 0 or len_lo < 0:
            logger.warning("FSO: failed to read length bytes")
            return None
        payload_len = (len_hi << 8) | len_lo

        if payload_len == 0 or payload_len > 1024:
            logger.warning("FSO: invalid payload length %d", payload_len)
            return None

        # Read payload bytes
        payload = bytearray()
        for i in range(payload_len):
            b = _receive_byte()
            if b < 0:
                logger.warning("FSO: byte %d/%d read error", i, payload_len)
                return None
            payload.append(b)

        # Read and verify checksum
        checksum_rx = _receive_byte()
        checksum_calc = 0
        for b in payload:
            checksum_calc ^= b

        if checksum_rx != checksum_calc:
            logger.warning("FSO: checksum mismatch (rx=0x%02X calc=0x%02X)",
                           checksum_rx, checksum_calc)
            return None

        # Decode JSON
        text = payload.decode("utf-8", errors="replace")
        logger.debug("FSO RX: %s", text)
        data = json.loads(text)
        data["_source"] = "fso"
        return data

    except json.JSONDecodeError as e:
        logger.warning("FSO JSON parse error: %s", e)
        return None
    except Exception as e:
        logger.error("FSO receive error: %s", e)
        return None
