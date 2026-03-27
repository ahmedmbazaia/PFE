# =============================================================
#  LoRa Receiver — SX1278 HAT via SPI
#  Listens for JSON telemetry from Telescope 1
#  Also provides downlink transmit for ground station
# =============================================================

import json
import logging
import time

import config

logger = logging.getLogger(__name__)

# LoRa module handle (initialized in setup)
_lora = None


def setup():
    """Initialize the SX1278 LoRa module via SPI."""
    global _lora

    if not config.LORA_ENABLED:
        logger.info("LoRa disabled in config")
        return False

    try:
        from SX127x.LoRa import LoRa as LoRaDriver
        from SX127x.board_config import BOARD

        BOARD.setup()

        class LoRaReceiver(LoRaDriver):
            def __init__(self):
                super().__init__(verbose=False)
                self.set_mode(0x01)         # standby
                self.set_freq(config.LORA_FREQUENCY / 1e6)
                self.set_pa_config(max_power=0x04, output_power=config.LORA_TX_POWER)
                self.set_bw(7)              # 125 kHz
                self.set_spreading_factor(7)
                self.set_coding_rate(5)     # 4/5
                self.set_rx_crc(True)
                self._rx_buffer = None

            def on_rx_done(self):
                payload = self.read_payload(nocheck=True)
                self._rx_buffer = bytes(payload)
                self.clear_irq_flags(RxDone=1)
                self.set_mode(0x05)  # back to RX continuous

            def get_packet(self):
                pkt = self._rx_buffer
                self._rx_buffer = None
                return pkt

        _lora = LoRaReceiver()
        _lora.set_mode(0x05)  # RX continuous
        logger.info("LoRa SX1278 initialized — listening on %.1f MHz", config.LORA_FREQUENCY / 1e6)
        return True

    except ImportError:
        logger.warning("SX127x library not available — using stub mode")
        _lora = None
        return False
    except Exception as e:
        logger.error("LoRa init failed: %s", e)
        _lora = None
        return False


def receive():
    """
    Check for an incoming LoRa packet from T1.
    Returns parsed dict or None if no data available.
    """
    if _lora is None:
        return None

    try:
        raw = _lora.get_packet()
        if raw is None:
            return None

        text = raw.decode("utf-8", errors="replace").strip()
        logger.debug("LoRa RX: %s", text)
        data = json.loads(text)
        data["_source"] = "lora"
        return data

    except json.JSONDecodeError as e:
        logger.warning("LoRa JSON parse error: %s", e)
        return None
    except Exception as e:
        logger.error("LoRa receive error: %s", e)
        return None


def transmit(data):
    """
    Send a JSON payload via LoRa downlink to ground station.
    """
    if _lora is None:
        logger.debug("LoRa TX skipped — not initialized")
        return False

    try:
        payload = json.dumps(data, separators=(",", ":"))
        _lora.set_mode(0x01)         # standby
        _lora.write_payload(list(payload.encode("utf-8")))
        _lora.set_mode(0x03)         # TX mode
        # Wait for TX done (with timeout)
        start = time.time()
        while time.time() - start < 5.0:
            irq = _lora.get_irq_flags()
            if irq.get("tx_done"):
                break
            time.sleep(0.01)
        _lora.clear_irq_flags(TxDone=1)
        _lora.set_mode(0x05)         # back to RX continuous
        logger.debug("LoRa TX: %d bytes sent", len(payload))
        return True

    except Exception as e:
        logger.error("LoRa transmit error: %s", e)
        return False
