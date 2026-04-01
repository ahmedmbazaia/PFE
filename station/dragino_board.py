# =============================================================
#  Dragino LoRa GPS HAT v1.4 — Board Configuration
#  Custom pin mapping for pySX127x on Raspberry Pi
#
#  Hardware SPI0 (CE0 = GPIO8)
#  DIO0:  GPIO4   (RxDone / TxDone interrupt)
#  RST:   GPIO11
#  DIO1:  GPIO23
#  DIO2:  GPIO24
#  DIO3:  GPIO25
# =============================================================

import RPi.GPIO as GPIO
import spidev


class BOARD:
    # SPI
    SPI_BUS = 0
    SPI_CS = 0  # CE0 = GPIO8

    # GPIO pins (BCM numbering)
    DIO0 = 4
    DIO1 = 23
    DIO2 = 24
    DIO3 = 25
    RST = 11

    # SPI handle
    spi = None

    @staticmethod
    def setup():
        """Configure GPIO and SPI for the Dragino HAT."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # DIO pins as inputs (directly from SX1278)
        GPIO.setup(BOARD.DIO0, GPIO.IN)
        GPIO.setup(BOARD.DIO1, GPIO.IN)
        GPIO.setup(BOARD.DIO2, GPIO.IN)
        GPIO.setup(BOARD.DIO3, GPIO.IN)

        # Reset pin as output
        GPIO.setup(BOARD.RST, GPIO.OUT)
        GPIO.output(BOARD.RST, GPIO.HIGH)

        # SPI bus
        BOARD.spi = spidev.SpiDev()
        BOARD.spi.open(BOARD.SPI_BUS, BOARD.SPI_CS)
        BOARD.spi.max_speed_hz = 5000000

    @staticmethod
    def teardown():
        """Release GPIO and SPI resources."""
        if BOARD.spi is not None:
            BOARD.spi.close()
            BOARD.spi = None
        GPIO.cleanup()

    @staticmethod
    def reset():
        """Pulse the RST pin to reset the SX1278."""
        import time
        GPIO.output(BOARD.RST, GPIO.LOW)
        time.sleep(0.01)
        GPIO.output(BOARD.RST, GPIO.HIGH)
        time.sleep(0.01)

    @staticmethod
    def add_event_detect(dio_pin, callback):
        """Register a rising-edge interrupt on a DIO pin."""
        GPIO.add_event_detect(dio_pin, GPIO.RISING, callback=callback)

    @staticmethod
    def add_events(cb_dio0=None, cb_dio1=None, cb_dio2=None, cb_dio3=None):
        """Register callbacks for all DIO interrupt lines."""
        if cb_dio0:
            BOARD.add_event_detect(BOARD.DIO0, cb_dio0)
        if cb_dio1:
            BOARD.add_event_detect(BOARD.DIO1, cb_dio1)
        if cb_dio2:
            BOARD.add_event_detect(BOARD.DIO2, cb_dio2)
        if cb_dio3:
            BOARD.add_event_detect(BOARD.DIO3, cb_dio3)

    @staticmethod
    def led_on(value=1):
        """No onboard LED on Dragino HAT — no-op."""
        pass

    @staticmethod
    def led_off():
        """No onboard LED on Dragino HAT — no-op."""
        pass
