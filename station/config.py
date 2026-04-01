# =============================================================
#  Central Station — Configuration
#  All pin numbers, serial ports, paths, and feature flags
# =============================================================

import os

# ─── Feature flags ───────────────────────────────────────────
LORA_ENABLED   = True
FSO_ENABLED    = True
CAMERA_ENABLED = True
GPS_ENABLED    = True

# ─── LoRa SX1278 — Dragino LoRa GPS HAT v1.4 ────────────────
LORA_SPI_BUS    = 0
LORA_SPI_CS     = 0       # CE0 = GPIO8
LORA_FREQUENCY  = 433000000  # 433 MHz — must match telescopes
LORA_RST_PIN    = 11      # BCM GPIO 11
LORA_DIO0_PIN   = 4       # BCM GPIO 4  — RxDone / TxDone interrupt
LORA_DIO1_PIN   = 23      # BCM GPIO 23
LORA_DIO2_PIN   = 24      # BCM GPIO 24
LORA_DIO3_PIN   = 25      # BCM GPIO 25
LORA_TX_POWER   = 17      # dBm for downlink

# ─── FSO receiver (BPW34 via ADC — MCP3008 on SPI1) ─────────
FSO_SPI_BUS     = 1
FSO_SPI_CS      = 0       # CE0 on SPI1
FSO_ADC_CHANNEL = 0       # MCP3008 channel 0
FSO_BITRATE     = 1000    # 1 kbps OOK — must match T2
FSO_THRESHOLD   = 512     # ADC threshold for HIGH (0–1023)
FSO_SYNC_BYTE   = 0x7E
FSO_PREAMBLE    = 0xAA

# ─── RPi Camera ─────────────────────────────────────────────
CAMERA_RESOLUTION = (1920, 1080)
CAMERA_FORMAT     = "jpeg"

# ─── GPS module (UART) ──────────────────────────────────────
GPS_SERIAL_PORT = "/dev/serial0"
GPS_BAUD_RATE   = 9600
GPS_TIMEOUT     = 1.0  # seconds

# ─── Data paths ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
LOG_DIR     = os.path.join(DATA_DIR, "logs")
IMAGE_DIR   = os.path.join(DATA_DIR, "images")
LOG_FILE    = os.path.join(LOG_DIR, "mission_log.csv")

# ─── Loop timing ────────────────────────────────────────────
LOOP_INTERVAL_S = 2.0
