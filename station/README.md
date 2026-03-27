# Central Station — Raspberry Pi

Ground station that aggregates telemetry from both telescopes, captures sky images, logs all data, and relays to the ground laptop.

## Architecture

```
T1 (ESP32-CAM) ──LoRa──► RPi ──LoRa──► Ground Laptop
T2 (ESP32)     ──FSO───► RPi
                          │
                     Camera + GPS
                          │
                    data/logs/mission_log.csv
                    data/images/sky_*.jpeg
```

## Modules

| File | Role |
|------|------|
| `config.py` | All pins, ports, paths, feature flags |
| `lora_receiver.py` | SX1278 LoRa RX (T1) and TX (downlink) |
| `fso_receiver.py` | BPW34 + MCP3008 OOK decoder (T2) |
| `camera.py` | RPi camera capture |
| `gps_parser.py` | NMEA GPS parsing via UART |
| `data_logger.py` | CSV logging with merged records |
| `main.py` | Main loop orchestrator |

## Feature Flags (config.py)

```python
LORA_ENABLED   = True
FSO_ENABLED    = True
CAMERA_ENABLED = True
GPS_ENABLED    = True
```

## Hardware

| Component | Interface | Config |
|-----------|-----------|--------|
| SX1278 LoRa HAT | SPI0/CE0 | RST=GPIO25, DIO0=GPIO24 |
| BPW34 + MCP3008 | SPI1/CE0 | ADC channel 0 |
| RPi Camera v2 | CSI | 1920x1080 JPEG |
| GPS module | UART | /dev/serial0, 9600 baud |

## Loop Cycle (every 2s)

1. Receive T1 JSON via LoRa
2. Receive T2 JSON via FSO OOK
3. Capture sky image
4. Read station GPS
5. Merge and log to CSV
6. Transmit merged packet via LoRa downlink
7. Print status

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Enable SPI and camera in raspi-config
sudo raspi-config

# Run
python3 main.py
```

## CSV Output

All data is logged to `data/logs/mission_log.csv` with columns:

```
rpi_timestamp, t1_timestamp, t1_pitch, t1_roll, t1_yaw, t1_light_intensity,
t1_sky_distance_mm, t1_baseline_distance_mm, t1_motor_angle,
t2_timestamp, t2_pitch, t2_roll, t2_yaw, t2_light_intensity,
t2_sky_distance_mm, t2_baseline_distance_mm, t2_motor_angle,
t2_lat, t2_lon, t2_altitude,
gps_lat, gps_lon, gps_altitude, gps_time, image_path
```
