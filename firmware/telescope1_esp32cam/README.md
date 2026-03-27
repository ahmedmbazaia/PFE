# Telescope 1 — ESP32-CAM Firmware

PlatformIO project for Telescope 1 node. Uses an ESP32-CAM board as a regular ESP32 (no camera).

## Hardware

| Component | Interface | Pins |
|-----------|-----------|------|
| MPU9250 IMU | I2C (0x68) | SDA=14, SCL=15 |
| VL53L0X ToF | I2C (0x29) | shared bus |
| BPW34 photodiode | ADC | GPIO 33 |
| DRV8825 stepper | GPIO | STEP=12, DIR=13, EN=2 |
| SX1278 LoRa | SPI | SCK=18, MISO=19, MOSI=23, NSS=5, RST=16, DIO0=4 |
| WiFi | internal | NTP sync only |

## Feature Flags

```cpp
#define MOTOR_ENABLED   false
#define FSO_ENABLED     false
#define CAMERA_ENABLED  false
```

## Loop Cycle (every 2s)

1. Read MPU9250 → pitch, roll, yaw
2. Read BPW34 → light intensity (0–4095)
3. Read VL53L0X → sky distance + baseline distance (mm)
4. Get NTP timestamp
5. Build JSON payload
6. Transmit via LoRa

## JSON Payload

```json
{
  "node": "T1",
  "timestamp": "HH:MM:SS",
  "pitch": 0.0,
  "roll": 0.0,
  "yaw": 0.0,
  "light_intensity": 0,
  "sky_distance_mm": 0,
  "baseline_distance_mm": 0,
  "motor_angle": 0
}
```

## Build & Flash

```bash
pio run -e esp32cam
pio run -e esp32cam -t upload
pio device monitor -b 115200
```
