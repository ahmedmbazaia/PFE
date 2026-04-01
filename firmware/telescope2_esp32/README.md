# Telescope 2 — ESP32 NodeMCU Firmware

PlatformIO project for Telescope 2 node. Standard ESP32 with GPS and FSO laser transmitter.

## Hardware

| Component | Interface | Pins |
|-----------|-----------|------|
| MPU9250 IMU | I2C (0x68) | SDA=21, SCL=22 |
| VL53L0X ToF | I2C (0x29) | shared bus |
| BPW34 photodiode | ADC | GPIO 34 |
| DRV8825 stepper | GPIO | STEP=25, DIR=26, EN=27 |
| GPS module | UART2 | RX=16, TX=17 (9600 baud) |
| FSO laser TX | GPIO (OOK) | GPIO 4 |

## Feature Flags

```cpp
#define MOTOR_ENABLED  false
#define FSO_ENABLED    true
#define GPS_ENABLED    true
#define LORA_ENABLED   true
#define TEST_LORA_ONLY true
```

With `TEST_LORA_ONLY` enabled, the TTGO sends a simple LoRa JSON test packet every 2 seconds so you can verify reception on the Raspberry Pi LoRa HAT without depending on sensor data.

## Loop Cycle (every 2s)

1. Read MPU9250 → pitch, roll, yaw
2. Read BPW34 → light intensity (0–4095)
3. Read VL53L0X → sky distance + baseline distance (mm)
4. Read GPS → lat, lon, altitude
5. Build JSON payload
6. Transmit via FSO laser (OOK at 1 kbps)

## JSON Payload

```json
{
  "node": "T2",
  "timestamp": 12345,
  "pitch": 0.0,
  "roll": 0.0,
  "yaw": 0.0,
  "light_intensity": 0,
  "sky_distance_mm": 0,
  "baseline_distance_mm": 0,
  "motor_angle": 0,
  "lat": 0.0,
  "lon": 0.0,
  "altitude": 0.0
}
```

## Test LoRa Payload

When `TEST_LORA_ONLY` is enabled, the board sends:

```json
{
  "node": "TTGO",
  "counter": 1,
  "message": "hello from ttgo",
  "timestamp_ms": 2000,
  "lora_freq": 433
}
```

## FSO OOK Protocol

The FSO transmitter uses On-Off Keying at 1 kbps (1000 us per bit):

| Field | Size | Description |
|-------|------|-------------|
| Preamble | 4 bytes | `0xAA` repeated for clock sync |
| Sync | 1 byte | `0x7E` marks payload start |
| Length | 2 bytes | Payload length, big-endian |
| Payload | N bytes | JSON ASCII data |
| Checksum | 1 byte | XOR of all payload bytes |

Each byte is framed with a start bit (HIGH) and stop bit (LOW), 8 data bits MSB-first.

## Build & Flash

```bash
pio run -e esp32dev
pio run -e esp32dev -t upload
pio device monitor -b 115200
```
