# LoRa Flask Dashboard

Simple Flask web UI for showing the latest packet received by the Raspberry Pi SX1278 LoRa HAT.

## Run

```bash
pip install -r station/requirements.txt
python dashboard/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## What it does

- Starts the existing `station/lora_receiver.py` code in a background thread
- Receives JSON packets sent by the TTGO LoRa board
- Exposes the latest packet at `/api/latest`
- Refreshes the dashboard automatically every 2 seconds

## Important

Make sure the sender and receiver use the same LoRa settings:

- Frequency: `433 MHz`
- Spreading factor: `7`
- Bandwidth: `125 kHz`
- Coding rate: `4/5`
