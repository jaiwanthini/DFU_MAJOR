"""
Paddrishti - BLE Receiver + Website API
---------------------------------------
1. Scans for the ESP32 named "Paddrishti - ESP32".
2. Connects over BLE.
3. Receives JSON notifications.
4. Applies FSR calibration.
5. Stores the latest reading.
6. Exposes it to your website at:

      GET http://localhost:5000/api/sensor-data

Install:
    pip install bleak flask

Run:
    python ble_receiver.py

Then open:
    http://localhost:5000/api/sensor-data
"""

import asyncio
import json
import threading
import urllib.request
from datetime import datetime, timezone

from bleak import BleakClient, BleakScanner
from flask import Flask, jsonify

try:
    from sensor_calibration import calibrate_packet
except ImportError:
    from .sensor_calibration import calibrate_packet


DEVICE_NAME = "Paddrishti - ESP32"
SERVICE_UUID = "7f6d0001-5a4e-4c3a-9b2e-123456789001"
CHARACTERISTIC_UUID = "7f6d0002-5a4e-4c3a-9b2e-123456789002"

BACKEND_FORWARD_URL = "http://127.0.0.1:5500/api/hardware-reading"

app = Flask(__name__)

latest_data = {
    "connected": False,
    "bigToe": 0,
    "ballFoot": 0,
    "outerFoot": 0,
    "heel": 0,
    "temperature": 0,
    "spo2": 0,
    "pressure": {
        "bigToe": 0,
        "ballFoot": 0,
        "outerFoot": 0,
        "heel": 0,
    },
    "receivedAt": None,
}


def forward_to_backend(data):
    """Maps calibrated ESP32 packet to backend model schema and forwards it."""
    try:
        pressure = data.get("pressure", {})
        # Map 0-100% calibrated intensities to model's 0-1000 scale
        fsr1 = float(pressure.get("bigToe", 0.0)) * 10.0
        fsr2 = float(pressure.get("ballFoot", 0.0)) * 10.0
        fsr3 = float(pressure.get("outerFoot", 0.0)) * 10.0
        fsr4 = float(pressure.get("heel", 0.0)) * 10.0

        payload = {
            "fsr1": fsr1,
            "fsr2": fsr2,
            "fsr3": fsr3,
            "fsr4": fsr4,
            "temperature": float(data.get("temperature", 36.5)),
            "spo2": float(data.get("spo2", 98.0)),
            "heart_rate": float(data.get("heart_rate", 72.0)),
            "source": "hardware",
            "connected": True
        }

        req = urllib.request.Request(
            BACKEND_FORWARD_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=0.5)
    except Exception:
        # Non-blocking: backend might be restarting or momentarily unreachable
        pass


def update_data(data):
    global latest_data

    data = calibrate_packet(data)
    data["receivedAt"] = datetime.now(timezone.utc).isoformat()
    data["connected"] = True

    latest_data = data

    print(
        f"FSR  BigToe={data['bigToe']}  "
        f"Ball={data['ballFoot']}  "
        f"Outer={data['outerFoot']}  "
        f"Heel={data['heel']}  "
        f"| Temp={data['temperature']} C  "
        f"| SpO2={data['spo2']}%"
    )

    forward_to_backend(data)


def notification_handler(sender, raw_data):
    try:
        text = raw_data.decode("utf-8")
        packet = json.loads(text)
        update_data(packet)
    except Exception as exc:
        print("Invalid BLE packet:", exc)


async def ble_worker():
    global latest_data

    while True:
        try:
            print(f"Scanning for {DEVICE_NAME}...")

            device = await BleakScanner.find_device_by_name(
                DEVICE_NAME,
                timeout=10.0
            )

            if device is None:
                print("ESP32 not found. Retrying in 3 seconds...")
                await asyncio.sleep(3)
                continue

            print(f"Found {device.name} [{device.address}]")
            print("Connecting...")

            async with BleakClient(device) as client:
                latest_data["connected"] = True
                print("Connected to ESP32.")

                await client.start_notify(
                    CHARACTERISTIC_UUID,
                    notification_handler
                )

                print("Receiving BLE sensor data...")

                while client.is_connected:
                    await asyncio.sleep(1)

        except Exception as exc:
            latest_data["connected"] = False
            print("BLE error:", exc)

        print("Disconnected. Reconnecting in 3 seconds...")
        await asyncio.sleep(3)


@app.after_request
def add_cors_headers(response):
    # Allows a separately hosted frontend to call this local API.
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.get("/api/sensor-data")
def sensor_data():
    return jsonify(latest_data)


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "bleConnected": latest_data["connected"]
    })


def start_ble_thread():
    def runner():
        asyncio.run(ble_worker())

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()


if __name__ == "__main__":
    start_ble_thread()

    print()
    print("Paddrishti BLE receiver started.")
    print("Website API: http://localhost:5000/api/sensor-data")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )
