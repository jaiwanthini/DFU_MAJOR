import asyncio
import json
from datetime import datetime

import requests
from bleak import BleakScanner, BleakClient


# ============================================================
# DFU SMART INSOLE
# BLE RECEIVER
# ============================================================


DEVICE_NAME = "DFU_SMART_INSOLE"


SERVICE_UUID = (
    "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
)


CHARACTERISTIC_UUID = (
    "beb5483e-36e1-4688-b7f5-ea07361b26a8"
)


# ============================================================
# ANTIGRAVITY BACKEND
# ============================================================
#
# We will verify this endpoint with your actual backend.
#
# For now:
#

BACKEND_URL = (
    "http://127.0.0.1:8000/api/sensor-data"
)


# ============================================================
# FIND ESP32
# ============================================================

async def find_device():

    print()
    print("=" * 60)
    print("SCANNING FOR DFU ESP32")
    print("=" * 60)


    devices = await BleakScanner.discover(
        timeout=10
    )


    for device in devices:

        print(
            f"Device: {device.name} "
            f"| Address: {device.address}"
        )


        if device.name == DEVICE_NAME:

            print()
            print(
                "DFU ESP32 FOUND!"
            )

            return device


    return None


# ============================================================
# SEND DATA TO BACKEND
# ============================================================

def send_to_backend(data):

    try:

        response = requests.post(

            BACKEND_URL,

            json=data,

            timeout=3
        )


        print(
            f"Backend status: "
            f"{response.status_code}"
        )


        if response.text:

            print(
                f"Backend response: "
                f"{response.text}"
            )


    except requests.exceptions.ConnectionError:

        print(
            "Backend is not currently reachable."
        )

        print(
            f"Backend URL: {BACKEND_URL}"
        )


    except requests.exceptions.RequestException as error:

        print(
            f"Backend error: {error}"
        )


# ============================================================
# PROCESS BLE DATA
# ============================================================

def sensor_notification(
    sender,
    data
):

    try:

        # BLE bytes → text

        message = data.decode(
            "utf-8"
        )


        print()
        print(
            "=" * 60
        )

        print(
            "BLE SENSOR PACKET"
        )

        print(
            "=" * 60
        )

        print(message)


        # JSON → Python dictionary

        sensor_data = json.loads(
            message
        )


        # Add PC reception time

        sensor_data[
            "received_at"
        ] = datetime.now().isoformat()


        # ====================================================
        # DISPLAY FOOT DATA
        # ====================================================

        foot = sensor_data.get(
            "foot",
            {}
        )


        print()
        print(
            "FOOT SENSORS"
        )


        print(
            f"Big Toe Pressure : "
            f"{foot.get('pressure_big_toe')}"
        )


        print(
            f"Heel Pressure    : "
            f"{foot.get('pressure_heel')}"
        )


        print(
            f"Medial Pressure  : "
            f"{foot.get('pressure_medial')}"
        )


        print(
            f"Lateral Pressure : "
            f"{foot.get('pressure_lateral')}"
        )


        print(
            f"Temperature      : "
            f"{foot.get('plantar_temperature')} °C"
        )


        # ====================================================
        # DISPLAY FINGER DATA
        # ====================================================

        finger = sensor_data.get(
            "finger",
            {}
        )


        print()
        print(
            "FINGER SENSORS"
        )


        print(
            f"SpO2             : "
            f"{finger.get('spo2')} %"
        )


        print(
            f"Heart Rate       : "
            f"{finger.get('heart_rate')} BPM"
        )


        # ====================================================
        # SEND TO BACKEND
        # ====================================================

        send_to_backend(
            sensor_data
        )


    except json.JSONDecodeError:

        print(
            "ERROR: Invalid JSON received."
        )


        print(
            message
        )


    except Exception as error:

        print(
            f"ERROR processing BLE packet: "
            f"{error}"
        )


# ============================================================
# CONNECT
# ============================================================

async def connect_to_esp32():

    device = await find_device()


    if device is None:

        print()
        print(
            "=" * 60
        )

        print(
            "ESP32 NOT FOUND"
        )

        print(
            "=" * 60
        )


        print(
            "Check:"
        )

        print(
            "1. ESP32 is powered."
        )

        print(
            "2. ESP32 code was uploaded."
        )

        print(
            "3. Bluetooth is enabled."
        )

        print(
            "4. Device is advertising."
        )


        return


    print()
    print(
        "Connecting to ESP32..."
    )


    try:

        async with BleakClient(
            device.address
        ) as client:


            print()
            print(
                "Connected:",
                client.is_connected
            )


            if not client.is_connected:

                print(
                    "Could not connect."
                )

                return


            print()
            print(
                "Starting BLE notifications..."
            )


            await client.start_notify(

                CHARACTERISTIC_UUID,

                sensor_notification
            )


            print()
            print(
                "=" * 60
            )

            print(
                "LIVE DFU SENSOR STREAM"
            )

            print(
                "=" * 60
            )


            print(
                "Waiting for sensor data..."
            )


            while True:

                await asyncio.sleep(
                    1
                )


    except Exception as error:

        print()
        print(
            "BLE CONNECTION ERROR:"
        )

        print(error)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            connect_to_esp32()
        )


    except KeyboardInterrupt:

        print()
        print(
            "BLE receiver stopped."
        )