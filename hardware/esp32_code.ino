/*
  Paddrishti - ESP32 Sensor Firmware
  --------------------------------
  FSR connections:
    GPIO 32 -> Big Toe
    GPIO 34 -> Ball Foot
    GPIO 35 -> Outer Foot
    GPIO 33 -> Heel

  Temperature and SpO2 are for development/demo purposes.
  The ESP32 sends sensor data over BLE as JSON notifications.

  Arduino IDE:
    Board: ESP32 Dev Module
    Serial Monitor: 115200 baud
*/

#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#define BIG_TOE_PIN     32
#define BALL_FOOT_PIN   34
#define OUTER_FOOT_PIN  35
#define HEEL_PIN        33

#define DEVICE_NAME "Paddrishti - ESP32"

// Fixed UUIDs: keep these the same in ble_receiver.py
#define SERVICE_UUID        "7f6d0001-5a4e-4c3a-9b2e-123456789001"
#define CHARACTERISTIC_UUID "7f6d0002-5a4e-4c3a-9b2e-123456789002"

BLECharacteristic *sensorCharacteristic;
bool deviceConnected = false;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override {
    deviceConnected = true;
    Serial.println("BLE client connected");
  }

  void onDisconnect(BLEServer *server) override {
    deviceConnected = false;
    Serial.println("BLE client disconnected");
    delay(100);
    server->getAdvertising()->start();
  }
};

float syntheticTemperature() {
  // 36.4 - 36.8 °C
  return 36.4f + (random(0, 41) / 100.0f);
}

int syntheticSpO2() {
  // 97 - 99 %
  return random(97, 100);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  analogReadResolution(12); // 0-4095
  randomSeed(analogRead(34) + micros());

  BLEDevice::init(DEVICE_NAME);

  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *service = server->createService(SERVICE_UUID);

  sensorCharacteristic = service->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ |
    BLECharacteristic::PROPERTY_NOTIFY
  );

  sensorCharacteristic->addDescriptor(new BLE2902());

  service->start();

  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  advertising->setMinPreferred(0x06);
  advertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.println();
  Serial.println("Paddrishti ESP32 started");
  Serial.println("Waiting for BLE connection...");
  Serial.println("--------------------------------");
}

void loop() {
  int bigToe    = analogRead(BIG_TOE_PIN);
  int ballFoot  = analogRead(BALL_FOOT_PIN);
  int outerFoot = analogRead(OUTER_FOOT_PIN);
  int heel      = analogRead(HEEL_PIN);

  float temperature = syntheticTemperature();
  int spo2 = syntheticSpO2();

  // JSON sent to the Python receiver / website backend.
  String json = "{";
  json += "\"bigToe\":" + String(bigToe);
  json += ",\"ballFoot\":" + String(ballFoot);
  json += ",\"outerFoot\":" + String(outerFoot);
  json += ",\"heel\":" + String(heel);
  json += ",\"temperature\":" + String(temperature, 2);
  json += ",\"spo2\":" + String(spo2);
  json += ",\"timestamp\":" + String(millis());
  json += "}";

  Serial.println(json);

  if (deviceConnected) {
    sensorCharacteristic->setValue(json.c_str());
    sensorCharacteristic->notify();
  }

  delay(500);
}
