#include <Arduino.h>
#include <Wire.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#include "MAX30100_PulseOximeter.h"


// ============================================================
// DFU SMART INSOLE
// ONE ESP32
//
// FOOT:
//   - Pressure: Big Toe
//   - Pressure: Heel
//   - Pressure: Medial Metatarsal
//   - Pressure: Lateral Metatarsal
//   - LM35 plantar temperature
//
// FINGER:
//   - MAX30100 SpO2
//   - MAX30100 Heart Rate
//
// COMMUNICATION:
//   ESP32 -> BLE -> Laptop -> Antigravity Backend
// ============================================================


// ============================================================
// PRESSURE SENSOR PINS
// ============================================================

#define PRESSURE_BIG_TOE_PIN   34
#define PRESSURE_HEEL_PIN      35
#define PRESSURE_MEDIAL_PIN    32
#define PRESSURE_LATERAL_PIN   33


// ============================================================
// LM35
// ============================================================

#define LM35_PIN 25


// ============================================================
// MAX30100 I2C
// ============================================================

#define MAX30100_SDA 21
#define MAX30100_SCL 22


// ============================================================
// BLE UUIDs
// ============================================================

#define SERVICE_UUID \
"4fafc201-1fb5-459e-8fcc-c5c9c331914b"

#define CHARACTERISTIC_UUID \
"beb5483e-36e1-4688-b7f5-ea07361b26a8"


// ============================================================
// TIMING
// ============================================================

#define BLE_SEND_INTERVAL 1000

#define MAX30100_REPORT_INTERVAL 1000


// ============================================================
// BLE VARIABLES
// ============================================================

BLECharacteristic *sensorCharacteristic;

bool deviceConnected = false;


// ============================================================
// MAX30100
// ============================================================

PulseOximeter pox;

float currentSpO2 = 0.0;

float currentHeartRate = 0.0;

unsigned long lastMAXReport = 0;


// ============================================================
// BLE CALLBACKS
// ============================================================

class ServerCallbacks : public BLEServerCallbacks {

  void onConnect(BLEServer *server) {

    deviceConnected = true;

    Serial.println();
    Serial.println("================================");
    Serial.println("BLE CLIENT CONNECTED");
    Serial.println("================================");
  }


  void onDisconnect(BLEServer *server) {

    deviceConnected = false;

    Serial.println();
    Serial.println("BLE CLIENT DISCONNECTED");

    server->getAdvertising()->start();

    Serial.println(
      "BLE advertising restarted."
    );
  }
};


// ============================================================
// MAX30100 BEAT CALLBACK
// ============================================================

void beatDetected() {

  Serial.println(
    "MAX30100: Heart beat detected"
  );
}


// ============================================================
// READ PRESSURE
// ============================================================

float readPressure(int pin) {

  const int samples = 10;

  long total = 0;

  for (int i = 0; i < samples; i++) {

    total += analogRead(pin);

    delayMicroseconds(200);
  }

  float averageADC =
    total / (float)samples;


  // Normalize ADC to 0-100.
  //
  // This is NOT yet a physical pressure unit.
  // Actual calibration will be done later.

  float pressure =
    (averageADC / 4095.0) * 100.0;


  if (pressure < 0)
    pressure = 0;

  if (pressure > 100)
    pressure = 100;


  return pressure;
}


// ============================================================
// READ LM35 TEMPERATURE
// ============================================================

float readTemperature() {

  const int samples = 20;

  long total = 0;


  for (int i = 0; i < samples; i++) {

    total += analogRead(LM35_PIN);

    delayMicroseconds(200);
  }


  float averageADC =
    total / (float)samples;


  // ESP32 ADC:
  //
  // 12-bit = 0 to 4095
  //
  // Approximate voltage:
  //
  // Voltage = ADC / 4095 × 3.3

  float voltage =
    (averageADC / 4095.0) * 3.3;


  // LM35:
  //
  // 10 mV per °C
  //
  // 0.010 V = 1 °C

  float temperature =
    voltage * 100.0;


  return temperature;
}


// ============================================================
// UPDATE MAX30100
//
// IMPORTANT:
// This function must run continuously.
// ============================================================

void updateMAX30100() {

  pox.update();


  if (
    millis() - lastMAXReport
    >= MAX30100_REPORT_INTERVAL
  ) {

    lastMAXReport = millis();


    currentHeartRate =
      pox.getHeartRate();


    currentSpO2 =
      pox.getSpO2();


    // Reject obviously invalid values.

    if (
      currentHeartRate < 0 ||
      currentHeartRate > 250
    ) {

      currentHeartRate = 0;
    }


    if (
      currentSpO2 < 0 ||
      currentSpO2 > 100
    ) {

      currentSpO2 = 0;
    }


    Serial.print(
      "Heart Rate: "
    );

    Serial.print(
      currentHeartRate,
      1
    );

    Serial.print(
      " BPM | SpO2: "
    );

    Serial.print(
      currentSpO2,
      1
    );

    Serial.println(
      " %"
    );
  }
}


// ============================================================
// CREATE SENSOR JSON
// ============================================================

String createSensorJSON() {

  // -------------------------------
  // FOOT
  // -------------------------------

  float bigToe =
    readPressure(
      PRESSURE_BIG_TOE_PIN
    );


  float heel =
    readPressure(
      PRESSURE_HEEL_PIN
    );


  float medial =
    readPressure(
      PRESSURE_MEDIAL_PIN
    );


  float lateral =
    readPressure(
      PRESSURE_LATERAL_PIN
    );


  float temperature =
    readTemperature();


  // -------------------------------
  // JSON
  // -------------------------------

  String json = "{";


  // Device

  json +=
    "\"device_id\":\"DFU_INSOLE_01\",";


  // Timestamp

  json +=
    "\"timestamp_ms\":";

  json +=
    String(millis());

  json += ",";


  // -------------------------------
  // FOOT DATA
  // -------------------------------

  json +=
    "\"foot\":{";


  json +=
    "\"pressure_big_toe\":";

  json +=
    String(bigToe, 2);

  json += ",";


  json +=
    "\"pressure_heel\":";

  json +=
    String(heel, 2);

  json += ",";


  json +=
    "\"pressure_medial\":";

  json +=
    String(medial, 2);

  json += ",";


  json +=
    "\"pressure_lateral\":";

  json +=
    String(lateral, 2);

  json += ",";


  json +=
    "\"plantar_temperature\":";

  json +=
    String(temperature, 2);


  json += "},";


  // -------------------------------
  // FINGER DATA
  // -------------------------------

  json +=
    "\"finger\":{";


  json +=
    "\"spo2\":";

  json +=
    String(currentSpO2, 2);

  json += ",";


  json +=
    "\"heart_rate\":";

  json +=
    String(currentHeartRate, 2);


  json += "}";


  json += "}";


  return json;
}


// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  delay(1000);


  Serial.println();
  Serial.println(
    "========================================"
  );

  Serial.println(
    "DFU SMART INSOLE"
  );

  Serial.println(
    "ONE ESP32 SENSOR SYSTEM"
  );

  Serial.println(
    "========================================"
  );


  // ==========================================================
  // ADC
  // ==========================================================

  analogReadResolution(12);


  pinMode(
    PRESSURE_BIG_TOE_PIN,
    INPUT
  );

  pinMode(
    PRESSURE_HEEL_PIN,
    INPUT
  );

  pinMode(
    PRESSURE_MEDIAL_PIN,
    INPUT
  );

  pinMode(
    PRESSURE_LATERAL_PIN,
    INPUT
  );

  pinMode(
    LM35_PIN,
    INPUT
  );


  Serial.println(
    "Analog sensors initialized."
  );


  // ==========================================================
  // I2C
  // ==========================================================

  Wire.begin(
    MAX30100_SDA,
    MAX30100_SCL
  );


  Serial.println(
    "Initializing MAX30100..."
  );


  if (!pox.begin()) {

    Serial.println(
      "WARNING: MAX30100 not detected!"
    );

    Serial.println(
      "Check SDA, SCL, VCC and GND."
    );

  } else {

    Serial.println(
      "MAX30100 initialized successfully."
    );


    pox.setIRLedCurrent(
      MAX30100_LED_CURR_7_6MA
    );


    pox.setOnBeatDetectedCallback(
      beatDetected
    );
  }


  // ==========================================================
  // BLE
  // ==========================================================

  Serial.println(
    "Initializing BLE..."
  );


  BLEDevice::init(
    "DFU_SMART_INSOLE"
  );


  BLEServer *server =
    BLEDevice::createServer();


  server->setCallbacks(
    new ServerCallbacks()
  );


  BLEService *service =
    server->createService(
      SERVICE_UUID
    );


  sensorCharacteristic =
    service->createCharacteristic(

      CHARACTERISTIC_UUID,

      BLECharacteristic::PROPERTY_READ |
      BLECharacteristic::PROPERTY_NOTIFY
    );


  sensorCharacteristic->addDescriptor(
    new BLE2902()
  );


  sensorCharacteristic->setValue(
    "{\"status\":\"ready\"}"
  );


  service->start();


  // ==========================================================
  // BLE ADVERTISING
  // ==========================================================

  BLEAdvertising *advertising =
    BLEDevice::getAdvertising();


  advertising->addServiceUUID(
    SERVICE_UUID
  );


  advertising->setScanResponse(
    true
  );


  advertising->setMinPreferred(
    0x06
  );


  advertising->setMinPreferred(
    0x12
  );


  BLEDevice::startAdvertising();


  Serial.println();
  Serial.println(
    "========================================"
  );

  Serial.println(
    "SYSTEM READY"
  );

  Serial.println(
    "BLE NAME: DFU_SMART_INSOLE"
  );

  Serial.println(
    "Waiting for laptop connection..."
  );

  Serial.println(
    "========================================"
  );
}


// ============================================================
// MAIN LOOP
// ============================================================

void loop() {

  // ==========================================================
  // MAX30100 MUST BE SERVICED CONTINUOUSLY
  // ==========================================================

  updateMAX30100();


  // ==========================================================
  // BLE TRANSMISSION
  // ==========================================================

  static unsigned long lastBLESend = 0;


  if (
    deviceConnected &&
    millis() - lastBLESend >= BLE_SEND_INTERVAL
  ) {

    lastBLESend = millis();


    String sensorData =
      createSensorJSON();


    Serial.println();
    Serial.println(
      "----------------------------------------"
    );

    Serial.println(
      "SENSOR DATA"
    );

    Serial.println(
      sensorData
    );


    sensorCharacteristic->setValue(
      sensorData.c_str()
    );


    sensorCharacteristic->notify();
  }


  // Keep loop fast so MAX30100 is serviced.
  delay(5);
}