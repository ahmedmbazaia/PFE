// =============================================================
//  Telescope 1 — ESP32-CAM 
//  Sensors: MPU9250, BPW34, VL53L0X, SX1278 LoRa, WiFi/NTP
// =============================================================

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <NTPClient.h>
#include <LoRa.h>
#include <MPU9250.h>
#include <VL53L0X.h>
#include <ArduinoJson.h>

// ─── Feature flags ──────────────────────────────────────────
#define MOTOR_ENABLED   false
#define FSO_ENABLED     false
#define CAMERA_ENABLED  false

// ─── WiFi credentials (NTP only) ────────────────────────────
const char* WIFI_SSID     = "YOUR_SSID";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";

// ─── Pin definitions ────────────────────────────────────────

// I2C bus (MPU9250 + VL53L0X share this bus)
#define I2C_SDA  14
#define I2C_SCL  15

// DRV8825 stepper driver
#define STEPPER_STEP_PIN   12
#define STEPPER_DIR_PIN    13
#define STEPPER_ENABLE_PIN  2

// BPW34 photodiode (analog input)
#define BPW34_PIN  33   // ADC1_CH5 — safe to use alongside WiFi

// SX1278 LoRa (SPI)
#define LORA_SCK   18
#define LORA_MISO  19
#define LORA_MOSI  23
#define LORA_NSS    5
#define LORA_RST   16
#define LORA_DIO0   4

// LoRa radio parameters
#define LORA_FREQUENCY  433E6   // 433 MHz band
#define LORA_TX_POWER   17      // dBm

// ─── Loop interval ──────────────────────────────────────────
#define LOOP_INTERVAL_MS  2000

// ─── Global objects ─────────────────────────────────────────
MPU9250 imu(Wire, 0x68);
VL53L0X tof;
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org", 0, 60000);

// Sensor-ready flags
bool imuReady  = false;
bool tofReady  = false;
bool loraReady = false;

// ─── Forward declarations ───────────────────────────────────
void setupWiFi();
void setupIMU();
void setupTOF();
void setupLoRa();
void readIMU(float &pitch, float &roll, float &yaw);
int  readBPW34();
void readTOF(int &skyDist, int &baselineDist);
void sendLoRa(const String &json);

// =============================================================
//  SETUP
// =============================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n===== Telescope 1 — Boot =====");

    // I2C bus
    Wire.begin(I2C_SDA, I2C_SCL);

    // Stepper pins (configured even if motor disabled)
    pinMode(STEPPER_STEP_PIN,   OUTPUT);
    pinMode(STEPPER_DIR_PIN,    OUTPUT);
    pinMode(STEPPER_ENABLE_PIN, OUTPUT);
    digitalWrite(STEPPER_ENABLE_PIN, HIGH);  // HIGH = disabled on DRV8825

    // Photodiode analog input
    analogReadResolution(12);  // 0–4095
    pinMode(BPW34_PIN, INPUT);

    // Subsystem init
    setupWiFi();
    setupIMU();
    setupTOF();
    setupLoRa();

    Serial.println("===== Setup complete =====\n");
}

// =============================================================
//  LOOP — runs every 2 seconds
// =============================================================
void loop() {
    static unsigned long lastRun = 0;
    unsigned long now = millis();
    if (now - lastRun < LOOP_INTERVAL_MS) return;
    lastRun = now;

    // 1. Read IMU
    float pitch = 0, roll = 0, yaw = 0;
    readIMU(pitch, roll, yaw);

    // 2. Read photodiode
    int lightIntensity = readBPW34();

    // 3. Read VL53L0X (two scheduled readings)
    int skyDist = 0, baselineDist = 0;
    readTOF(skyDist, baselineDist);

    // 4. Get NTP timestamp
    timeClient.update();
    String timestamp = timeClient.getFormattedTime();

    // 5. Build JSON payload
    JsonDocument doc;
    doc["node"]                = "T1";
    doc["timestamp"]           = timestamp;
    doc["pitch"]               = pitch;
    doc["roll"]                = roll;
    doc["yaw"]                 = yaw;
    doc["light_intensity"]     = lightIntensity;
    doc["sky_distance_mm"]     = skyDist;
    doc["baseline_distance_mm"]= baselineDist;
    doc["motor_angle"]         = 0;

    String json;
    serializeJson(doc, json);

    Serial.println(json);

    // 6. Send via LoRa
    sendLoRa(json);
}

// =============================================================
//  WiFi — NTP time sync only
// =============================================================
void setupWiFi() {
    Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    unsigned long start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < 10000) {
        delay(500);
        Serial.print(".");
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf(" connected (%s)\n", WiFi.localIP().toString().c_str());
        timeClient.begin();
        timeClient.update();
    } else {
        Serial.println(" FAILED — NTP unavailable");
    }
}

// =============================================================
//  MPU9250 — pitch, roll, yaw
// =============================================================
void setupIMU() {
    Serial.print("[IMU] Initializing MPU9250... ");
    int status = imu.begin();
    if (status < 0) {
        Serial.printf("FAILED (error %d)\n", status);
        imuReady = false;
    } else {
        Serial.println("OK");
        imuReady = true;
    }
}

void readIMU(float &pitch, float &roll, float &yaw) {
    if (!imuReady) {
        pitch = roll = yaw = 0;
        return;
    }

    imu.readSensor();

    float ax = imu.getAccelX_mss();
    float ay = imu.getAccelY_mss();
    float az = imu.getAccelZ_mss();

    // Pitch and roll from accelerometer
    pitch = atan2(ay, sqrt(ax * ax + az * az)) * 180.0 / PI;
    roll  = atan2(-ax, az) * 180.0 / PI;

    // Yaw from magnetometer (tilt-uncompensated heading)
    float mx = imu.getMagX_uT();
    float my = imu.getMagY_uT();
    yaw = atan2(my, mx) * 180.0 / PI;
    if (yaw < 0) yaw += 360.0;
}

// =============================================================
//  BPW34 — light intensity (analog 0–4095)
// =============================================================
int readBPW34() {
    int value = analogRead(BPW34_PIN);
    return value;
}

// =============================================================
//  VL53L0X — two distance readings (sky + baseline to T2)
//  Single sensor, two sequential measurements
// =============================================================
void setupTOF() {
    Serial.print("[TOF] Initializing VL53L0X... ");
    tof.setTimeout(500);
    if (!tof.init()) {
        Serial.println("FAILED");
        tofReady = false;
    } else {
        tof.setMeasurementTimingBudget(200000);  // 200 ms — higher accuracy
        Serial.println("OK");
        tofReady = true;
    }
}

void readTOF(int &skyDist, int &baselineDist) {
    if (!tofReady) {
        skyDist = baselineDist = 0;
        return;
    }

    // Reading 1: distance to sky (LED matrix above)
    skyDist = tof.readRangeSingleMillimeters();
    if (tof.timeoutOccurred()) {
        Serial.println("[TOF] Sky reading timeout");
        skyDist = 0;
    }

    // Small delay between readings for sensor settling
    delay(50);

    // Reading 2: distance to T2 (baseline measurement)
    baselineDist = tof.readRangeSingleMillimeters();
    if (tof.timeoutOccurred()) {
        Serial.println("[TOF] Baseline reading timeout");
        baselineDist = 0;
    }
}

// =============================================================
//  SX1278 LoRa — transmit JSON packets
// =============================================================
void setupLoRa() {
    Serial.print("[LoRa] Initializing SX1278... ");

    SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_NSS);
    LoRa.setPins(LORA_NSS, LORA_RST, LORA_DIO0);

    if (!LoRa.begin(LORA_FREQUENCY)) {
        Serial.println("FAILED");
        loraReady = false;
    } else {
        LoRa.setTxPower(LORA_TX_POWER);
        Serial.println("OK");
        loraReady = true;
    }
}

void sendLoRa(const String &json) {
    if (!loraReady) {
        Serial.println("[LoRa] Not available — skipping send");
        return;
    }

    LoRa.beginPacket();
    LoRa.print(json);
    LoRa.endPacket();

    Serial.println("[LoRa] Packet sent");
}
