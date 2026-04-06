// =============================================================
//  Telescope 1 — NodeMCU (ESP8266)
//  Sensors: MPU9250 (I2C SDA=D2 SCL=D1), VL53L0X (I2C), BPW34 (A0)
//  Telemetry: WiFi HTTP POST every 2 s — no FSO, no GPS
// =============================================================

#include <Arduino.h>
#include <Wire.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <MPU9250.h>
#include <VL53L0X.h>
#include <ArduinoJson.h>

// ─── WiFi + server ──────────────────────────────────────────
#define WIFI_SSID      "TT_F590"
#define WIFI_PASSWORD  "upi492l8ok"
#define SERVER_URL     "http://192.168.1.18:5000/telemetry"

// ─── Pin definitions ────────────────────────────────────────

// I2C bus — ESP8266 NodeMCU default I2C pins
// D2 = GPIO4 (SDA), D1 = GPIO5 (SCL)
#define I2C_SDA  D2
#define I2C_SCL  D1

// BPW34 photodiode — ESP8266 has a single 10-bit ADC on A0 (0–1023)
#define BPW34_PIN  A0

// ─── Loop interval ──────────────────────────────────────────
#define LOOP_INTERVAL_MS  2000

// ─── Global objects ─────────────────────────────────────────
MPU9250 imu(Wire, 0x68);
VL53L0X tof;
WiFiClient wifiClient;

bool imuReady  = false;
bool tofReady  = false;
bool wifiReady = false;

// ─── Forward declarations ───────────────────────────────────
void setupWiFi();
void setupIMU();
void setupTOF();
void httpPost(const String &json);
void readIMU(float &pitch, float &roll, float &yaw);
int  readBPW34();
void readTOF(int &skyDist, int &baselineDist);

// =============================================================
//  SETUP
// =============================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n===== Telescope 1 — Boot =====");

    // I2C bus
    Wire.begin(I2C_SDA, I2C_SCL);

    // WiFi scan before connecting
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    delay(100);
    int n = WiFi.scanNetworks();
    Serial.println("[WiFi] Networks found:");
    for (int i = 0; i < n; i++) {
        Serial.print("  ");
        Serial.print(WiFi.SSID(i));
        Serial.print(" (");
        Serial.print(WiFi.RSSI(i));
        Serial.println(" dBm)");
    }

    // Subsystem init
    setupWiFi();
    setupIMU();
    setupTOF();

    Serial.println("===== Setup complete =====\n");
}

// =============================================================
//  LOOP — runs every 2 seconds
// =============================================================
void loop() {
    static unsigned long lastRun = 0;
    static uint32_t packetCounter = 0;
    unsigned long now = millis();

    if (now - lastRun < LOOP_INTERVAL_MS) return;
    lastRun = now;
    packetCounter++;

    // 1. Read IMU
    float pitch = 0, roll = 0, yaw = 0;
    readIMU(pitch, roll, yaw);

    // 2. Read photodiode
    int lightIntensity = readBPW34();

    // 3. Read VL53L0X (sky + baseline)
    int skyDist = 0, baselineDist = 0;
    readTOF(skyDist, baselineDist);

    // 4. Build JSON payload
    JsonDocument doc;
    doc["node"]                  = "T1";
    doc["counter"]               = packetCounter;
    doc["timestamp"]             = millis();
    doc["pitch"]                 = pitch;
    doc["roll"]                  = roll;
    doc["yaw"]                   = yaw;
    doc["light_intensity"]       = lightIntensity;
    doc["sky_distance_mm"]       = skyDist;
    doc["baseline_distance_mm"]  = baselineDist;
    doc["motor_angle"]           = 0;

    String json;
    serializeJson(doc, json);
    Serial.println(json);

    // 5. HTTP POST to central station
    httpPost(json);
}

// =============================================================
//  WiFi — connect and maintain
// =============================================================
void setupWiFi() {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("[WiFi] Connecting to ");
    Serial.print(WIFI_SSID);
    int retries = 0;
    while (WiFi.status() != WL_CONNECTED && retries < 30) {
        delay(500);
        Serial.print(".");
        retries++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        wifiReady = true;
        Serial.println(" OK");
        Serial.print("[WiFi] IP: ");
        Serial.println(WiFi.localIP());
    } else {
        wifiReady = false;
        Serial.println(" FAILED");
    }
}

void httpPost(const String &json) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[HTTP] WiFi not connected — reconnecting...");
        WiFi.reconnect();
        delay(1000);
        if (WiFi.status() != WL_CONNECTED) return;
        wifiReady = true;
    }

    HTTPClient http;
    http.begin(wifiClient, SERVER_URL);
    http.setTimeout(15000);
    http.addHeader("Content-Type", "application/json");

    int code = http.POST(json);
    if (code > 0) {
        Serial.print("[HTTP] POST ");
        Serial.print(code);
        Serial.print(" (");
        Serial.print(json.length());
        Serial.println(" bytes)");
    } else {
        Serial.print("[HTTP] POST failed: ");
        Serial.println(http.errorToString(code));
    }
    http.end();
}

// =============================================================
//  MPU9250 — pitch, roll, yaw
// =============================================================
void setupIMU() {
    Serial.print("[IMU] Initializing MPU9250... ");
    int status = imu.begin();
    if (status < 0) {
        Serial.print("FAILED (error ");
        Serial.print(status);
        Serial.println(")");
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
//  BPW34 — light intensity (analog 0–1023, 10-bit ADC)
// =============================================================
int readBPW34() {
    return analogRead(BPW34_PIN);
}

// =============================================================
//  VL53L0X — sky distance + baseline to T2
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

    // Reading 1: distance to sky
    skyDist = tof.readRangeSingleMillimeters();
    if (tof.timeoutOccurred()) {
        Serial.println("[TOF] Sky reading timeout");
        skyDist = 0;
    }

    delay(50);  // settling time between readings

    // Reading 2: baseline distance to T2
    baselineDist = tof.readRangeSingleMillimeters();
    if (tof.timeoutOccurred()) {
        Serial.println("[TOF] Baseline reading timeout");
        baselineDist = 0;
    }
}
