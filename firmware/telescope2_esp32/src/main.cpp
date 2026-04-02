// =============================================================
//  Telescope 2 — ESP32 (TTGO LoRa32 V1)
//  Sensors: MPU9250, BPW34, VL53L0X, GPS, FSO laser TX
//  Telemetry: WiFi HTTP POST + FSO
// =============================================================

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <MPU9250.h>
#include <VL53L0X.h>
#include <TinyGPSPlus.h>
#include <ArduinoJson.h>

// ─── Feature flags ──────────────────────────────────────────
#define MOTOR_ENABLED  false
#define FSO_ENABLED    true
#define GPS_ENABLED    true
#define WIFI_ENABLED   true

// ─── WiFi + server ──────────────────────────────────────────
#define WIFI_SSID      "TT_F590"
#define WIFI_PASSWORD  "upi492l8ok"
#define SERVER_URL     "http://192.168.1.18:5000/telemetry"

// ─── Pin definitions ────────────────────────────────────────

// I2C bus (MPU9250 + VL53L0X share this bus)
#define I2C_SDA  21
#define I2C_SCL  22

// DRV8825 stepper driver
#define STEPPER_STEP_PIN   25
#define STEPPER_DIR_PIN    26
#define STEPPER_ENABLE_PIN 27

// BPW34 photodiode (analog input)
#define BPW34_PIN  34   // ADC1_CH6 — input-only pin, safe with WiFi

// GPS module (UART2)
#define GPS_RX_PIN  16
#define GPS_TX_PIN  17
#define GPS_BAUD    9600

// FSO laser transmitter (OOK modulation)
#define FSO_LASER_PIN  4
#define FSO_BITRATE     1000  // 1 kbps OOK

// ─── Loop interval ──────────────────────────────────────────
#define LOOP_INTERVAL_MS  2000

// ─── Global objects ─────────────────────────────────────────
MPU9250 imu(Wire, 0x68);
VL53L0X tof;
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);  // UART2

// Sensor-ready flags
bool imuReady  = false;
bool tofReady  = false;
bool gpsReady  = false;
bool wifiReady = false;

// ─── Forward declarations ───────────────────────────────────
void setupIMU();
void setupTOF();
void setupGPS();
void setupFSO();
void setupWiFi();
void httpPost(const String &json);
void readIMU(float &pitch, float &roll, float &yaw);
int  readBPW34();
void readTOF(int &skyDist, int &baselineDist);
void readGPS(double &lat, double &lon, double &altitude);
void fsoTransmit(const String &data);
void fsoSendByte(uint8_t b);

// =============================================================
//  SETUP
// =============================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n===== Telescope 2 — Boot =====");

    // I2C bus
    Wire.begin(I2C_SDA, I2C_SCL);

    #if MOTOR_ENABLED
    // Only configure stepper pins when the motor is actually enabled.
    // On the TTGO LoRa32 board, some of these GPIOs overlap with LoRa pins.
    pinMode(STEPPER_STEP_PIN, OUTPUT);
    pinMode(STEPPER_DIR_PIN, OUTPUT);
    pinMode(STEPPER_ENABLE_PIN, OUTPUT);
    digitalWrite(STEPPER_ENABLE_PIN, HIGH);  // HIGH = disabled on DRV8825
    #endif

    // Photodiode analog input
    analogReadResolution(12);  // 0–4095

    // Subsystem init
    setupIMU();
    setupTOF();
    setupGPS();
    setupFSO();
    setupWiFi();

    Serial.println("===== Setup complete =====\n");
}

// =============================================================
//  LOOP — runs every 2 seconds
// =============================================================
void loop() {
    static unsigned long lastRun = 0;
    static uint32_t packetCounter = 0;
    unsigned long now = millis();

    // Feed GPS parser continuously between cycles
    #if GPS_ENABLED
    while (gpsSerial.available()) {
        gps.encode(gpsSerial.read());
    }
    #endif

    if (now - lastRun < LOOP_INTERVAL_MS) return;
    lastRun = now;
    packetCounter++;

    // 1. Read IMU
    float pitch = 0, roll = 0, yaw = 0;
    readIMU(pitch, roll, yaw);

    // 2. Read photodiode
    int lightIntensity = readBPW34();

    // 3. Read VL53L0X (two scheduled readings)
    int skyDist = 0, baselineDist = 0;
    readTOF(skyDist, baselineDist);

    // 4. Read GPS
    double lat = 0, lon = 0, altitude = 0;
    readGPS(lat, lon, altitude);

    // 5. Build JSON payload
    JsonDocument doc;
    doc["node"]                 = "T2";
    doc["counter"]              = packetCounter;
    doc["timestamp"]            = millis();
    doc["pitch"]                = pitch;
    doc["roll"]                 = roll;
    doc["yaw"]                  = yaw;
    doc["light_intensity"]      = lightIntensity;
    doc["sky_distance_mm"]      = skyDist;
    doc["baseline_distance_mm"] = baselineDist;
    doc["motor_angle"]          = 0;
    doc["lat"]                  = lat;
    doc["lon"]                  = lon;
    doc["altitude"]             = altitude;

    String json;
    serializeJson(doc, json);

    Serial.println(json);

    // 6. Transmit via FSO laser (OOK)
    #if FSO_ENABLED
    fsoTransmit(json);
    #endif

    // 7. Send via WiFi HTTP POST
    #if WIFI_ENABLED
    httpPost(json);
    #endif
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
//  VL53L0X — two distance readings (sky + baseline to T1)
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

    delay(50);  // settling time between readings

    // Reading 2: distance to T1 (baseline measurement)
    baselineDist = tof.readRangeSingleMillimeters();
    if (tof.timeoutOccurred()) {
        Serial.println("[TOF] Baseline reading timeout");
        baselineDist = 0;
    }
}

// =============================================================
//  GPS — latitude, longitude, altitude via UART NMEA
// =============================================================
void setupGPS() {
    #if GPS_ENABLED
    Serial.print("[GPS] Initializing UART2... ");
    gpsSerial.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
    gpsReady = true;
    Serial.println("OK (waiting for fix)");
    #else
    Serial.println("[GPS] Disabled");
    gpsReady = false;
    #endif
}

void readGPS(double &lat, double &lon, double &altitude) {
    if (!gpsReady) {
        lat = lon = altitude = 0;
        return;
    }

    // Drain any remaining bytes in the buffer
    while (gpsSerial.available()) {
        gps.encode(gpsSerial.read());
    }

    if (gps.location.isValid()) {
        lat = gps.location.lat();
        lon = gps.location.lng();
    } else {
        Serial.println("[GPS] No fix — location unavailable");
        lat = lon = 0;
    }

    if (gps.altitude.isValid()) {
        altitude = gps.altitude.meters();
    } else {
        altitude = 0;
    }
}

// =============================================================
//  FSO — OOK laser transmitter at 1 kbps
//  Encodes each byte as 8 bits (MSB first) with start/stop bits
//  Bit period = 1000 us (1 kbps)
// =============================================================
#define FSO_BIT_PERIOD_US  (1000000 / FSO_BITRATE)  // 1000 us per bit

void setupFSO() {
    #if FSO_ENABLED
    Serial.print("[FSO] Initializing laser TX on GPIO ");
    Serial.print(FSO_LASER_PIN);
    pinMode(FSO_LASER_PIN, OUTPUT);
    digitalWrite(FSO_LASER_PIN, LOW);  // laser off by default
    Serial.println("... OK");
    #else
    Serial.println("[FSO] Disabled");
    #endif
}

void fsoSendByte(uint8_t b) {
    // Start bit: laser ON
    digitalWrite(FSO_LASER_PIN, HIGH);
    delayMicroseconds(FSO_BIT_PERIOD_US);

    // 8 data bits, MSB first
    for (int i = 7; i >= 0; i--) {
        digitalWrite(FSO_LASER_PIN, (b >> i) & 0x01 ? HIGH : LOW);
        delayMicroseconds(FSO_BIT_PERIOD_US);
    }

    // Stop bit: laser OFF
    digitalWrite(FSO_LASER_PIN, LOW);
    delayMicroseconds(FSO_BIT_PERIOD_US);
}

void fsoTransmit(const String &data) {
    #if FSO_ENABLED
    Serial.printf("[FSO] Transmitting %d bytes at %d bps\n", data.length(), FSO_BITRATE);

    // Preamble: 4 bytes of 0xAA for receiver clock sync
    for (int i = 0; i < 4; i++) {
        fsoSendByte(0xAA);
    }

    // Sync byte to mark start of payload
    fsoSendByte(0x7E);

    // Length (2 bytes, big-endian)
    uint16_t len = data.length();
    fsoSendByte((len >> 8) & 0xFF);
    fsoSendByte(len & 0xFF);

    // Payload
    for (unsigned int i = 0; i < data.length(); i++) {
        fsoSendByte((uint8_t)data[i]);
    }

    // Simple XOR checksum
    uint8_t checksum = 0;
    for (unsigned int i = 0; i < data.length(); i++) {
        checksum ^= (uint8_t)data[i];
    }
    fsoSendByte(checksum);

    Serial.println("[FSO] Transmission complete");
    #endif
}

// =============================================================
//  WiFi — HTTP POST to central station
// =============================================================
void setupWiFi() {
    #if WIFI_ENABLED
    Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int retries = 0;
    while (WiFi.status() != WL_CONNECTED && retries < 20) {
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
    #else
    Serial.println("[WiFi] Disabled");
    wifiReady = false;
    #endif
}

void httpPost(const String &json) {
    #if WIFI_ENABLED
    if (!wifiReady) {
        // Try to reconnect
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[WiFi] Reconnecting...");
            WiFi.reconnect();
            delay(1000);
            if (WiFi.status() != WL_CONNECTED) return;
            wifiReady = true;
        }
    }

    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");

    int code = http.POST(json);
    if (code > 0) {
        Serial.printf("[HTTP] POST %d (%d bytes)\n", code, json.length());
    } else {
        Serial.printf("[HTTP] POST failed: %s\n", http.errorToString(code).c_str());
    }
    http.end();
    #endif
}
