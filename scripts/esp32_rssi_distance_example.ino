// ESP32 RFID distance estimation example
// This is a rough estimate using RSSI. For real RFID readers,
// the exact RSSI field depends on the reader module and library.

float estimateDistance(float rssi, float rssiAtOneMeter = -55.0, float pathLossExponent = 2.2) {
  // Log-distance path loss model
  // distance (m) = 10 ^ ((RSSI_1m - RSSI) / (10 * n))
  return pow(10.0, (rssiAtOneMeter - rssi) / (10.0 * pathLossExponent));
}

void setup() {
  Serial.begin(115200);
}

void loop() {
  // Example RSSI values from a reader module (dBm)
  float rssi = -70.0;
  float distance = estimateDistance(rssi);

  Serial.print("RSSI: ");
  Serial.print(rssi);
  Serial.print(" dBm | Estimated Distance: ");
  Serial.print(distance, 2);
  Serial.println(" m");

  delay(2000);
}
