void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);  // Built-in LED (can also use D2, D3 etc.)
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input == "Drone") {
      digitalWrite(LED_BUILTIN, HIGH); // Turn on LED
    } else if (input == "Bird") {
      digitalWrite(LED_BUILTIN, LOW);  // Turn off LED
    }
  }
}
