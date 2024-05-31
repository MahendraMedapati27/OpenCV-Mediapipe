int ledPin = 9; // Pin connected to the LED
int brightness = 0; // Variable to store the brightness value

void setup() {
  Serial.begin(9600); // Initialize serial communication at 9600 bits per second
  pinMode(ledPin, OUTPUT); // Set the LED pin as an output
}

void loop() {
  if (Serial.available() > 0) {
    // Read the incoming serial data
    brightness = Serial.parseInt();

    // Ensure the brightness is within the valid range
    brightness = constrain(brightness, 0, 255);

    // Set the brightness of the LED
    analogWrite(ledPin, brightness);
  }
}
