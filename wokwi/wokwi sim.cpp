#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_BMP085.h>

Adafruit_MPU6050 mpu;
Adafruit_BMP085 bmp;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  mpu.begin();
  bmp.begin();
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  Serial.printf("MPU Temp: %.2f °C\n", temp.temperature);
  Serial.printf("Accel Y: %.2f m/s²  Z: %.2f m/s²\n", a.acceleration.y, a.acceleration.z);
  Serial.printf("BMP Temp: %.2f °C\n", bmp.readTemperature());
  Serial.println("---");

  delay(1000);
}