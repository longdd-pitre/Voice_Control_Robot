#include <AFMotor.h>
#include <SoftwareSerial.h>

SoftwareSerial bluetooth(2, 3);  // Chân 2 là TX, chân 3 là RX

AF_DCMotor motor1(1);
AF_DCMotor motor2(2);
int defaultSpeed = 255;  // Tốc độ mặc định

void setup() {
  Serial.begin(9600);  
  motor1.setSpeed(0);
  motor2.setSpeed(0);
}

void moveDistance() {
  
  motor1.setSpeed(defaultSpeed);
  motor2.setSpeed(defaultSpeed);
  motor1.run(FORWARD);
  motor2.run(FORWARD);
  delay(200);
  stopMotors();
}

void moveBackward() {
  
  motor1.setSpeed(defaultSpeed);
  motor2.setSpeed(defaultSpeed);
  motor1.run(BACKWARD);
  motor2.run(BACKWARD);
  delay(200);
  stopMotors();
}

void turnLeft() {
  
  motor1.setSpeed(defaultSpeed);
  motor2.setSpeed(defaultSpeed);
  motor1.run(BACKWARD);
  motor2.run(FORWARD);
  delay(200);
  stopMotors();
}

void turnRight() {
  
  motor1.setSpeed(defaultSpeed);
  motor2.setSpeed(defaultSpeed);
  motor1.run(FORWARD);
  motor2.run(BACKWARD);
  delay(200);
  stopMotors();
}

void stopMotors() {
  motor1.setSpeed(0);
  motor2.setSpeed(0);
  motor1.run(RELEASE);
  motor2.run(RELEASE);
}

void executeCommand(char cmd) {
  switch (cmd) {
    case 'F':
      moveDistance();
      break;
    case 'B':
      moveBackward();
      break;
    case 'L':
      turnLeft();
      break;
    case 'R':
      turnRight();
      break;
    case 'S':
      stopMotors();
      break;
    default:
      // Xử lý tín hiệu không xác định tùy ý
      break;
  }
}

void loop() {
  if (bluetooth.available() > 0) {
    char command = bluetooth.read();
    executeCommand(command);
  }
}
