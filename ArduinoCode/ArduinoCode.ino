// Code that is uploaded to the arduino esp32 that is responsible for connecting to the localhost server which is created on node.js
// Created by Mo Kouli
// Date: 3 March 2024

// DOCS were used to create a websocket communication 'https://github.com/Links2004/arduinoWebSockets'


#include <WiFi.h> // Include the WiFi library to connect to the WiFi network
#include <ArduinoWebsockets.h> // Include the ArduinoWebsockets library for WebSocket communication
#include "DHT.h"
#include "time.h"
#include "DFRobot_MICS.h"
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Stepper.h>
#define CALIBRATION_TIME 1 // Default calibration time is three minutes
const char* ssid = "M Kouli"; //  WiFi SSID (network name)
const char* password = "Mohammad";  //  WiFi password
const char* websockets_server = "ws://192.168.43.75:8080"; // WebSocket server URI (address)


using namespace websockets; // Using the websockets namespace to avoid any clashes with `websockets::` function
WebsocketsClient client; // Create a WebsocketsClient object for connecting and communicating with the WebSocket server

// This function is called whenever a WebSocket message is received
void onMessageCallback(WebsocketsMessage message) {
  Serial.print("Got Message: "); // Print a log message to the Serial monitor
  Serial.println(message.data()); // Print the actual message data received to the Serial monitor
  if(message.data() == "toggle") { // Check if the received message is "toggle"
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); // Toggle the onboard LED state
  }
}

// Start of the sesnors and actuators
const int thermistorPin = A0;  // The analog pin where the thermistor is connected
const float referenceResistor = 10000;  // Reference resistor value (ohms)
const float nominalResistance = 10000;  // Resistance at 25 degrees C (ohms)
const float nominalTemperature = 25;    // Nominal temperature (Celsius)
const float bCoefficient = 3950;        // Beta coefficient of the thermistor
#define DHTPIN 2 // The pin where the DHT sensor is connected
#define DHTTYPE DHT11 // Change this to your DHT type if different
DHT dht(DHTPIN, DHTTYPE);
#define SensorPin A0            //pH meter Analog output to Arduino Analog Input 0
#define Offset 0.00            //deviation compensate
#define samplingInterval 20
#define ArrayLenth  40    //times of collection
int pHArray[ArrayLenth];   //Store the average value of the sensor feedback
int pHArrayIndex=0;
#define ADC_PIN A2
#define POWER_PIN 7
DFRobot_MICS_ADC mics(/*adcPin*/ADC_PIN, /*powerPin*/POWER_PIN);
// Define pin numbers for the pumps
const int pumpPin1 = 1;  // Digital pin connected to relay/transistor for Pump 1
const int pumpPin2 = 12;  // Digital pin connected to relay/transistor for Pump 2
const int pumpPin3 = 13;  // Digital pin connected to relay/transistor for Pump 3
const int stepsPerRevolution = 200;  // change this to fit the number of steps per revolution for your motor
// Initialize the stepper library on pins 8 through 11:
Stepper myStepper(stepsPerRevolution, 8, 9, 10, 11);
// Define the pins for the ultrasonic sensor
const int triggerPin = 5;
const int echoPin = 6;
// Speed of sound in cm/us for distance calculation
const float speedOfSound = 0.0343;
// Variable to hold the baseline distance measured at setup
float baselineDistance = 0.0;
LiquidCrystal_I2C lcd(0x27, 16, 2); // Initialize the LCD with I2C
// Joystick setup
const int joystickXPin = A0;  // Analog input pin for the X-axis
const int joystickYPin = A1;  // Analog input pin for the Y-axis (not used)
const int joystickButtonPin = 2;  // Digital pin for the joystick button
// Sensor data variables
float temperature = 0.0;
float humidity = 0.0;
float pHValue = 0.0;
int liquidLevel = 0;
// Array to store sensor data pointers for easy switching
float* sensorValues[3] = {&temperature, &humidity, &pHValue}; // Cast int to float pointer
String sensorNames[3] = {"Temp", "Hum", "pH"};
int sensorIndex = 0;  // Index to keep track of the current sensor being displayed

// Function to calculate temperature from NTC resistance
float calculateTemperature(float resistance) {
  float temp;
  temp = resistance / nominalResistance;     // (R/Ro)
  temp = log(temp);                          // ln(R/Ro)
  temp /= bCoefficient;                      // 1/B * ln(R/Ro)
  temp += 1.0 / (nominalTemperature + 273.15); // + (1/To)
  temp = 1.0 / temp;                         // Invert
  temp -= 273.15;                            // convert to Celsius
  return temp;
}
double avergearray(int* arr, int number){
  int i;
  int max,min;
  double avg;
  long amount=0;
  if(number<=0){
    Serial.println("Error number for the array to avraging!/n");
    return 0;
  }
  if(number<5){   //less than 5, calculated directly statistics
    for(i=0;i<number;i++){
      amount+=arr[i];
    }
    avg = amount/number;
    return avg;
  }else{
    if(arr[0]<arr[1]){
      min = arr[0];max=arr[1];
    }
    else{
      min=arr[1];max=arr[0];
    }
    for(i=2;i<number;i++){
      if(arr[i]<min){
        amount+=min;        //arr<min
        min=arr[i];
      }else {
        if(arr[i]>max){
          amount+=max;    //arr>max
          max=arr[i];
        }else{
          amount+=arr[i]; //min<=arr<=max
        }
      }//if
    }//for
    avg = (double)amount/(number-2);
  }//if
  return avg;
}
// Function to measure distance using the ultrasonic sensor
float getDistance() {
  // Ensure a clean pulse signal
  digitalWrite(triggerPin, LOW);
  delayMicroseconds(2);
  
  // Send out a pulse
  digitalWrite(triggerPin, HIGH);
  delayMicroseconds(10); // Ensuring a proper high signal
  digitalWrite(triggerPin, LOW);
  
  // Measure the duration of the echo to calculate distance
  long duration = pulseIn(echoPin, HIGH,30000);
  float distance = duration * speedOfSound / 2; // Calculate the distance
  return distance; // Return the measured distance
}

void setup() {
  lcd.init(); // Initialize the LCD
  lcd.backlight(); // Turn on the backlight
  // Set the pump pins as outputs
  pinMode(pumpPin1, OUTPUT);
  pinMode(pumpPin2, OUTPUT);
  pinMode(pumpPin3, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT); // Initialize the built-in LED pin as an output
  // set the speed at 60 rpm:
  myStepper.setSpeed(60);
  // Set sensor pins as output and input respectively
  pinMode(triggerPin, OUTPUT);
  pinMode(echoPin, INPUT);

  // Warm up the sensor by taking some preliminary readings
  for(int i = 0; i < 10; i++) {
    getDistance(); // Take a distance measurement
    delay(50);
  }
  
  // Establish a baseline distance for future comparisons
  baselineDistance = getDistance();
  Serial.begin(115200); // Start serial communication at 115200 baud rate
  
  // Connect to Wi-Fi with the provided SSID and password
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { // Wait for the connection to complete
    delay(1000); // Wait a second before retrying
    Serial.println("Connecting to WiFi.."); // Print a log message to the Serial monitor
  }

  // Setup the WebSocket connection
  client.onMessage(onMessageCallback); // Assign the message callback function
  client.connect(websockets_server); // Connect to the WebSocket server using the specified URI
  while(!Serial);
  while(!mics.begin()){
    Serial.println("NO Deivces !");
    delay(1000);
  } Serial.println("Device connected successfully !");

  /**!
    Gets the power mode of the sensor
    The sensor is in sleep mode when power is on,so it needs to wake up the sensor. 
    The data obtained in sleep mode is wrong
   */
  uint8_t mode = mics.getPowerState();
  if(mode == SLEEP_MODE){
    mics.wakeUpMode();
    Serial.println("wake up sensor success!");
  }else{
    Serial.println("The sensor is wake up mode");
  }

  /**!
     Do not touch the sensor probe when preheating the sensor.
     Place the sensor in clean air.
     The default calibration time is 3 minutes.
  */
  while(!mics.warmUpTime(CALIBRATION_TIME)){
    Serial.println("Please wait until the warm-up time is over!");
    delay(1000);
  }

}

void loop() {
  static unsigned long samplingTime = millis();
  static float pHValue,voltage;
  int aVal = analogRead(thermistorPin);  // Read the analog value from sensor
  float resistance = (1023.0 / aVal) - 1;  // Calculate the resistance of the thermistor
  resistance = referenceResistor / resistance;
  float temperature = calculateTemperature(resistance);  // Calculate temperature
  float humidity = dht.readHumidity();
    /**!
    Gas type:
    MICS-4514 You can get all gas concentration
    MICS-5524 You can get the concentration of CH4, C2H5OH, H2, NH3, CO
    MICS-2714 You can get the concentration of NO2
      Methane          (CH4)    (1000 - 25000)PPM
      Ethanol          (C2H5OH) (10   - 500)PPM
      Hydrogen         (H2)     (1    - 1000)PPM
      Ammonia          (NH3)    (1    - 500)PPM
      Carbon Monoxide  (CO)     (1    - 1000)PPM
      Nitrogen Dioxide (NO2)    (0.1  - 10)PPM
  */
  float gas_C2H5OH = mics.getGasData(C2H5OH);
  float gas_CH4 = mics.getGasData(CH4);
  float gas_H2 = mics.getGasData(H2);
  float gas_NH3 = mics.getGasData(NH3);
  float gas_CO = mics.getGasData(CO);
  liquidLevel = digitalRead(4);
  if(millis()-samplingTime > samplingInterval)
  {
      pHArray[pHArrayIndex++]=analogRead(SensorPin);
      if(pHArrayIndex==ArrayLenth)pHArrayIndex=0;
      voltage = avergearray(pHArray, ArrayLenth)*5.0/1024;
      pHValue = 3.5*voltage+Offset;
      samplingTime=millis();
  }
  Serial.println(pHValue,2);
  Serial.println(temperature,2);
  Serial.println(humidity,2);

  //digitalWrite(pumpPin2,HIGH);
  //myStepper.step(stepsPerRevolution);

  // Calculate the current distance and adjust it by subtracting the baseline
  float distance = getDistance() - baselineDistance;

  int xValue = analogRead(joystickXPin);  // Read the X-axis
  int buttonState = digitalRead(joystickButtonPin);  // Read the joystick button state

  // Joystick X-axis for switching sensor display
  if (xValue < 400) {
    sensorIndex--;  
    if (sensorIndex < 0) sensorIndex = 1;  
  } else if (xValue > 600) {
    sensorIndex++;  
    sensorIndex %= 2;  
  }

  // Button for resetting the sensor index
  if (buttonState == LOW) {  // Assuming button is active LOW
    sensorIndex = 0;  // Reset to the first sensor
  }

  // Update display
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(sensorNames[sensorIndex] + ": ");
  lcd.print(*sensorValues[sensorIndex], 2);

  // Create a JSON string with the sensor values
  String jsonData = "{\"humidity\":" + String(humidity, 2) + ", \"temperature\":" + String(temperature, 2) + "}";
  
  client.send(jsonData);
  Serial.println("Sent data: " + jsonData);
  client.poll(); // Keep the WebSocket connection alive and check for new messages
  delay(1000); // Delay between measurements
}
