# IoT Weather Station with AI Forecasting

## Overview
This project sets up an IoT based weather monitoring system using an ESP32 to read environmental data and a Python Flask server to run a machine learning model that predicts short term rain. Sensor readings and prediction values are displayed on a ThingSpeak dashboard in real time.

## Features
- Live data collection from multiple sensors
- Logistic Regression model for rain prediction
- Local Flask server for processing
- Cloud visualization using ThingSpeak
- Timing strategy to avoid network rate limit issues

## System Architecture
**Edge Device (ESP32)**  
Reads sensors and sends two POST requests, one to a local Flask endpoint and one to ThingSpeak.

**Local Server (Python + Flask)**  
Receives data, performs prediction with a trained model, and forwards the result to ThingSpeak.

**Cloud Dashboard (ThingSpeak)**  
Shows all sensor fields and prediction values.

## Hardware
- ESP32
- DHT11 (Temperature and Humidity)
- BMP180 (Pressure)
- Soil Moisture / Water Sensor
- Wi Fi router

## Software
- Arduino IDE (C++)
- Python
- Flask
- Scikit Learn
- Joblib
- ThingSpeak

## AI Model
- Model type: Logistic Regression
- Output: 1 = Rain, 0 = No Rain
- Features: Temperature, Humidity, Pressure, Water Sensor value
- Prediction based on calculated rain probability

## Networking
**Protocols:** HTTP and HTTPS  
**Local API:** `http://192.168.0.x:5000/data`  
**Cloud API:** ThingSpeak update endpoint  

### Rate Limiting Strategy
ThingSpeak allows one update every 15 seconds. To avoid errors:

- ESP32 raw sensor update every 60 seconds
- ESP32 to Flask data send every 30 seconds
- Flask prediction update 10 seconds after receiving data

This keeps at least 20 seconds between any two POST requests.

### TCP/IP
- TCP ensures reliable ordered delivery
- Private IP routing for ESP32 to Flask
- Public internet routing for ThingSpeak updates

## Status
The project handles data acquisition, AI prediction, and cloud logging with secure communication and controlled timing.
