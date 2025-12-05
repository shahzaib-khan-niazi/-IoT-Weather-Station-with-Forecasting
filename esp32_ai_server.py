from flask import Flask, request, jsonify
import requests
import numpy as np
import joblib
import threading
import time

app = Flask(__name__)

# =========================================================
# --- GLOBAL STATE ---
# =========================================================
latest_prediction = 0  # Stores latest AI prediction

# =========================================================
# --- ThingSpeak Configuration ---
# =========================================================
THINGSPEAK_WRITE_API_KEY = "9N60V6HWWU57PLSA"
THINGSPEAK_CHANNEL_ID = "3175714"  # NOTE: The Python update targets fields 1 and 5
THINGSPEAK_UPDATE_URL = "https://api.thingspeak.com/update"
# Increased delay to 45 seconds to avoid conflicts with ESP32's 60-second cycle
THINGSPEAK_PREDICTION_DELAY = 45

# =========================================================
# --- Flask Server Host/Port ---
# =========================================================
FLASK_LISTEN_HOST = '0.0.0.0'
FLASK_LISTEN_PORT = 5000

# =========================================================
# --- Load AI Model ---
# =========================================================
MODEL_FILENAME = 'trained_rain_prediction_model.pkl'
trained_ai_model = None
model_features_order = ['temp', 'humidity', 'pressure', 'water']  # Corrected feature names to match form data

try:
    trained_ai_model = joblib.load(MODEL_FILENAME)
    print(f"✅ Successfully loaded AI model: {MODEL_FILENAME}")
except Exception as e:
    print(f"❌ ERROR loading AI model: {e}. Predictions will use fallback 0. Ensure '{MODEL_FILENAME}' exists.")


# =========================================================
# --- Helper: Send to ThingSpeak ---
# =========================================================
def send_prediction_to_thingspeak(prediction):
    # Field 1 is included to ensure the update is logged, even if the ESP32 is using it for temp.
    # The ESP32 is the primary source for Fields 1-4. Flask only targets Field 5.
    payload = {
        'api_key': THINGSPEAK_WRITE_API_KEY,
        'field1': 1,  # Dummy value to log entry
        'field5': int(prediction)  # AI prediction
    }

    print(f"[ThingSpeak Thread] Waiting {THINGSPEAK_PREDICTION_DELAY}s to avoid rate limits...")
    time.sleep(THINGSPEAK_PREDICTION_DELAY)

    try:
        print(f"[ThingSpeak Thread] Attempting POST with payload: {payload}")
        response = requests.post(THINGSPEAK_UPDATE_URL, data=payload, timeout=15)

        if response.status_code == 200:
            print(f"✅ [ThingSpeak Thread] Prediction {payload['field5']} sent. ThingSpeak Entry ID: {response.text}")
        elif response.status_code == 429:
            print(
                f"❌ [ThingSpeak Thread] Failed (429 Rate Limit). Too many requests. Check ESP32 and Flask update intervals.")
        else:
            print(f"❌ [ThingSpeak Thread] Failed to send. Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"❌ [ThingSpeak Thread] Error sending prediction: {e}")


# =========================================================
# --- Flask Routes ---
# =========================================================
@app.route('/data', methods=['POST'])
def receive_data():
    global latest_prediction
    try:
        # Data keys match the ESP32 code's POST request: temp, humidity, pressure, water
        temp = float(request.form.get('temp', 0.0))
        humidity = float(request.form.get('humidity', 0.0))
        pressure = float(request.form.get('pressure', 0.0))
        water = int(request.form.get('water', 0))
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Invalid data received: {str(e)}"}), 400

    print(f"\nReceived data from ESP32: Temp={temp}°C, Hum={humidity}%, Pres={pressure}, Water={water}")

    # Prepare AI input (features must be in the order expected by the model)
    prediction = 0
    if trained_ai_model:
        try:
            # Note: The model_features_order is: temp, humidity, pressure, water
            X_input = np.array([[temp, humidity, pressure, water]])
            prediction = int(trained_ai_model.predict(X_input)[0])
        except Exception as e:
            print(f"❌ Prediction error: {e}, using fallback 0")

    latest_prediction = prediction
    print(f"Updated latest_prediction: {latest_prediction}")

    # Send prediction to ThingSpeak asynchronously
    # The delay inside this function will prevent the main thread from blocking.
    thread = threading.Thread(target=send_prediction_to_thingspeak, args=(prediction,))
    thread.daemon = True
    thread.start()

    return jsonify({
        "status": "success",
        "message": "Data received and prediction processed.",
        "prediction_value": prediction
    }), 200


@app.route('/control', methods=['GET'])
def get_control_state():
    # This endpoint is provided for the ESP32 to potentially check the prediction,
    # but the current ESP32 code doesn't use it.
    print(f"ESP32 GET /control → latest_prediction={latest_prediction}")
    return jsonify({"prediction": latest_prediction}), 200


# =========================================================
# --- Run Flask Server ---
# =========================================================
if __name__ == '__main__':
    print("---------------------------------------------------------")
    print("Starting Flask server...")
    print(f"ESP32 must send data to this PC's IP on port {FLASK_LISTEN_PORT}")
    print("---------------------------------------------------------")
    app.run(host=FLASK_LISTEN_HOST, port=FLASK_LISTEN_PORT, debug=False)
