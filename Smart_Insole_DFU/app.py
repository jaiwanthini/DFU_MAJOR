from flask import Flask, render_template

app = Flask(__name__)

def get_risk_badge(risk_score):
    if risk_score < 30:
        return 'LOW RISK', 'low'
    if risk_score < 60:
        return 'MEDIUM RISK', 'medium'
    return 'HIGH RISK', 'high'


def get_temperature_status(delta_t):
    if delta_t < 1.0:
        return 'Normal', 'low'
    if delta_t <= 2.2:
        return 'Inflammation Suspected', 'medium'
    return 'High Inflammation', 'high'


def get_spo2_status(spo2):
    if spo2 >= 95:
        return 'Normal', 'low'
    if spo2 >= 90:
        return 'Borderline', 'medium'
    return 'Low', 'high'


@app.route('/')
def dashboard():
    # ── Sensor readings (will come from ESP32/BLE later) ──
    temperature_diff = 1.4
    spo2 = 96
    heart_rate = 78
    risk_score = 72

    # ── Derived statuses ──
    risk_badge, risk_badge_class = get_risk_badge(risk_score)
    temperature_status, temperature_class = get_temperature_status(temperature_diff)
    spo2_status, spo2_class = get_spo2_status(spo2)

    # ── Pressure values per region (0-100 scale) ──
    pressure_values = {
        'heel': 45,
        'medial_forefoot': 62,
        'lateral_forefoot': 49,
        'toe': 38,
    }

    # ── Sensor connectivity (True = connected) ──
    sensors = {
        'FSR1': True,
        'FSR2': True,
        'FSR3': True,
        'FSR4': True,
        'LM35': True,
        'MAX30102': True,
    }

    # ── Prediction history (dummy; will be DB-driven later) ──
    prediction_history = [
        {'timestamp': '2026-07-15 10:00', 'risk_score': 72, 'status': 'HIGH RISK', 'status_class': 'high'},
        {'timestamp': '2026-07-15 09:50', 'risk_score': 65, 'status': 'HIGH RISK', 'status_class': 'high'},
        {'timestamp': '2026-07-15 09:40', 'risk_score': 58, 'status': 'MEDIUM RISK', 'status_class': 'medium'},
        {'timestamp': '2026-07-15 09:30', 'risk_score': 52, 'status': 'MEDIUM RISK', 'status_class': 'medium'},
        {'timestamp': '2026-07-15 09:20', 'risk_score': 44, 'status': 'MEDIUM RISK', 'status_class': 'medium'},
        {'timestamp': '2026-07-15 09:10', 'risk_score': 38, 'status': 'MEDIUM RISK', 'status_class': 'medium'},
        {'timestamp': '2026-07-15 09:00', 'risk_score': 25, 'status': 'LOW RISK', 'status_class': 'low'},
        {'timestamp': '2026-07-15 08:50', 'risk_score': 18, 'status': 'LOW RISK', 'status_class': 'low'},
    ]

    context = {
        # Patient / session
        'patient_id': 'DFU-2026-0042',
        'session_id': 'S-00187',
        'age': 58,
        'gender': 'Male',
        'walk_duration': '12 min 34 sec',

        # Risk
        'risk_score': risk_score,
        'risk_message': 'Monitor closely and review foot pressure distribution daily.',
        'risk_badge': risk_badge,
        'risk_badge_class': risk_badge_class,

        # Vitals
        'spo2': spo2,
        'spo2_status': spo2_status,
        'spo2_status_class': spo2_class,
        'heart_rate': heart_rate,
        'temperature_diff': temperature_diff,
        'temperature_status': temperature_status,
        'temperature_status_class': temperature_class,

        # Pressure
        'epti': 4120,
        'pressure_values': pressure_values,
        'regional_ptis': {
            'Heel PTI': 3150,
            'Mid PTI': 2890,
            'Fore PTI': 3010,
            'Toe PTI': 2450,
        },

        # Device
        'ble_status': 'Connected',
        'battery': 82,
        'sensors': sensors,

        # Timestamps
        'last_update': '2026-07-15 10:00',
        'last_prediction': '2026-07-15 10:00:12',

        # History
        'prediction_history': prediction_history,

        # System
        'status': 'Stable',
    }
    return render_template('index.html', **context)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
