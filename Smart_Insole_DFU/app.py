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
    temperature_diff = 1.4
    spo2 = 96
    risk_score = 72
    risk_badge, risk_badge_class = get_risk_badge(risk_score)
    temperature_status, temperature_class = get_temperature_status(temperature_diff)
    spo2_status, spo2_class = get_spo2_status(spo2)

    context = {
        'patient_id': 'Patient_A',
        'risk_score': risk_score,
        'risk_message': 'Monitor closely and review foot pressure distribution daily.',
        'risk_badge': risk_badge,
        'risk_badge_class': risk_badge_class,
        'temperature_diff': temperature_diff,
        'temperature_status': temperature_status,
        'temperature_status_class': temperature_class,
        'spo2': spo2,
        'spo2_status': spo2_status,
        'spo2_status_class': spo2_class,
        'epti': 4120,
        'regional_ptis': {
            'Heel PTI': 3150,
            'Mid PTI': 2890,
            'Fore PTI': 3010,
            'Toe PTI': 2450,
        },
        'pressure_chart_labels': ['Heel', 'Midfoot', 'Forefoot', 'Toe'],
        'pressure_chart_values': [45, 52, 49, 38],
        'pressure_snapshot_values': [45, 52, 49, 38],
        'pressure_snapshot_labels': ['Heel', 'Midfoot', 'Forefoot', 'Toe'],
        'last_update': '2026-06-26 19:45',
        'status': 'Stable',
    }
    return render_template('index.html', **context)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
