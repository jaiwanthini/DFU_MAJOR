"""
Paddrishti - FSR Calibration
----------------------------
Converts the ESP32's raw 12-bit ADC readings (0-4095)
into normalized 0-100 pressure/intensity values.

This is a SOFTWARE calibration layer for development.
For real force/pressure units (N, kgf, kPa), the FSRs must
be calibrated against known loads and a sensor-specific curve.
"""

from typing import Dict


SENSORS = ("bigToe", "ballFoot", "outerFoot", "heel")

# Change these after taking real calibration measurements.
# minimum = no/very little pressure
# maximum = chosen reference pressure for 100%
DEFAULT_CALIBRATION = {
    "bigToe":    {"min": 0, "max": 3000},
    "ballFoot":  {"min": 0, "max": 3000},
    "outerFoot": {"min": 0, "max": 3000},
    "heel":      {"min": 0, "max": 3000},
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def normalize(raw: int, minimum: int, maximum: int) -> float:
    if maximum <= minimum:
        raise ValueError("Calibration maximum must be greater than minimum.")

    percentage = (raw - minimum) / (maximum - minimum) * 100.0
    return round(clamp(percentage, 0.0, 100.0), 1)


def calibrate_fsr(raw_values: Dict[str, int],
                  calibration=None) -> Dict[str, float]:
    """Return each FSR as a 0-100 normalized intensity."""
    calibration = calibration or DEFAULT_CALIBRATION

    result = {}
    for sensor in SENSORS:
        raw = int(raw_values.get(sensor, 0))
        limits = calibration[sensor]
        result[sensor] = normalize(raw, limits["min"], limits["max"])

    return result


def calibrate_packet(packet: Dict) -> Dict:
    """Add a 'pressure' object to an incoming ESP32 packet."""
    raw_values = {sensor: int(packet.get(sensor, 0)) for sensor in SENSORS}

    output = dict(packet)
    output["pressure"] = calibrate_fsr(raw_values)
    return output


if __name__ == "__main__":
    example = {
        "bigToe": 1200,
        "ballFoot": 2100,
        "outerFoot": 700,
        "heel": 1600,
    }

    print("Calibration test:")
    print(calibrate_fsr(example))
