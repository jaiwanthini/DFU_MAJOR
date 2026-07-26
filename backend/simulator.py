"""
==========================================================
Smart Insole DFU Risk Prediction
Hardware Simulator

Generates synthetic live sensor data mimicking an ESP32.
Provides different operational modes (Normal, Medium, High).
Designed to be swapped out for a true BLE/Serial hardware
client without modifying upstream processing.
==========================================================
"""

import time
import random
from typing import Dict, Any, Tuple

class SensorSimulator:
    """
    Simulates hardware input from the Smart Insole.
    
    Provides a `.get_reading()` interface that can be exactly 
    replicated by future hardware (ESP32) modules.
    """
    
    MODES = ["Normal", "Medium Risk", "High Risk"]
    
    # Define ranges for each mode for clamping and initialization
    RANGES = {
        "Normal": {
            "fsr": (100, 400), "temp": (34.0, 35.5), "hr": (60, 80), "spo2": (97.0, 100.0)
        },
        "Medium Risk": {
            "fsr": (400, 700), "temp": (36.0, 36.9), "hr": (80, 95), "spo2": (94.0, 96.5)
        },
        "High Risk": {
            "fsr": (700, 1000), "temp": (37.2, 38.5), "hr": (100, 125), "spo2": (88.0, 93.0)
        }
    }

    def __init__(self, mode: str = "Normal"):
        """
        Initialize the simulator.
        
        Parameters
        ----------
        mode : str
            The initial simulation mode. Default is "Normal".
        """
        self.mode = mode if mode in self.MODES else "Normal"
        self._init_state()

    def _init_state(self) -> None:
        """Initialize current values based on the active mode."""
        bounds = self.RANGES[self.mode]
        self.current_fsr1 = random.uniform(*bounds["fsr"])
        self.current_fsr2 = random.uniform(*bounds["fsr"])
        self.current_fsr3 = random.uniform(*bounds["fsr"])
        self.current_fsr4 = random.uniform(*bounds["fsr"])
        self.current_temperature = random.uniform(*bounds["temp"])
        self.current_heart_rate = random.uniform(*bounds["hr"])
        self.current_spo2 = random.uniform(*bounds["spo2"])
        
    def _smooth_update(self, current: float, min_val: float, max_val: float, max_delta: float, noise_std: float) -> Tuple[float, float]:
        """
        Updates a value with a random walk, clamps it to bounds, 
        and adds Gaussian noise to simulate sensor inaccuracy.
        Returns (new_current_state, noisy_reading).
        """
        current += random.uniform(-max_delta, max_delta)
        current = max(min_val, min(max_val, current))
        noisy_val = current + random.gauss(0, noise_std)
        return current, noisy_val
        
    def reset(self) -> None:
        """
        Resets the simulator to its default 'Normal' state and reinitializes bounds.
        """
        self.mode = "Normal"
        self._init_state()

    def set_mode(self, mode: str) -> None:
        """
        Change the simulation mode on the fly.
        
        Parameters
        ----------
        mode : str
            Must be one of: "Normal", "Medium Risk", "High Risk"
        """
        if mode in self.MODES:
            if self.mode != mode:
                self.mode = mode
                self._init_state()
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected one of {self.MODES}")

    def get_reading(self) -> Dict[str, Any]:
        """
        Generates one frame of simulated sensor data based on the current mode.
        Returns a dictionary mimicking a JSON payload from the ESP32.
        
        Returns
        -------
        Dict[str, Any]
            The mock sensor payload containing fsr values and vitals.
        """
        timestamp = time.time()
        bounds = self.RANGES[self.mode]
        
        # Smoothly update all sensors
        self.current_fsr1, fsr1 = self._smooth_update(self.current_fsr1, *bounds["fsr"], max_delta=20.0, noise_std=2.0)
        self.current_fsr2, fsr2 = self._smooth_update(self.current_fsr2, *bounds["fsr"], max_delta=20.0, noise_std=2.0)
        self.current_fsr3, fsr3 = self._smooth_update(self.current_fsr3, *bounds["fsr"], max_delta=20.0, noise_std=2.0)
        self.current_fsr4, fsr4 = self._smooth_update(self.current_fsr4, *bounds["fsr"], max_delta=20.0, noise_std=2.0)
        
        self.current_temperature, temp = self._smooth_update(self.current_temperature, *bounds["temp"], max_delta=0.05, noise_std=0.02)
        self.current_heart_rate, hr = self._smooth_update(self.current_heart_rate, *bounds["hr"], max_delta=1.0, noise_std=0.5)
        self.current_spo2, spo2 = self._smooth_update(self.current_spo2, *bounds["spo2"], max_delta=0.2, noise_std=0.1)

        # Build standard payload matching expected ESP32 output
        payload = {
            "timestamp": timestamp,
            "fsr1": round(max(0, fsr1), 2),
            "fsr2": round(max(0, fsr2), 2),
            "fsr3": round(max(0, fsr3), 2),
            "fsr4": round(max(0, fsr4), 2),
            "temperature": round(temp, 2),
            "spo2": round(spo2, 1),
            "heart_rate": round(hr, 1)
        }
        
        return payload
