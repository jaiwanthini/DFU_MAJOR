import random
import numpy as np

def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))

class ESP32DigitalTwin:
    """
    Digital Twin of the ESP32 Hardware.
    Generates exact sensor streams based on biological parameters and activity.
    """
    def __init__(self, patient):
        self.patient = patient
        
        # Running physiological state
        self.current_temp = patient.base_temp
        self.current_hr = patient.base_hr
        self.current_spo2 = patient.base_spo2
        self.pressure_drift = 0.0

    def apply_imperfections(self, val, vtype):
        """Simulates real ESP32 hardware noise, dropouts, and spikes."""
        if random.random() < 0.003: # 0.3% missing
            return np.nan
            
        if random.random() < 0.001: # 0.1% spike
            if vtype == 'fsr': return 1023
            if vtype == 'temp': return round(random.uniform(38.0, 42.0), 2)
            if vtype == 'hr': return int(random.uniform(150, 200))
            if vtype == 'spo2': return round(random.uniform(70, 85), 1)
            
        return val

    def step(self, activity_params, gait_phase):
        """Advances the digital twin by 1 second, observing hardware behaviour."""
        is_walking, speed_hz, p_mult = activity_params
        
        # 1. FSR Model
        
        # Adjust base scales for gait symmetry
        if self.patient.gait_symmetry == "Heel_Bias":
            h_mult, f_mult = 1.3, 0.7
        elif self.patient.gait_symmetry == "Forefoot_Bias":
            h_mult, f_mult = 0.7, 1.3
        else:
            h_mult, f_mult = 1.0, 1.0
            
        if is_walking:
            # Random Walk Drift
            self.pressure_drift += random.gauss(0, self.patient.severity * 5 + 2)
            amplitude = self.patient.foot_strike_force * p_mult
            floor = self.patient.min_pressure * p_mult
            
            # Sinusoidal gait scaled to [0, 1] so it never drops below the floor during active gait
            wave0 = (np.sin(gait_phase) + 1.0) / 2.0
            wave1 = (np.sin(gait_phase - np.pi/4) + 1.0) / 2.0
            wave2 = (np.sin(gait_phase - np.pi/2) + 1.0) / 2.0
            wave3 = (np.sin(gait_phase - 3*np.pi/4) + 1.0) / 2.0
            
            f0_raw = floor + (amplitude * 1.0 * wave0 * h_mult)    # Heel
            f1_raw = floor + (amplitude * 0.6 * wave1 * h_mult)    # Midfoot
            f2_raw = floor + (amplitude * 0.9 * wave2 * f_mult)    # Forefoot
            f3_raw = floor + (amplitude * 0.7 * wave3 * f_mult)    # Toe
        else:
            # Sitting / Standing (Recovery periods)
            self.pressure_drift *= 0.8  # Drift decays quickly
            if p_mult < 0.1: # Sitting
                floor = random.uniform(5.0, 25.0) # Near zero pressure
            else: # Standing
                floor = self.patient.min_pressure * p_mult
                
            f0_raw = floor * 1.2 * h_mult
            f1_raw = floor * 0.8 * h_mult
            f2_raw = floor * 1.0 * f_mult
            f3_raw = floor * 0.5 * f_mult

        # Add Gaussian noise and hardware bias
        f0 = clamp(int(f0_raw + self.pressure_drift + self.patient.sensor_offsets["fsr_1"] + random.gauss(0, 5)), 0, 1023)
        f1 = clamp(int(f1_raw + self.pressure_drift + self.patient.sensor_offsets["fsr_2"] + random.gauss(0, 5)), 0, 1023)
        f2 = clamp(int(f2_raw + self.pressure_drift + self.patient.sensor_offsets["fsr_3"] + random.gauss(0, 5)), 0, 1023)
        f3 = clamp(int(f3_raw + self.pressure_drift + self.patient.sensor_offsets["fsr_4"] + random.gauss(0, 5)), 0, 1023)
        
        mean_fsr = (f0 + f1 + f2 + f3) / 4.0
        
        # Track rolling pressure for physiological coupling (so sine wave doesn't suppress EMA)
        self.rolling_fsr = getattr(self, 'rolling_fsr', mean_fsr)
        self.rolling_fsr = (self.rolling_fsr * 0.9) + (mean_fsr * 0.1)

        # 2. Temperature Model (EMA based on sustained rolling pressure)
        target_temp = self.patient.base_temp + (self.rolling_fsr / 300.0) * 5.0 * self.patient.sensitivity_factor
        self.current_temp += (target_temp - self.current_temp) * self.patient.healing_factor
        self.current_temp += random.gauss(0, 0.02)
        
        # 3. Heart Rate Model (Activity + Temperature stress)
        target_hr = self.patient.base_hr 
        
        if is_walking:
            if speed_hz > 0.25: # Fast Walking / Stairs
                target_hr += 35.0 
            else: # Slow / Normal Walking
                target_hr += 20.0
        elif p_mult > 1.0: # Sustained pressure
            target_hr += 25.0 
        else: # Standing / Recovery / Sitting
            target_hr += 5.0
            
        # Physiological stress multiplier
        hr_stress = (self.rolling_fsr / 1000.0) * 10.0 + (self.current_temp - 31.0) * 3.0
        target_hr += hr_stress * self.patient.sensitivity_factor
            
        self.current_hr += (target_hr - self.current_hr) * 0.1
        self.current_hr += random.gauss(0, 0.5)

        # 4. SpO2 Model (Slow decay in severe cases)
        spo2_drop = (self.patient.severity * 8.0) + (self.rolling_fsr / 300.0) * 2.0
        target_spo2 = self.patient.base_spo2 - spo2_drop
        self.current_spo2 += (target_spo2 - self.current_spo2) * 0.05
        self.current_spo2 += random.gauss(0, 0.1)

        # Output Formatting (Digital Twin constraints)
        temp_out = round(clamp(self.current_temp + self.patient.sensor_offsets["temp"], 30.0, 42.0), 2)
        hr_out = int(clamp(self.current_hr + self.patient.sensor_offsets["hr"], 40, 200))
        spo2_out = round(clamp(self.current_spo2 + self.patient.sensor_offsets["spo2"], 80.0, 100.0), 1)

        # Final Hardware Spikes/Drops
        f0 = self.apply_imperfections(f0, 'fsr')
        f1 = self.apply_imperfections(f1, 'fsr')
        f2 = self.apply_imperfections(f2, 'fsr')
        f3 = self.apply_imperfections(f3, 'fsr')
        temp_out = self.apply_imperfections(temp_out, 'temp')
        hr_out = self.apply_imperfections(hr_out, 'hr')
        spo2_out = self.apply_imperfections(spo2_out, 'spo2')

        return f0, f1, f2, f3, temp_out, hr_out, spo2_out, self.rolling_fsr
