import random

class Patient:
    """
    Represents the internal physiological profile of a virtual patient.
    These variables are NOT outputted to the ESP32 dataset, but they 
    dictate how the patient's body inherently responds to physical activity.
    """
    def __init__(self, patient_id, target_severity):
        self.patient_id = patient_id
        
        # severity [0.0, 1.0] controls the base likelihood of disease manifestation
        self.severity = target_severity 
        
        # 1. Demographics
        self.age = random.randint(40, 85)
        self.weight_kg = random.uniform(45.0, 110.0)
        self.height_cm = random.uniform(145.0, 190.0)
        self.bmi = self.weight_kg / ((self.height_cm / 100.0) ** 2)
        
        self.is_obese = self.bmi > 30.0
        self.is_athletic = self.bmi < 25.0 and self.age < 60
        self.is_elderly = self.age >= 70
        
        # 2. Base Physiological States (Fixed for this patient)
        # HR: 60-90 bpm (higher for obese/elderly/severe)
        base_hr_val = 60.0 + (self.severity * 10.0) + (5.0 if self.is_obese else 0.0) - (5.0 if self.is_athletic else 0.0)
        self.base_hr = max(60.0, min(90.0, base_hr_val + random.uniform(-2, 2)))
        
        # Temp: 34.0 - 36.5 (higher for severe/obese)
        base_temp_val = 34.0 + (self.severity * 1.5) + (0.5 if self.is_obese else 0.0)
        self.base_temp = max(34.0, min(36.5, base_temp_val + random.uniform(-0.2, 0.2)))
        
        # SpO2: 95 - 100 (lower for severe/elderly)
        base_spo2_val = 99.0 - (self.severity * 4.0) - (2.0 if self.is_elderly else 0.0)
        self.base_spo2 = max(95.0, min(100.0, base_spo2_val + random.uniform(-0.5, 0.5)))
        
        # 3. Biomechanical Profiles
        # Enforce strict pressure boundaries to avoid class overlap during active gait
        if self.severity < 0.33:
            self.min_pressure = random.uniform(150, 200)
            self.max_pressure = random.uniform(250, 350)
        elif self.severity < 0.66:
            self.min_pressure = random.uniform(350, 450)
            self.max_pressure = random.uniform(550, 700)
        else:
            self.min_pressure = random.uniform(700, 800)
            self.max_pressure = random.uniform(900, 1023)
            
        # Modify peak pressure based on weight
        weight_factor = (self.weight_kg - 75.0) / 75.0 # -40% to +46%
        self.max_pressure = min(1023, self.max_pressure * (1.0 + weight_factor * 0.3))
        self.foot_strike_force = self.max_pressure - self.min_pressure
        
        # 4. Behavioral & Gait Profiles
        self.cadence_profile = random.choice(["Slow", "Normal", "Fast"])
        if self.is_elderly or self.is_obese:
            self.cadence_profile = random.choice(["Slow", "Normal"])
            
        self.gait_symmetry = random.choice(["Balanced", "Heel_Bias", "Forefoot_Bias"])
        self.recovery_speed = "Fast" if self.is_athletic else ("Slow" if self.is_elderly else "Medium")
        
        # Natural biomechanical variations based on profile
        if self.recovery_speed == "Fast":
            self.healing_factor = random.uniform(0.04, 0.06)
        elif self.recovery_speed == "Slow":
            self.healing_factor = random.uniform(0.005, 0.02)
        else:
            self.healing_factor = random.uniform(0.02, 0.04)
            
        self.sensitivity_factor = random.uniform(0.8, 1.2) + (0.2 if self.is_obese else 0.0)
        
        # 5. Hardware Calibration Offsets (Unique fixed bias for this ESP32 unit)
        self.sensor_offsets = {
            "fsr_1": random.gauss(0, 10),
            "fsr_2": random.gauss(0, 10),
            "fsr_3": random.gauss(0, 10),
            "fsr_4": random.gauss(0, 10),
            "temp": random.gauss(0, 0.2),
            "hr": random.gauss(0, 2),
            "spo2": random.gauss(0, 0.5)
        }
