"""
===========================================================
PadaDrishti - ESP32 Smart Insole Digital Twin Simulator
===========================================================
Generates continuous, realistic temporal sequences identical
to physical hardware for LSTM sliding window prediction.
"""
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from patient import Patient
from activity_model import ActivityModel
from sensor_model import ESP32DigitalTwin
from risk_model import RiskModel

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_FILE = os.path.join(ROOT_DIR, "data", "raw", "dfu_raw_dataset_100000.csv")
TOTAL_ROWS = 100000

# Deterministic build
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def simulate_patient_session(patient, start_time):
    """Orchestrates the Patient, Activity, and Hardware Twin models for a single session."""
    num_seconds = random.randint(30, 60)
    activity = ActivityModel(num_seconds, patient)
    twin = ESP32DigitalTwin(patient)
    
    rows = []
    curr_time = start_time
    
    for act_type, duration in activity.phases:
        params = activity.get_activity_params(act_type)
        is_walking, speed_hz, _ = params
        
        state_rows = []
        state_stats = {'f0': [], 'f1': [], 'f2': [], 'f3': [], 'temp': [], 'hr': [], 'spo2': [], 'rolling_fsr': []}
        
        for _ in range(duration):
            if is_walking:
                activity.gait_phase += speed_hz * 2 * np.pi
                
            # Step the digital twin physics
            f0, f1, f2, f3, temp, hr, spo2, rolling_fsr = twin.step(params, activity.gait_phase)
            
            state_stats['f0'].append(f0)
            state_stats['f1'].append(f1)
            state_stats['f2'].append(f2)
            state_stats['f3'].append(f3)
            state_stats['temp'].append(temp)
            state_stats['hr'].append(hr)
            state_stats['spo2'].append(spo2)
            state_stats['rolling_fsr'].append(rolling_fsr)
            
            state_rows.append({
                "patient_id": patient.patient_id,
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                "fsr_1": f0,
                "fsr_2": f1,
                "fsr_3": f2,
                "fsr_4": f3,
                "temperature": temp,
                "spo2": spo2,
                "heart_rate": hr,
                "risk_label": None # Placeholder
            })
            curr_time += timedelta(seconds=1)
            
        # Compute the risk label for the entire state based on average physiological load
        avg_f0 = np.nanmean(state_stats['f0'])
        avg_f1 = np.nanmean(state_stats['f1'])
        avg_f2 = np.nanmean(state_stats['f2'])
        avg_f3 = np.nanmean(state_stats['f3'])
        avg_temp = np.nanmean(state_stats['temp'])
        avg_hr = np.nanmean(state_stats['hr'])
        avg_spo2 = np.nanmean(state_stats['spo2'])
        avg_rolling_fsr = np.nanmean(state_stats['rolling_fsr'])
        
        label = RiskModel.compute_balancing_label(avg_f0, avg_f1, avg_f2, avg_f3, avg_temp, avg_hr, avg_spo2, avg_rolling_fsr)
        
        # We output None to the CSV so that the official label is exclusively assigned by the preprocessing pipeline.
        # But we still attach the balancing label internally for cohort balancing.
        for r in state_rows:
            r['internal_balancing_label'] = label
            
        rows.extend(state_rows)
            
    return rows

def main():
    print(f"Initializing PadaDrishti ESP32 Digital Twin Simulator (Multi-Session Patient Diversity)...")
    
    target_rows_per_class = TOTAL_ROWS / 3.0
    label_counts = {0: 0, 1: 0, 2: 0}
    
    all_rows = []
    patient_count = 1
    start_time_base = datetime(2026, 1, 1, 8, 0, 0)
    
    while len(all_rows) < TOTAL_ROWS:
        # Dynamic cohort balancing
        needed = {k: target_rows_per_class - v for k, v in label_counts.items()}
        target_class = max(needed, key=needed.get)
        
        if target_class == 0:
            severity = random.uniform(0.0, 0.3)
        elif target_class == 1:
            severity = random.uniform(0.35, 0.65)
        else:
            severity = random.uniform(0.7, 1.0)
            
        pid = f"P{patient_count:05d}"
        patient = Patient(pid, severity)
        
        # Give each patient 1 to 5 separate sessions to capture session-to-session variability
        num_sessions = random.randint(1, 5)
        patient_start_time = start_time_base + timedelta(days=patient_count)
        
        for session_idx in range(num_sessions):
            session_start_time = patient_start_time + timedelta(hours=session_idx * 4)
            session_rows = simulate_patient_session(patient, session_start_time)
            
            if len(all_rows) + len(session_rows) > TOTAL_ROWS:
                session_rows = session_rows[:TOTAL_ROWS - len(all_rows)]
                
            for r in session_rows:
                label_counts[r['internal_balancing_label']] += 1
                
            # Remove the internal label before extending
            for r in session_rows:
                del r['internal_balancing_label']
                
            all_rows.extend(session_rows)
            
            if len(all_rows) >= TOTAL_ROWS:
                break
                
        patient_count += 1
        
        if patient_count % 100 == 0:
            print(f"Generated {len(all_rows)} rows across {patient_count} unique patients...")

    df = pd.DataFrame(all_rows)
    
    print("Shuffling patients to prevent sequential clustering...")
    unique_patients = df['patient_id'].unique().tolist()
    random.shuffle(unique_patients)
    
    patient_order = {pid: i for i, pid in enumerate(unique_patients)}
    df['sort_key'] = df['patient_id'].map(patient_order)
    df = df.sort_values(by=['sort_key', 'timestamp']).drop('sort_key', axis=1)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n=======================================================")
    print("Digital Twin Generation Complete!")
    print(f"Total Rows      : {len(df)}")
    print(f"Total Patients  : {patient_count - 1}")
    print("Final Simulator Class Distribution (Based on internal heuristics, actual labels assigned in preprocessing):")
    # Since we set risk_label to None, this won't print anything useful.
    # The actual distribution will be reported by sequence_generator.py
    print(f"Output File     : {os.path.abspath(OUTPUT_FILE)}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
