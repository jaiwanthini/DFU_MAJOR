import random
import numpy as np

class ActivityState:
    SITTING = "Sitting"
    STANDING = "Standing"
    SLOW_WALKING = "Slow Walking"
    NORMAL_WALKING = "Normal Walking"
    FAST_WALKING = "Fast Walking"
    STAIR_CLIMBING = "Stair Climbing"
    SUSTAINED_PRESSURE = "Sustained Pressure"
    RECOVERY = "Recovery"

class ActivityModel:
    """
    Simulates the patient's physical activity sequence.
    This creates realistic bouts of movement followed by rest.
    """
    def __init__(self, num_seconds, patient):
        self.num_seconds = num_seconds
        self.patient = patient
        
        # Session Variability (does not change patient baseline, but affects this specific session)
        self.session_fatigue = random.uniform(0.9, 1.1)
        self.session_ambient_temp_offset = random.uniform(-0.5, 0.5)
        
        self.phases = self._generate_phases(num_seconds)
        self.gait_phase = random.uniform(0, 2 * np.pi)
        
    def _generate_phases(self, total_time):
        """Creates a realistic sequence of activities for the session."""
        activities = [
            ActivityState.SITTING,
            ActivityState.STANDING,
            ActivityState.SLOW_WALKING,
            ActivityState.NORMAL_WALKING,
            ActivityState.FAST_WALKING,
            ActivityState.STAIR_CLIMBING,
            ActivityState.SUSTAINED_PRESSURE,
            ActivityState.RECOVERY
        ]
        
        probs = [0.10, 0.10, 0.20, 0.25, 0.10, 0.05, 0.10, 0.10]
        
        phases = []
        t_rem = total_time
        while t_rem > 0:
            duration = random.randint(3, 10)
            duration = min(duration, t_rem)
            
            activity = random.choices(activities, weights=probs, k=1)[0]
            phases.append((activity, duration))
            t_rem -= duration
            
        return phases

    def get_activity_params(self, activity):
        """
        Returns the physical parameters associated with an activity.
        Returns: (is_walking, gait_speed_hz, pressure_multiplier)
        """
        is_walking = False
        speed_hz = 0.0
        p_mult = 0.0
        
        if activity == ActivityState.SITTING:
            p_mult = 0.05
        elif activity == ActivityState.STANDING:
            p_mult = 0.6
        elif activity == ActivityState.SLOW_WALKING:
            is_walking, speed_hz, p_mult = True, 0.15, 0.8
        elif activity == ActivityState.NORMAL_WALKING:
            is_walking, speed_hz, p_mult = True, 0.25, 1.0
        elif activity == ActivityState.FAST_WALKING:
            is_walking, speed_hz, p_mult = True, 0.35, 1.3
        elif activity == ActivityState.STAIR_CLIMBING:
            is_walking, speed_hz, p_mult = True, 0.20, 1.5
        elif activity == ActivityState.SUSTAINED_PRESSURE:
            p_mult = 1.5 # Not walking, but extreme sustained pressure
        elif activity == ActivityState.RECOVERY:
            p_mult = 0.1 # Very low pressure
            
        # Apply patient cadence profile if walking
        if is_walking:
            if self.patient.cadence_profile == "Fast":
                speed_hz *= 1.2
            elif self.patient.cadence_profile == "Slow":
                speed_hz *= 0.8
                
            # Apply session fatigue to cadence
            speed_hz *= (2.0 - self.session_fatigue) # High fatigue (1.1) reduces speed
            
        # Apply session fatigue to pressure
        p_mult *= self.session_fatigue
            
        return is_walking, speed_hz, p_mult
