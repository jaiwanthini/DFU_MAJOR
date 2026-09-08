def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))

class RiskModel:
    """
    Computes Ground-Truth DFU Risk Scores dynamically based on sensor behavior.
    """
    
    @staticmethod
    def compute_balancing_label(f0, f1, f2, f3, temp, hr, spo2, rolling_fsr):
        '''
        Compute a rough heuristic label used EXCLUSIVELY for generating a balanced cohort.
        The official, mathematically rigorous label is calculated during preprocessing.
        '''
        # If temp/hr/spo2 are NaN, default to baseline for score calculation to prevent crash
        t_val = temp if temp == temp else 31.0
        h_val = hr if hr == hr else 60.0
        s_val = spo2 if spo2 == spo2 else 98.0
        
        
        # Normalize features (use rolling FSR to represent actual physiological load)
        p_score = clamp(rolling_fsr / 1023.0, 0.0, 1.0)
        t_score = clamp((t_val - 31.0) / 7.0, 0.0, 1.0)
        h_score = clamp((h_val - 60.0) / 60.0, 0.0, 1.0)
        s_score = clamp((100.0 - s_val) / 12.0, 0.0, 1.0)
        
        # Weightings based on DFU etiology
        risk_val = (0.35 * p_score) + (0.25 * t_score) + (0.20 * s_score) + (0.20 * h_score)
        risk_score = risk_val * 100.0
        
        if risk_score <= 34:
            return 0  # Low Risk
        elif risk_score <= 64:
            return 1  # Medium Risk
        else:
            return 2  # High Risk
