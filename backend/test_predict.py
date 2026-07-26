import time
import requests

def run_test():
    print("Testing /predict endpoint...")
    
    # Send up to 35 samples to fill the 30-sample rolling window
    for i in range(1, 36):
        # 1. Get a simulated reading
        sim_resp = requests.get("http://127.0.0.1:5000/simulate")
        if sim_resp.status_code != 200:
            print(f"Error getting simulation: {sim_resp.text}")
            return
            
        reading = sim_resp.json()
        
        # 2. Post to /predict
        pred_resp = requests.post("http://127.0.0.1:5000/predict", json=reading)
        
        if pred_resp.status_code == 202:
            print(f"[{i}/30] Buffering... {pred_resp.json()['progress']}%")
        elif pred_resp.status_code == 200:
            print(f"\n[{i}] Prediction Triggered!")
            result = pred_resp.json()
            print(f"Risk Score: {result['risk_score']}")
            print(f"Risk Label: {result['risk_label']}")
            if 'explanation' in result:
                print(f"SHAP Summary: {result['explanation'].get('summary', 'No summary')}")
            else:
                print("No explanation field!")
            break
        else:
            print(f"Error {pred_resp.status_code}: {pred_resp.text}")
            break

if __name__ == "__main__":
    run_test()
