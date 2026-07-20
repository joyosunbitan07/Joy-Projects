"""
Test Script - Quick Prediction Runner
This loads the trained model binary and runs a prediction on a sample row.
"""

import os
import pickle
import pandas as pd
# Paths simplified to load right from the flat folder layout
MODEL_PATH = "./malware_lgb_model.pkl"
VAL_DATA_PATH = "./val_flat.csv"
DECISION_THRESHOLD = 0.38


def run_sample_prediction():
    print("[!] Initializing prediction test script...")

    # 1 This is a verification check for the model binary
    if not os.path.exists(MODEL_PATH):
        print(f"[-] Error: Model file not found at {MODEL_PATH}.")
        print("[->] Please run 'train_models.py' first to generate it.")
        return

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("[+] Model loaded successfully.")

    #2 This section here reads the expected feature name and orders them directly from the trained model
    feature_names = model.feature_name_

    #3 Creates a clean mock dictionary that sets all expected API features to 0.0
    mock_input_dict = {feature: 0.0 for feature in feature_names}

    #4 live application test
    #This triggers a few API features and then manually updates the metrics
    #by manually mapping this, the script bypasses the need for the validation CSV that I made before
    if len(feature_names) > 15:
        mock_input_dict[feature_names[0]] = 4.0
        mock_input_dict[feature_names[5]] = 10.0
        mock_input_dict[feature_names[12]] = 1.0

    mock_input_dict["total_api_calls"] = 15.0
    mock_input_dict["unique_apis_used"] = 3.0

    #5 This wraps the dictionary into a single row DF then forces the columns to follow the exact order in feature_names
    X_sample = pd.DataFrame([mock_input_dict], columns=feature_names)
    print("[+] Successfully constructed a self-contained test sample payload.")
    #6 Predicts the prbability and applies the threshold
    prob = model.predict_proba(X_sample)[0, 1]
    prediction = 1 if prob >= DECISION_THRESHOLD else 0
    result_text = "Malware (High Risk)" if prediction == 1 else "Benign (Low Risk)"

    print("      PREDICTION RESULTS      ")
    print("==============================")
    print(f"Predicted Probability: {prob:.4f}")
    print(f"Applied Threshold:     {DECISION_THRESHOLD}")
    print(f"Final Decision:        {result_text}")

if __name__ == "__main__":
    run_sample_prediction()





