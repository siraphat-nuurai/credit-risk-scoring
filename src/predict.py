"""Runs batch predictions on new, unseen test data using the saved models, ensuring there is no data leakage."""

import pandas as pd
import joblib
import os
from data_prep import process_data

def run_batch_inference():
    print("Initializing Batch Risk Evaluation...")
    
    # Ensure processed directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    # 1. Load unseen test data
    X_test, test_ids = process_data('data/raw/credit-score-test.csv', is_train=False)
    
    # 2. Load trained assets
    try:
        model = joblib.load('models/credit_risk_model.pkl')
        imputation_params = joblib.load('models/imputation_params.pkl')
    except FileNotFoundError:
        print("Error: Trained models not found. Please run 'python src/train.py' first.")
        return
    
    # 3. Apply exact same imputation values from training to test data
    for col in X_test.columns:
        if col in imputation_params:
            X_test[col] = X_test[col].fillna(imputation_params[col])
            
    # 4. Predict Risk Tiers
    predictions = model.predict(X_test)
    
    # Map predictions back to business logic labels
    reverse_map = {2: 'Good (Low Risk)', 1: 'Standard (Medium Risk)', 0: 'Poor (High Risk)'}
    predicted_labels = [reverse_map[p] for p in predictions]
    
    # 5. Export Results
    output_df = pd.DataFrame({
        'Applicant_ID': test_ids,
        'Assigned_Risk_Tier': predicted_labels
    })
    
    output_path = 'data/processed/test_predictions.csv'
    output_df.to_csv(output_path, index=False)
    print(f"Batch inference complete. Results exported to: {output_path}")

if __name__ == "__main__":
    run_batch_inference()