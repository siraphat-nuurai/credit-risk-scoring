"""Handles median imputation, trains the LightGBM model, evaluates its accuracy, and saves the final models to the models/directory."""

import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
from data_prep import process_data

def train_pipeline():
    print("🚀 Starting Training Pipeline...")
    
    # Ensure save directory exists
    os.makedirs('models', exist_ok=True)
    
    # 1. Process Data
    X, y = process_data('data/raw/credit-score-train.csv', is_train=True)
    
    # 2. Impute Missing Values (Calculate Medians & Save for Inference)
    imputation_params = {}
    for col in X.columns:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        imputation_params[col] = median_val
        
    joblib.dump(imputation_params, 'models/imputation_params.pkl')
    print("✅ Imputation parameters saved.")
    
    # 3. Train-Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 4. Train LightGBM Classifier
    print("📈 Training LightGBM Risk Model...")
    model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=10, 
        objective='multiclass', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluate Model
    preds = model.predict(X_val)
    print("\n=== Model Validation Results ===")
    print(f"Accuracy: {accuracy_score(y_val, preds):.4f}")
    print(classification_report(y_val, preds, target_names=['Poor', 'Standard', 'Good']))
    
    # 6. Save Model
    joblib.dump(model, 'models/credit_risk_model.pkl')
    print("💾 Model successfully saved to models/credit_risk_model.pkl")
    
    # 7. Generate Feature Importance Plot
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(X.columns)[indices], palette='viridis')
    plt.title('Feature Importance (Risk Drivers)', fontsize=14)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    print("📊 Feature importance chart saved.")

if __name__ == "__main__":
    train_pipeline()