"""cleans the data, strips text artifacts from numeric columns, and handles extreme financial outliers."""

import pandas as pd
import numpy as np

def clean_financial_string(val):
    """Strip trailing characters and convert messy text to float."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip().replace('_', '')
    if val_str in ['', 'NA', 'nan']:
        return np.nan
    try:
        return float(val_str)
    except ValueError:
        return np.nan

def process_data(file_path, is_train=True):
    """Loads dataset, cleans artifacts, and selects core risk features."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Clean messy string columns
    text_cols = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 
                 'Outstanding_Debt', 'Amount_invested_monthly', 'Changed_Credit_Limit']
    
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_financial_string)
            
    # 2. Handle Extreme Financial Outliers
    if 'Age' in df.columns:
        df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = np.nan
    if 'Interest_Rate' in df.columns:
        df.loc[df['Interest_Rate'] > 100, 'Interest_Rate'] = np.nan
    if 'Num_of_Loan' in df.columns:
        df.loc[(df['Num_of_Loan'] < 0) | (df['Num_of_Loan'] > 20), 'Num_of_Loan'] = np.nan
    
    # 3. Select standard predictive features
    features = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
        'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Outstanding_Debt', 
        'Credit_Utilization_Ratio', 'Total_EMI_per_month'
    ]
    
    # Ensure all features exist in the current dataframe
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].copy()
    
    if is_train:
        # Target Encoding for Risk Tiers
        target_map = {'Good': 2, 'Standard': 1, 'Poor': 0}
        y = df['Credit_Score'].map(target_map)
        return X, y
    
    # For testing/inference, return features and the applicant ID
    return X, df['ID']