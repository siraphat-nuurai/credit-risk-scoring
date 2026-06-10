# Credit Risk Scoring & Risk Analytics Engine

An enterprise-grade Machine Learning system designed to evaluate applicant creditworthiness, classify financial risk tiers, and support credit risk management workflows. Built using modern pipeline architecture and a high-performance **LightGBM Ensemble Classifier**.

---

## Data Source

This dataset is sourced from the **[Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)** project on Kaggle. 
* **License:** Public Domain / CC0
* **Data Description:** The dataset contains financial and behavioral features of credit applicants, used to categorize them into "Good", "Standard", and "Poor" risk tiers.

---

## Project Summary
Real-world credit evaluation is often hindered by corrupted, non-linear financial data. This repository implements an end-to-end operational pipeline that cleans systemic data quality anomalies (e.g., negative demographic inputs, extreme interest rate artifacts) and categorizes borrower profiles into three predictive risk classifications:
* **Good** (Low Default Risk — Recommended for Standard Processing)
* **Standard** (Medium Default Risk — Flagged for Detailed Operational Review)
* **Poor** (High Default Risk — Flagged for Risk Containment or Capital Preservation Protocols)

---

## Repository Structure
Following strict production repository layouts, the pipeline is modularly organized as follows:

```text
credit-risk-scoring/
├── data/
│   ├── raw/                    # Protected directory for source datasets (e.g., credit-score-train.csv)
│   └── processed/              # Storage for cleaned features and final batch predictions
├── models/                     # Serialized production binaries and imputation parameters (.pkl / .joblib)
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis, outlier profiling, and visualization
│   └── 02_Modeling.ipynb      # Model experimentation, feature engineering, and evaluation
└── src/
    ├── __init__.py            # Makes the src directory a recognizable Python module
    ├── data_prep.py            # Financial domain cleaning, text artifact stripping, and preprocessing
    ├── predict.py              # Automated batch inference pipeline for unseen testing data
    └── train.py                # Model training, stratified cross-validation, and feature importance generation
```
---

## Installation & Environment Setup

1. Clone this repository:

```bash
git clone [https://github.com/YOUR_USERNAME/credit-risk-scoring.git](https://github.com/YOUR_USERNAME/credit-risk-scoring.git)
cd credit-risk-scoring
```

2. Install core dependencies:

```bash
pip install -r requirements.txt
```

_(Ensure your requirements.txt includes essential libraries: pandas, numpy, lightgbm, scikit-learn, joblib, matplotlib, seaborn)_

---

## Data Strategy & Management

To keep the repository clean and optimized, large raw datasets are managed locally and isolated from source control versioning via .gitignore.

1. Obtain Data: Place your open-source credit-score-train.csv and credit-score-test.csv source files directly inside the data/raw/ directory.

2. Local Security: Ensure your local file tracking respects .gitignore rules to keep massive file blobs and binary outputs untracked.

---

## Execution Guide

1. Research & Exploration
Open the notebooks inside the notebooks/ directory to review data distributions and prototyping steps:

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

2. Execute the Data Prep & Training Pipeline
To handle data leakage protection, compute median imputation parameters, train the core risk model, and auto-generate feature importance charts:

```bash
python src/train.py
```

3. Run Production Batch Inference
To load the serialized training parameters and execute risk evaluations on unseen testing applications:

```bash
python src/predict.py
```
_Outputs will be securely compiled in data/processed/test_predictions.csv._

---

## Analytical Insights & Core Drivers
Using robust evaluation frameworks, behavioral credit metrics proved significantly more predictive than basic demographic variables like Annual Income or Age. The top predictors determining default tiers include:

1. Outstanding Debt: The most significant coefficient determining risk classification.
2. Interest Rate Caps: Clear statistical thresholds showing higher risk concentration for products above an 11 interest rate.
3. Delay From Due Date: Behavioral late-payment velocities heavily shift candidates out of the "Good" tier.