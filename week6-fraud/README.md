# Credit Card Fraud Detection

## Problem
Detect fraudulent credit card transactions. Only 492 of 284,807 rows
(0.17%) are fraud, so accuracy is a misleading metric.

## Dataset
Kaggle "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud).
Not committed to the repo. Download `creditcard.csv` into `data/`.

## Method
- Stratified 60/20/20 train/validation/test split (seed 42)
- Pipeline: StandardScaler + LogisticRegression(class_weight="balanced")
- Scaler fit on training data only (no leakage)
- Test set untouched so far

## Results (validation, threshold 0.5)
| Model | Accuracy | Fraud precision | Fraud recall | Fraud F1 |
|---|---|---|---|---|
| Always-genuine baseline | 0.9983 | 0.000 | 0.000 | 0.000 |
| Logistic regression (balanced) | 0.975 | 0.060 | 0.899 | 0.112 |

(PR-AUC, tuned threshold and test results: to be added)

## What went wrong / lessons
- The baseline scores 99.83% accuracy while catching zero fraud.
- Balanced weights give high recall but very low precision at 0.5.
- Next: choose the threshold on validation using the PR curve.

## How to run
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook   (run from the project root, open notebooks/fraud.ipynb)