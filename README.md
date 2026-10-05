# Customer Churn Prediction

Predicts whether a telecom customer is likely to churn (cancel their service), using account and service-usage data. Built as an end-to-end ML pipeline: data preprocessing → model training → evaluation → feature importance analysis.

## Problem

Customer churn is expensive for subscription-based businesses — it's far cheaper to retain an existing customer than acquire a new one. This project builds a classifier that flags at-risk customers based on their contract type, billing, and service usage, so a business could proactively intervene.

## Dataset

`data/customer_churn.csv` — 2,000 customer records with the following features:

| Column | Description |
|---|---|
| `SeniorCitizen` | Whether the customer is a senior citizen (0/1) |
| `tenure` | Number of months the customer has stayed |
| `Contract` | Contract type (Month-to-month, One year, Two year) |
| `InternetService` | Type of internet service (DSL, Fiber optic, No) |
| `TechSupport` | Whether the customer has tech support (Yes/No) |
| `PaymentMethod` | How the customer pays their bill |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total amount billed to date |
| `Churn` | Target variable — did the customer churn? (Yes/No) |

*Note: this dataset is synthetically generated with realistic churn-driving relationships (e.g., month-to-month contracts and lack of tech support increase churn risk) for portfolio/demonstration purposes.*

## Approach

1. **Preprocessing** — dropped the ID column, label-encoded categorical features, encoded the target as binary.
2. **Train/test split** — 80/20 split, stratified on the target to preserve class balance.
3. **Models trained:**
   - Logistic Regression (on scaled features)
   - Random Forest Classifier
4. **Evaluation** — accuracy, precision, recall, F1-score, and confusion matrix for both models.
5. **Feature importance** — extracted from the Random Forest to identify the strongest churn drivers.

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.713 | 0.600 | 0.218 | 0.320 |
| Random Forest | 0.695 | 0.512 | 0.331 | 0.402 |

**Top churn drivers (Random Forest feature importance):** `TotalCharges`, `MonthlyCharges`, `tenure`, `Contract`, `PaymentMethod`.

**Takeaway:** Random Forest achieves a better recall and F1-score, meaning it catches more actual churn cases — which matters more than raw accuracy in a churn problem, since missing an at-risk customer is costlier than a false alarm.

## How to run

```bash
pip install -r requirements.txt
python churn_prediction.py
```

## Possible next steps

- Address class imbalance with SMOTE or class weighting to improve recall further
- Try gradient boosting models (XGBoost/LightGBM) for comparison
- Add hyperparameter tuning via GridSearchCV
- Build a simple Streamlit dashboard to visualize predictions

## Tech stack

Python · pandas · scikit-learn · NumPy
