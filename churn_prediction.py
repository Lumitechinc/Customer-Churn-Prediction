"""
Customer Churn Prediction
--------------------------
Predicts whether a telecom customer will churn (leave the service)
based on account and service usage attributes.

Pipeline: load data -> clean/encode -> train/test split ->
train Logistic Regression & Random Forest -> evaluate -> compare.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

RANDOM_STATE = 42


def load_data(path="data/customer_churn.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess(df):
    df = df.drop(columns=["customerID"])

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categorical features
    categorical_cols = ["Contract", "InternetService", "TechSupport", "PaymentMethod"]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "confusion_matrix": confusion_matrix(y_test, preds),
            "model": model,
        }

    # Feature importance from Random Forest
    rf_model = results["Random Forest"]["model"]
    importance = pd.Series(rf_model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)

    return results, importance, (X_test, y_test)


def print_report(results, importance):
    print("\n=== MODEL COMPARISON ===")
    for name, metrics in results.items():
        print(f"\n{name}")
        print(f"  Accuracy : {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall   : {metrics['recall']:.3f}")
        print(f"  F1-score : {metrics['f1']:.3f}")
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

    print("\n=== TOP FEATURES DRIVING CHURN (Random Forest) ===")
    print(importance.head(5).to_string())


if __name__ == "__main__":
    df = load_data()
    X, y = preprocess(df)
    results, importance, test_data = train_and_evaluate(X, y)
    print_report(results, importance)
