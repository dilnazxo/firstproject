"""
Customer Churn Prediction Pipeline
Industrial Practice ML Project

This script trains machine learning models for customer churn prediction.
It supports either:
1. a real dataset located at data/customer_churn.csv, or
2. synthetic fallback data generated automatically for demonstration.

Author: Amanzholova Dilnaz
Program: Mathematical and Computational Sciences
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "customer_churn.csv"
FIGURES_DIR = ROOT_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"

FIGURES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def create_synthetic_churn_data(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """Create a synthetic customer churn dataset for demonstration."""
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(1, 72, size=n_samples)
    monthly_charges = rng.normal(70, 25, size=n_samples).clip(20, 130)
    total_charges = tenure * monthly_charges + rng.normal(0, 150, size=n_samples)
    support_calls = rng.poisson(1.5, size=n_samples)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.35, 0.50, 0.15])
    tech_support = rng.choice(["Yes", "No"], size=n_samples, p=[0.38, 0.62])
    payment_method = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        size=n_samples,
        p=[0.35, 0.25, 0.25, 0.15],
    )

    # Construct churn probability from interpretable business factors.
    logit = (
        -2.2
        + 1.2 * (contract == "Month-to-month")
        + 0.8 * (internet_service == "Fiber optic")
        + 0.7 * (tech_support == "No")
        + 0.015 * (monthly_charges - 70)
        - 0.025 * tenure
        + 0.18 * support_calls
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    df = pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly_charges.round(2),
        "TotalCharges": total_charges.round(2),
        "SupportCalls": support_calls,
        "Contract": contract,
        "InternetService": internet_service,
        "TechSupport": tech_support,
        "PaymentMethod": payment_method,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })

    return df


def load_data() -> pd.DataFrame:
    """Load real dataset if available, otherwise generate synthetic data."""
    if DATA_PATH.exists():
        print(f"Loading dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("Real dataset was not found. Creating synthetic demonstration dataset.")
        df = create_synthetic_churn_data()
        df.to_csv(DATA_PATH, index=False)
        print(f"Synthetic dataset saved to {DATA_PATH}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning for churn dataset."""
    df = df.copy()
    df = df.drop_duplicates()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def save_churn_distribution(df: pd.DataFrame) -> None:
    """Save churn distribution chart."""
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Churn")
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_distribution.png", dpi=200)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    """Save correlation heatmap for numerical variables."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for numerical and categorical columns."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    return preprocessor


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model and return metrics."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close()

    return metrics


def save_feature_importance(best_pipeline: Pipeline, X: pd.DataFrame) -> None:
    """Save feature importance chart for Random Forest model."""
    classifier = best_pipeline.named_steps["classifier"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    if not hasattr(classifier, "feature_importances_"):
        return

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=200)
    plt.close()


def main() -> None:
    df = load_data()
    df = clean_data(df)

    print("\nDataset preview:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    save_churn_distribution(df)
    save_correlation_heatmap(df)

    target = "Churn"
    if target not in df.columns:
        raise ValueError("Target column 'Churn' was not found in the dataset.")

    X = df.drop(columns=[target])
    y = df[target]

    if y.dtype == "object":
        y = y.map({"Yes": 1, "No": 0, "yes": 1, "no": 0, "True": 1, "False": 0})
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        ),
    }

    results = []
    trained_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(ROOT_DIR / "model_results.csv", index=False)

    print("\nModel comparison:")
    print(results_df)

    best_model_name = results_df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]

    joblib.dump(best_pipeline, MODELS_DIR / "best_churn_model.joblib")
    print(f"\nBest model saved: {best_model_name}")

    save_feature_importance(best_pipeline, X_train)

    print("\nGenerated files:")
    print(f"- {FIGURES_DIR / 'churn_distribution.png'}")
    print(f"- {FIGURES_DIR / 'correlation_heatmap.png'}")
    print(f"- {FIGURES_DIR / 'feature_importance.png'}")
    print(f"- {ROOT_DIR / 'model_results.csv'}")
    print(f"- {MODELS_DIR / 'best_churn_model.joblib'}")


if __name__ == "__main__":
    main()
