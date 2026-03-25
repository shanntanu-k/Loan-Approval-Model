from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "loan_data.csv"
MODEL_PATH = "model.pkl"
METADATA_PATH = "model_metadata.json"

NUMERIC_FEATURES = ["Income", "Credit_Score", "Loan_Amount", "DTI_Ratio"]
CATEGORICAL_FEATURES = ["Employment_Status"]
TARGET_COLUMN = "Approval"
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
TARGET_MAP = {"Approved": 1, "Rejected": 0}


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def load_training_data(path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df = df[REQUIRED_COLUMNS].dropna(subset=REQUIRED_COLUMNS).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_MAP)
    df = df.dropna(subset=[TARGET_COLUMN])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def evaluate_model(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def train_and_save(data_path: str = DATA_PATH) -> dict[str, float]:
    X, y = load_training_data(data_path)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)
    y_prob = pipeline.predict_proba(X_valid)[:, 1]
    metrics = evaluate_model(y_valid, y_pred, y_prob)

    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": data_path,
        "rows_used": int(len(X)),
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "employment_status_values": sorted(
            pd.read_csv(data_path)["Employment_Status"].dropna().astype(str).str.lower().unique().tolist()
        ),
        "metrics": metrics,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metrics


if __name__ == "__main__":
    results = train_and_save()
    print("Model trained and saved to model.pkl")
    print(f"Validation metrics: {results}")
