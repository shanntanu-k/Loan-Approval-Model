from __future__ import annotations

import os
from typing import Any

import joblib
from flask import Flask, jsonify, render_template, request

from train import train_and_save

MODEL_PATH = "model.pkl"
NUMERIC_FIELDS = {
    "Income": (0, 1_000_000),
    "Credit_Score": (300, 900),
    "Loan_Amount": (0, 2_000_000),
    "DTI_Ratio": (0, 100),
}
ALLOWED_EMPLOYMENT_STATUS = {"employed", "unemployed"}

app = Flask(__name__)
_model: Any = None


def ensure_model_exists() -> None:
    if os.path.exists(MODEL_PATH):
        return
    train_and_save()


def get_model() -> Any:
    global _model
    if _model is None:
        ensure_model_exists()
        _model = joblib.load(MODEL_PATH)
    return _model


def validate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    cleaned: dict[str, Any] = {}
    errors: list[str] = []

    for field, (low, high) in NUMERIC_FIELDS.items():
        raw_value = payload.get(field)
        if raw_value is None or str(raw_value).strip() == "":
            errors.append(f"{field} is required.")
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number.")
            continue

        if value < low or value > high:
            errors.append(f"{field} must be between {low} and {high}.")
            continue

        cleaned[field] = value

    status = str(payload.get("Employment_Status", "")).strip().lower()
    if status not in ALLOWED_EMPLOYMENT_STATUS:
        allowed = ", ".join(sorted(ALLOWED_EMPLOYMENT_STATUS))
        errors.append(f"Employment_Status must be one of: {allowed}.")
    else:
        cleaned["Employment_Status"] = status

    return cleaned, errors


@app.route("/")
def home() -> str:
    return render_template("index.html")


@app.route("/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200


@app.route("/predict", methods=["POST"])
def predict() -> tuple[Any, int]:
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()

    cleaned, errors = validate_payload(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    model = get_model()
    feature_row = [[cleaned["Income"], cleaned["Credit_Score"], cleaned["Loan_Amount"], cleaned["DTI_Ratio"], cleaned["Employment_Status"]]]
    prediction = int(model.predict(feature_row)[0])
    probability = float(model.predict_proba(feature_row)[0][prediction])
    confidence = round(probability * 100, 2)

    decision = "Approved" if prediction == 1 else "Rejected"
    message = f"Loan {decision} (Confidence: {confidence}%)"

    return (
        jsonify(
            {
                "ok": True,
                "prediction": prediction,
                "decision": decision,
                "confidence": confidence,
                "message": message,
            }
        ),
        200,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
