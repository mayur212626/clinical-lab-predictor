# api/main.py
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI endpoint for the clinical lab predictor.
#
# Design decisions:
#   - Pydantic validation on input so we catch bad values before they hit the model
#   - SHAP explanations are optional (they're a bit slow) — toggle with ?explain=true
#   - Predictions are logged asynchronously so the response isn't delayed
#   - /health and /ready split: health = is the process running,
#     ready = is the model actually loaded and usable
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import joblib
import numpy as np
import pandas as pd
import json
import logging
import os
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

app = FastAPI(
    title="Clinical Lab Predictor API",
    description=(
        "Predicts diabetes risk from clinical lab values. "
        "Research prototype — not for clinical use without physician review."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Globals (loaded once at startup) ─────────────────────────────────────────
_model    = None
_scaler   = None
_explainer = None


@app.on_event("startup")
async def startup():
    global _model, _scaler, _explainer
    try:
        _model  = joblib.load("models/rf_model.pkl")
        _scaler = joblib.load("models/scaler.pkl")
        log.info("Model and scaler loaded successfully")

        # SHAP explainer — wrap in try/except so the API still starts even if SHAP fails
        try:
            import shap
            _explainer = shap.TreeExplainer(_model)
            log.info("SHAP explainer ready")
        except Exception as e:
            log.warning(f"SHAP not available: {e}")
    except FileNotFoundError:
        log.error("Model files not found. Run the training pipeline first.")


# ── Input schema ──────────────────────────────────────────────────────────────

class LabValues(BaseModel):
    """
    Clinical lab values for a single patient.
    Ranges are based on realistic clinical bounds — values outside these
    are almost certainly data entry errors.
    """
    pregnancies:   float = Field(..., ge=0,   le=20,  description="Number of pregnancies")
    glucose:       float = Field(..., ge=44,  le=400, description="Plasma glucose (mg/dL)")
    blood_pressure: float = Field(..., ge=20, le=200, description="Diastolic BP (mm Hg)")
    skin_thickness: float = Field(..., ge=0,  le=100, description="Triceps skin fold (mm)")
    insulin:       float = Field(..., ge=0,   le=900, description="2-hour serum insulin (μU/mL)")
    bmi:           float = Field(..., ge=10,  le=70,  description="BMI (kg/m²)")
    diabetes_pedigree: float = Field(..., ge=0.0, le=3.0, description="Diabetes pedigree function")
    age:           float = Field(..., ge=18,  le=120, description="Age in years")

    class Config:
        schema_extra = {
            "example": {
                "pregnancies": 2,
                "glucose": 148,
                "blood_pressure": 72,
                "skin_thickness": 35,
                "insulin": 50,
                "bmi": 33.6,
                "diabetes_pedigree": 0.627,
                "age": 50,
            }
        }


class BatchRequest(BaseModel):
    records: List[LabValues]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prep_features(lab: LabValues) -> pd.DataFrame:
    """Convert API input into a feature DataFrame matching the training schema."""
    row = {
        "Pregnancies":            lab.pregnancies,
        "Glucose":                lab.glucose,
        "BloodPressure":          lab.blood_pressure,
        "SkinThickness":          lab.skin_thickness,
        "Insulin":                lab.insulin,
        "BMI":                    lab.bmi,
        "DiabetesPedigreeFunction": lab.diabetes_pedigree,
        "Age":                    lab.age,
        "Outcome":                0,   # placeholder so feature engineering code doesn't break
    }
    df = pd.DataFrame([row])

    # mirror what feature_engineering.py does
    df["GlucoseInsulinRatio"] = df["Glucose"] / (df["Insulin"] + 1)
    df["BMICategory"] = int(pd.cut([lab.bmi], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])[0])
    df["AgeGroup"]    = int(pd.cut([lab.age],  bins=[0, 30, 45, 60, 120],  labels=[0, 1, 2, 3])[0])
    df = df.drop("Outcome", axis=1)

    return df


def _risk_label(prob: float) -> str:
    if prob < 0.30:  return "LOW"
    if prob < 0.60:  return "MODERATE"
    return "HIGH"


def _log_prediction(data: dict, pred: int, prob: float):
    """Append prediction to the monitoring log. Called in the background."""
    entry = {
        "ts":          datetime.utcnow().isoformat(),
        "input":       data,
        "prediction":  pred,
        "probability": prob,
    }
    os.makedirs("monitoring", exist_ok=True)
    with open("monitoring/predictions.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


@app.get("/ready")
def ready():
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    return {"status": "ready", "model": "rf_model v1.0.0"}


@app.get("/model/info")
def model_info():
    try:
        with open("models/rf_metadata.json") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(404, "Model metadata not found — run training pipeline first")


@app.post("/predict")
def predict(
    lab: LabValues,
    bg: BackgroundTasks,
    explain: bool = Query(False, description="Include SHAP feature explanations"),
):
    """
    Predict diabetes risk from a single set of lab values.
    Add ?explain=true to get SHAP feature-level explanations (slightly slower).
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    X = _prep_features(lab)
    X_scaled = _scaler.transform(X)

    prob = float(_model.predict_proba(X_scaled)[0][1])
    pred = int(prob >= 0.5)

    response: Dict[str, Any] = {
        "prediction":   pred,
        "probability":  round(prob, 4),
        "risk_level":   _risk_label(prob),
        "model_version": "1.0.0",
        "timestamp":    datetime.utcnow().isoformat(),
    }

    # SHAP explanation — only computed if requested
    if explain:
        if _explainer is None:
            response["shap_note"] = "SHAP not available in this environment"
        else:
            shap_vals = _explainer.shap_values(X_scaled)
            vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            importance = dict(zip(X.columns, [round(float(v), 4) for v in vals]))
            top_factor = max(importance, key=lambda k: abs(importance[k]))
            response["shap"] = {
                "feature_contributions": importance,
                "top_risk_factor":       top_factor,
            }

    bg.add_task(_log_prediction, lab.dict(), pred, prob)
    return response


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    """Predict for multiple patients in one call."""
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    results = []
    for lab in req.records:
        X        = _prep_features(lab)
        X_scaled = _scaler.transform(X)
        prob     = float(_model.predict_proba(X_scaled)[0][1])
        results.append({
            "probability": round(prob, 4),
            "risk_level":  _risk_label(prob),
            "prediction":  int(prob >= 0.5),
        })

    return {
        "results":   results,
        "n":         len(results),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/monitoring/summary")
def monitoring_summary():
    """Basic stats on recent predictions — useful for spotting drift."""
    path = "monitoring/predictions.jsonl"
    if not os.path.exists(path):
        return {"message": "No predictions logged yet"}

    rows = [json.loads(l) for l in open(path)]
    probs = [r["probability"] for r in rows]
    preds = [r["prediction"]  for r in rows]

    return {
        "n_predictions":   len(rows),
        "positive_rate":   round(sum(preds) / len(preds), 4),
        "mean_probability": round(sum(probs) / len(probs), 4),
        "min_probability":  round(min(probs), 4),
        "max_probability":  round(max(probs), 4),
        "last_seen":        rows[-1]["ts"],
    }
# monitoring endpoint added 
