import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from app.schemas import CustomerData, PredictionResponse
from src.feature_engineering import engineer_features

app = FastAPI(
    title="BankWise Term Deposit Propensity API",
    description="Production REST API for predicting customer term deposit subscription propensity.",
    version="1.0.0"
)

# Load Champion Model Artifact
MODEL_PATH = "models/champion_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

OPTIMAL_THRESHOLD = 0.7687

@app.get("/", tags=["Health Check"])
def root():
    return {"status": "online", "message": "BankWise Propensity Engine is active."}

@app.get("/health", tags=["Health Check"])
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact missing.")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict_propensity(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized.")

    try:
        # Convert input Pydantic model to DataFrame
        raw_dict = customer.dict()
        df_raw = pd.DataFrame([raw_dict])

        # Apply production feature engineering pipeline
        df_engineered = engineer_features(df_raw)

        # Align columns with model training schema
        expected_cols = getattr(model, "feature_names_in_", None)
        if expected_cols is not None:
            for col in expected_cols:
                if col not in df_engineered.columns:
                    df_engineered[col] = 0
            df_engineered = df_engineered[expected_cols]

        # Score Probability
        prob = float(model.predict_proba(df_engineered)[:, 1][0])
        pred_class = 1 if prob >= OPTIMAL_THRESHOLD else 0

        action = (
            "HIGH PROPENSITY: Route to call center agent for outreach."
            if pred_class == 1
            else "LOW PROPENSITY: Skip outbound call to minimize campaign expenses."
        )

        return PredictionResponse(
            conversion_probability=round(prob, 4),
            predicted_class=pred_class,
            recommendation=action,
            optimal_threshold=OPTIMAL_THRESHOLD
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")