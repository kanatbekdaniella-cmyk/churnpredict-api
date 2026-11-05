import os
from pathlib import Path
from functools import lru_cache
from typing import Any
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="ChurnPredict AI - Real TensorFlow Inference")
origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "https://fascinating-lebkuchen-4ff629.netlify.app").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

@lru_cache(maxsize=1)
def artifacts():
    # Only load trusted pickle files: joblib/pickle can execute arbitrary code.
    preprocessor = joblib.load(ROOT / "preprocessor.pkl")
    model = tf.keras.models.load_model(ROOT / "churn_model.keras", compile=False)
    return preprocessor, model

class PredictionRequest(BaseModel):
    customer_features: dict[str, Any] | None = None
    ping: bool = False
    check: str | None = None

@app.get("/health")
def health():
    try:
        pre, model = artifacts()
        return {"status": "ok", "input_features": int(model.input_shape[-1]), "preprocessor": type(pre).__name__}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model initialization failed: {type(exc).__name__}")

@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        pre, model = artifacts()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model initialization failed: {type(exc).__name__}")
    if payload.ping:
        return {"status": "ok", "input_features": int(model.input_shape[-1])}
    if not payload.customer_features:
        raise HTTPException(status_code=422, detail="customer_features is required")
    try:
        raw = pd.DataFrame([payload.customer_features])
        expected = list(getattr(pre, 'feature_names_in_', []))
        if expected:
            missing = [x for x in expected if x not in raw.columns]
            if missing:
                raise ValueError(f"Missing input columns: {missing}")
            raw = raw[expected]
        for field in ('tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'):
            if field in raw:
                raw[field] = pd.to_numeric(raw[field], errors='raise')
        transformed = pre.transform(raw)
        if hasattr(transformed, 'toarray'):
            transformed = transformed.toarray()
        matrix = np.asarray(transformed, dtype=np.float32)
        required = int(model.input_shape[-1])
        if matrix.shape != (1, required):
            raise ValueError(f"Preprocessor produced {matrix.shape}, model expects (1, {required})")
        probability = float(model.predict(matrix, verbose=0).reshape(-1)[0])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid model probability')
        return {"probability": probability, "model_version": "Uploaded TensorFlow Keras model", "features_used": required}
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
