import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import OrderRequest, PredictionResponse
from src.prediction.predictor import predict

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "classification_threshold.joblib"

# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    model_metadata = json.load(f)


MODEL_VERSION = model_metadata["model_version"]

classification_threshold = joblib.load(THRESHOLD_PATH)

# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Olist Delivery Prediction API",
    description="API for predicting whether an Olist order will be delivered late.",
    version=MODEL_VERSION,
)

Instrumentator().instrument(app).expose(app)

# --------------------------------------------------
# Health check
# --------------------------------------------------


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# --------------------------------------------------
# Model information
# --------------------------------------------------


@app.get("/model")
def model_info():
    return model_metadata


# --------------------------------------------------
# Model version
# --------------------------------------------------


@app.get("/model/version")
def model_version():
    return {"model_version": MODEL_VERSION}


# =======================================================


@app.post("/predict", response_model=PredictionResponse)
def predict_order(order: OrderRequest):
    result = predict(
        # input_data=order.model_dump(),
        input_data=pd.DataFrame([order.model_dump()]),
        model=model,
        preprocessor=preprocessor,
        model_version=MODEL_VERSION,
        classification_threshold=classification_threshold,
    )

    return {
        "prediction": result["prediction"],
        "probability": result["probability"],
        "model_version": MODEL_VERSION,
    }
