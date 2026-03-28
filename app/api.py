"""
api.py
------
FastAPI application exposing a /predict endpoint for real-time credit scoring.

Run locally
-----------
    uvicorn app.api:app --reload

Endpoints
---------
POST /predict   – accepts applicant features, returns PD estimate + risk band
GET  /health    – liveness check
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Credit Risk API", version="0.1.0")


class ApplicantFeatures(BaseModel):
    # TODO: define input fields matching the feature set
    pass


class PredictionResponse(BaseModel):
    probability_of_default: float
    risk_band: str  # e.g. "LOW" | "MEDIUM" | "HIGH"
    gini_at_training: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ApplicantFeatures) -> PredictionResponse:
    # TODO: load model, run feature pipeline, return score
    raise NotImplementedError
