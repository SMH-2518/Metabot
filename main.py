from typing import List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import config
from model import cgm_service

# Initialize FastAPI App
app = FastAPI(
    title=config.PROJECT_NAME,
    version="1.0.0",
    description="Simple, high-performance FastAPI server predicting 30-minute CGM diabetes trajectories.",
    docs_url="/docs",
    redoc_url="/redoc"
)


# --- Input & Output Data Validation Models ---
class CGMWindowInput(BaseModel):
    temporal_features: List[List[float]] = Field(
        ..., 
        description="6 sequence steps x 7 features representing 3 hours of historical telemetry."
    )
    static_features: List[float] = Field(
        ..., 
        description="2 static features [scaled_age, scaled_bmi]."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "temporal_features": [
                    [140.0, 1.2, 0.8, 15.0, 0.500, 0.866, 0.20],  # t - 150m
                    [142.0, 1.0, 0.8, 10.0, 0.707, 0.707, 0.67],  # t - 120m
                    [145.0, 0.8, 0.8,  5.0, 0.866, 0.500, 1.00],  # t - 90m
                    [150.0, 0.6, 0.8,  2.0, 0.966, 0.259, 1.67],  # t - 60m
                    [158.0, 0.4, 0.8,  0.0, 1.000, 0.000, 2.67],  # t - 30m
                    [168.0, 0.2, 0.8,  0.0, 0.966,-0.259, 3.33]   # t (current)
                ],
                "static_features": [0.45, 0.62]
            }
        }
    }

class CGMPredictionResponse(BaseModel):
    danger_probability: float
    predicted_status: str
    predicted_cgm_next_30m: float
    routing_recommendation: str
    inference_engine: str

class BatchCGMWindowInput(BaseModel):
    samples: List[CGMWindowInput]

class BatchCGMPredictionResponse(BaseModel):
    predictions: List[CGMPredictionResponse]


# --- Health Endpoints ---
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online", 
        "service": config.PROJECT_NAME, 
        "docs": "/docs", 
        "engine": cgm_service.engine_name
    }

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "model_engine": cgm_service.engine_name,
        "sequence_length": config.SEQUENCE_LENGTH,
        "temporal_dim": config.TEMPORAL_FEATURE_DIM,
        "static_dim": config.STATIC_FEATURE_DIM
    }


# --- Prediction Endpoints ---
@app.post("/predict", response_model=CGMPredictionResponse, tags=["Inference"])
@app.post("/api/v1/inference/", response_model=CGMPredictionResponse, tags=["Inference"])
def predict(payload: CGMWindowInput):
    """Predicts diabetes status & danger risk for the next 30 minutes."""
    # 1. Validate temporal sequence length (6 intervals of 30 minutes = 3 hours)
    if len(payload.temporal_features) != config.SEQUENCE_LENGTH or any(
        len(step) != config.TEMPORAL_FEATURE_DIM for step in payload.temporal_features
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"temporal_features must be shape ({config.SEQUENCE_LENGTH}, {config.TEMPORAL_FEATURE_DIM})."
        )

    # 2. Validate static demographics shape
    if len(payload.static_features) != config.STATIC_FEATURE_DIM:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"static_features must be shape ({config.STATIC_FEATURE_DIM},)."
        )

    try:
        # 3. Execute inference
        result = cgm_service.predict(payload.temporal_features, payload.static_features)
        prob = result["danger_probability"]
        pred_cgm = result["predicted_cgm_next_30m"]

        # 4. Clinical Status Classification
        if prob > 0.65 or pred_cgm < 70.0 or pred_cgm > 250.0:
            status_label = "DANGER"
            recommendation = "CRITICAL RISK: Impending hypo/hyperglycemia within 30 mins. Check glucose & IOB."
        elif prob > 0.40 or pred_cgm < config.HYPO_THRESH or pred_cgm > config.HYPER_THRESH:
            status_label = "EVALUATE"
            recommendation = "MODERATE RISK: Volatility detected. Monitor glucose trajectory."
        else:
            status_label = "SAFE"
            recommendation = "STABLE: Glucose level and 30-minute predicted trajectory remain within safe target boundaries."

        return CGMPredictionResponse(
            danger_probability=prob,
            predicted_status=status_label,
            predicted_cgm_next_30m=pred_cgm,
            routing_recommendation=recommendation,
            inference_engine=result["engine"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchCGMPredictionResponse, tags=["Inference"])
def predict_batch(payload: BatchCGMWindowInput):
    """Batch inference endpoint."""
    return BatchCGMPredictionResponse(predictions=[predict(sample) for sample in payload.samples])
