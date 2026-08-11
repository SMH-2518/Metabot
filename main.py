"""
===============================================================================
CGM Diabetes Medical AI - Simplified FastAPI Backend
===============================================================================
This single-file backend handles 3 hours of historical diabetes data (6 intervals 
of 30 minutes) and predicts the next 30-minute diabetes status & danger risk.

Run locally with:
  python -m uvicorn main:app --reload
===============================================================================
"""

import os
import numpy as np
import tensorflow as tf
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================
PROJECT_NAME = "CGM Medical AI Inference Engine"
MODEL_PATH = os.getenv("MODEL_PATH", "models/hybrid_cgm_brain_quantized.tflite")

# Glucose Risk Thresholds (mg/dL)
HYPO_THRESH = 75.0   # Hypoglycemia boundary
HYPER_THRESH = 180.0  # Hyperglycemia boundary

# Window dimensions: 3 hours of data sampled every 30 minutes = 6 steps
SEQUENCE_LENGTH = 6       # 6 timesteps (t-150m, t-120m, t-90m, t-60m, t-30m, t)
TEMPORAL_FEATURE_DIM = 7  # [CGM, IOB, basal, COB, time_sin, time_cos, cgm_velocity]
STATIC_FEATURE_DIM = 2    # [scaled_age, scaled_bmi]


# =============================================================================
# 2. PYDANTIC DATA MODELS (Input Validation & API Responses)
# =============================================================================
class CGMWindowInput(BaseModel):
    """
    Input schema sent by the client.
    Expects 3 hours of historical data (6 steps x 7 features) + 2 static features.
    """
    temporal_features: List[List[float]] = Field(
        ..., 
        description="6-step temporal sequence of 7 features: [CGM, IOB, basal, COB, time_sin, time_cos, cgm_velocity]."
    )
    static_features: List[float] = Field(
        ..., 
        description="List of shape (2,) representing [scaled_age, scaled_bmi]."
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
    """
    Response schema returned to the client.
    """
    danger_probability: float = Field(..., description="Predicted danger probability for next 30 mins (0.0 to 1.0).")
    predicted_status: str = Field(..., description="Clinical risk status: SAFE, EVALUATE, or DANGER.")
    predicted_cgm_next_30m: float = Field(..., description="Predicted glucose reading (mg/dL) for the next 30 minutes.")
    routing_recommendation: str = Field(..., description="Actionable recommendation.")
    inference_engine: str = Field(..., description="Active inference engine name.")

class BatchCGMWindowInput(BaseModel):
    samples: List[CGMWindowInput]

class BatchCGMPredictionResponse(BaseModel):
    predictions: List[CGMPredictionResponse]


# =============================================================================
# 3. MODEL INFERENCE SERVICE (Singleton Loader)
# =============================================================================
class CGMModelService:
    def __init__(self):
        self.interpreter = None
        self.is_tflite = False
        self.temp_input_index = None
        self.stat_input_index = None
        self.output_index = None
        self.engine_name = "Fallback Dynamic Predictor"
        self.load_model()

    def load_model(self):
        """Loads the TFLite model at application startup."""
        print(f"[ModelService] Loading model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"[ModelService] Warning: Model file '{MODEL_PATH}' not found. Using fallback predictor.")
            return

        try:
            self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            
            # Map input tensors by shape ([1, 6, 7] and [1, 2])
            for detail in self.interpreter.get_input_details():
                shape = list(detail['shape'])
                if shape == [1, SEQUENCE_LENGTH, TEMPORAL_FEATURE_DIM]:
                    self.temp_input_index = detail['index']
                elif shape == [1, STATIC_FEATURE_DIM]:
                    self.stat_input_index = detail['index']

            if self.temp_input_index is None:
                self.temp_input_index = self.interpreter.get_input_details()[0]['index']
            if self.stat_input_index is None and len(self.interpreter.get_input_details()) > 1:
                self.stat_input_index = self.interpreter.get_input_details()[1]['index']

            self.output_index = self.interpreter.get_output_details()[0]['index']
            self.is_tflite = True
            self.engine_name = "TFLite Model Engine (Quantized)"
            print("[ModelService] TFLite model successfully loaded into memory!")
        except Exception as e:
            print(f"[ModelService] Note: TFLite allocation notice: {e}")
            print("[ModelService] Running in dynamic trajectory fallback mode.")

    def _fallback_predict(self, temporal_data: list, static_data: list) -> tuple:
        """Calculates trajectory & risk fallback if TFLite hardware delegate is unavailable."""
        temp_arr = np.array(temporal_data, dtype=np.float32)
        cgm_series = temp_arr[:, 0]
        iob_curr = temp_arr[-1, 1]
        cob_curr = temp_arr[-1, 3]
        velocity = temp_arr[-1, 6]
        cgm_curr = cgm_series[-1]

        # Predict next 30m glucose = current + (velocity * 30m) + carbs - insulin
        pred_cgm = max(30.0, min(500.0, float(cgm_curr + (velocity * 30.0) + (cob_curr * 2.0) - (iob_curr * 10.0))))
        
        # Risk assessment
        hypo_risk = max(0.0, (HYPO_THRESH - pred_cgm) / 35.0) if pred_cgm < HYPO_THRESH else 0.0
        hyper_risk = max(0.0, (pred_cgm - HYPER_THRESH) / 120.0) if pred_cgm > HYPER_THRESH else 0.0
        vel_risk = min(0.9, abs(velocity) / 4.0) if velocity < -1.5 else 0.0
        
        prob = float(np.clip(max(hypo_risk, hyper_risk, vel_risk), 0.01, 0.99))
        return prob, round(pred_cgm, 2)

    def predict(self, temporal_data: list, static_data: list) -> dict:
        """Runs prediction on 3-hour window data and static patient features."""
        X_temp = np.array(temporal_data, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, TEMPORAL_FEATURE_DIM)
        X_stat = np.array(static_data, dtype=np.float32).reshape(1, STATIC_FEATURE_DIM)

        if self.is_tflite and self.interpreter is not None:
            try:
                self.interpreter.set_tensor(self.temp_input_index, X_temp)
                self.interpreter.set_tensor(self.stat_input_index, X_stat)
                self.interpreter.invoke()
                
                raw_prob = float(self.interpreter.get_tensor(self.output_index)[0][0])
                cgm_curr = float(X_temp[0, -1, 0])
                vel_curr = float(X_temp[0, -1, 6])
                pred_cgm = cgm_curr + (vel_curr * 30.0)
                
                return {
                    "danger_probability": float(np.clip(raw_prob, 0.0, 1.0)),
                    "predicted_cgm_next_30m": round(pred_cgm, 2),
                    "engine": self.engine_name
                }
            except Exception as e:
                print(f"[ModelService] TFLite invoke note: {e}. Using fallback.")

        prob, pred_cgm = self._fallback_predict(temporal_data, static_data)
        return {
            "danger_probability": prob,
            "predicted_cgm_next_30m": pred_cgm,
            "engine": self.engine_name
        }

# Instantiate Global Service
cgm_service = CGMModelService()


# =============================================================================
# 4. FASTAPI APPLICATION & API ENDPOINTS
# =============================================================================
app = FastAPI(
    title=PROJECT_NAME,
    version="1.0.0",
    description="Simple, high-performance FastAPI server predicting 30-minute CGM diabetes trajectories.",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/", tags=["Health Check"])
def root():
    """Root endpoint to check server status."""
    return {
        "status": "online",
        "service": PROJECT_NAME,
        "docs": "/docs",
        "model_engine": cgm_service.engine_name
    }


@app.get("/health", tags=["Health Check"])
def health():
    """Detailed health check for parameters and model status."""
    return {
        "status": "healthy",
        "model_path": MODEL_PATH,
        "model_engine": cgm_service.engine_name,
        "sequence_length": SEQUENCE_LENGTH,
        "temporal_feature_dim": TEMPORAL_FEATURE_DIM,
        "static_feature_dim": STATIC_FEATURE_DIM
    }


@app.post("/predict", response_model=CGMPredictionResponse, tags=["Inference"])
@app.post("/api/v1/inference/", response_model=CGMPredictionResponse, tags=["Inference"])
def predict_cgm(payload: CGMWindowInput):
    """
    Predicts diabetes status & danger risk for the next 30 minutes 
    based on 3 hours of historical data (6 steps x 7 features) and 2 static features.
    """
    # 1. Validate temporal window shape
    if len(payload.temporal_features) != SEQUENCE_LENGTH or any(
        len(step) != TEMPORAL_FEATURE_DIM for step in payload.temporal_features
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid shape for temporal_features. Must be exactly ({SEQUENCE_LENGTH}, {TEMPORAL_FEATURE_DIM})."
        )

    # 2. Validate static demographics shape
    if len(payload.static_features) != STATIC_FEATURE_DIM:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid shape for static_features. Must be exactly ({STATIC_FEATURE_DIM},)."
        )

    try:
        # 3. Run Inference
        result = cgm_service.predict(payload.temporal_features, payload.static_features)
        prob = result["danger_probability"]
        pred_cgm = result["predicted_cgm_next_30m"]
        engine_used = result["engine"]

        # 4. Clinical Status Classification
        if prob > 0.65 or pred_cgm < 70.0 or pred_cgm > 250.0:
            status_label = "DANGER"
            recommendation = "CRITICAL RISK: Impending hypo/hyperglycemia detected within 30 mins. Check glucose & IOB immediately."
        elif prob > 0.40 or pred_cgm < HYPO_THRESH or pred_cgm > HYPER_THRESH:
            status_label = "EVALUATE"
            recommendation = "MODERATE RISK: Volatility detected. Monitor glucose trajectory closely."
        else:
            status_label = "SAFE"
            recommendation = "STABLE: Glucose level and 30-minute predicted trajectory are within target ranges."

        return CGMPredictionResponse(
            danger_probability=round(prob, 4),
            predicted_status=status_label,
            predicted_cgm_next_30m=pred_cgm,
            routing_recommendation=recommendation,
            inference_engine=engine_used
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchCGMPredictionResponse, tags=["Inference"])
def predict_batch(payload: BatchCGMWindowInput):
    """Processes batch inference for multiple patient windows."""
    results = [predict_cgm(sample) for sample in payload.samples]
    return BatchCGMPredictionResponse(predictions=results)
