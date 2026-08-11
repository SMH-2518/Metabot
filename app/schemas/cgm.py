from pydantic import BaseModel, Field
from typing import List, Optional

class CGMWindowInput(BaseModel):
    # Expects a 6-step temporal sequence of 7 features:
    # [CGM, IOB, basal, COB, time_sin, time_cos, cgm_velocity]
    # representing 3 hours of historical data (6 intervals of 30 minutes)
    temporal_features: List[List[float]] = Field(
        ..., 
        description="Nested list of shape (6, 7) representing 3 hours of 30-minute diabetes feature history: [CGM, IOB, basal, COB, time_sin, time_cos, cgm_velocity]."
    )
    static_features: List[float] = Field(
        ..., 
        description="List of shape (2,) representing patient static demographics [scaled_age, scaled_bmi]."
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
    danger_probability: float = Field(..., description="Predicted danger probability for the next 30-minute window (0.0 to 1.0).")
    predicted_status: str = Field(..., description="High-level clinical status: SAFE, EVALUATE, or DANGER.")
    predicted_cgm_next_30m: float = Field(..., description="Predicted continuous glucose level (mg/dL) for the next 30 minutes.")
    routing_recommendation: str = Field(..., description="Clinical action recommendation based on predicted trajectory.")
    inference_engine: str = Field(default="TFLite Model Singleton", description="Engine used for inference.")

class BatchCGMWindowInput(BaseModel):
    samples: List[CGMWindowInput] = Field(..., description="List of multi-patient temporal window inputs.")

class BatchCGMPredictionResponse(BaseModel):
    predictions: List[CGMPredictionResponse] = Field(..., description="List of prediction outputs for each sample window.")
