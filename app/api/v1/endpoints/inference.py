from fastapi import APIRouter, HTTPException, status
from app.schemas.cgm import (
    CGMWindowInput, 
    CGMPredictionResponse, 
    BatchCGMWindowInput, 
    BatchCGMPredictionResponse
)
from app.services.predictor import cgm_service
from app.core.config import settings

router = APIRouter(prefix="/inference", tags=["Inference"])

@router.post("/", response_model=CGMPredictionResponse, summary="Predict Next 30-Minute Diabetes Status & Danger Risk")
def get_cgm_prediction(payload: CGMWindowInput):
    """
    Accepts 3 hours of historical data (6 timesteps spaced at 30 minutes, 7 features each) 
    and 2 static patient features.
    
    Predicts:
    1. Danger probability for the upcoming 30-minute horizon.
    2. Estimated glucose level (mg/dL) for the next 30 minutes.
    3. Clinical status and actionable routing recommendations.
    """
    # Validate temporal sequence length (6 intervals of 30 minutes = 3 hours)
    if len(payload.temporal_features) != settings.SEQUENCE_LENGTH or any(
        len(step) != settings.TEMPORAL_FEATURE_DIM for step in payload.temporal_features
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Invalid shape for temporal_features. Must be exactly ({settings.SEQUENCE_LENGTH}, {settings.TEMPORAL_FEATURE_DIM}) representing 3 hours of 30-min interval features [CGM, IOB, basal, COB, time_sin, time_cos, cgm_velocity]."
        )
    
    # Validate static demographics shape
    if len(payload.static_features) != settings.STATIC_FEATURE_DIM:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Invalid shape for static_features. Must be exactly ({settings.STATIC_FEATURE_DIM},) representing [scaled_age, scaled_bmi]."
        )

    try:
        res = cgm_service.predict(payload.temporal_features, payload.static_features)
        prob = res["danger_probability"]
        pred_cgm = res["predicted_cgm_next_30m"]
        engine_used = res["engine"]

        # Clinical status classification based on probability and thresholds
        if prob > 0.65 or pred_cgm < 70.0 or pred_cgm > 250.0:
            status_label = "DANGER"
            recommendation = "CRITICAL RISK DETECTED: Impending hypoglycemia/hyperglycemia expected within 30 mins. Review IOB and check glucose level."
        elif prob > 0.40 or pred_cgm < settings.HYPO_THRESH or pred_cgm > settings.HYPER_THRESH:
            status_label = "EVALUATE"
            recommendation = "MODERATE RISK: Volatility detected. Monitor glucose trajectory closely over the next 30 minutes."
        else:
            status_label = "SAFE"
            recommendation = "STABLE: Glucose level and 30-minute predicted trajectory remain within safe target boundaries."

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
            detail=f"Inference execution failed: {str(e)}"
        )

@router.post("/batch", response_model=BatchCGMPredictionResponse, summary="Batch Predict Next 30-Minute Status for Multiple Windows")
def get_batch_cgm_prediction(payload: BatchCGMWindowInput):
    """
    Batch endpoint for analyzing multiple patient windows concurrently.
    """
    results = []
    for sample in payload.samples:
        results.append(get_cgm_prediction(sample))
    return BatchCGMPredictionResponse(predictions=results)
