import os

class Settings:
    PROJECT_NAME: str = "CGM Medical AI Inference Engine"
    API_V1_STR: str = "/api/v1"
    
    # Path to compiled TFLite or Keras model
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH", 
        "models/hybrid_cgm_brain_quantized.tflite"
    )
    
    # Clinical Glucose Thresholds (mg/dL)
    HYPO_THRESH: float = 75.0
    HYPER_THRESH: float = 180.0
    
    # Window sequence definition: 3 hours of datasampled every 30 minutes = 6 temporal windows
    SEQUENCE_LENGTH: int = 6
    TEMPORAL_FEATURE_DIM: int = 7  # [CGM, IOB, basal, COB, time_sin, time_cos, cgm_velocity]
    STATIC_FEATURE_DIM: int = 2    # [scaled_age, scaled_bmi]

settings = Settings()
