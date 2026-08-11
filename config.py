import os

# Project Name & Model Path
PROJECT_NAME = "CGM Medical AI Inference Engine"
MODEL_PATH = os.getenv("MODEL_PATH", "models/hybrid_cgm_brain_quantized.tflite")

# Clinical Glucose Thresholds (mg/dL)
HYPO_THRESH = 75.0
HYPER_THRESH = 180.0

# 3 Hours of telemetry sampled every 30 minutes = 6 sequence steps
SEQUENCE_LENGTH = 6       # [t-150m, t-120m, t-90m, t-60m, t-30m, t]
TEMPORAL_FEATURE_DIM = 7  # [CGM, IOB, basal, COB, time_sin, time_cos, cgm_velocity]
STATIC_FEATURE_DIM = 2    # [scaled_age, scaled_bmi]
