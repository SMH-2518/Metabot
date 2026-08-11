from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.endpoints import inference
from app.services.predictor import cgm_service

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description=(
        "Production-grade FastAPI inference server for Continuous Glucose Monitoring (CGM) AI models. "
        "Processes 3-hour temporal sequences (6 steps of 30-min intervals with 7 features) "
        "and predicts next 30-minute glucose trajectories and danger risk probabilities."
    ),
    docs_url="/docs",
    redoc_url="/redoc"
)

# Register versioned API routers
app.include_router(inference.router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Health Check"])
def root_health_check():
    """
    Root health check endpoint verifying server operational status.
    """
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "model_engine": cgm_service.engine_name
    }

@app.get("/health", tags=["Health Check"])
def detailed_health_check():
    """
    Detailed health check inspecting model singleton status and configuration parameters.
    """
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "model_path": settings.MODEL_PATH,
        "model_engine": cgm_service.engine_name,
        "is_tflite": cgm_service.is_tflite,
        "sequence_length": settings.SEQUENCE_LENGTH,
        "temporal_feature_dim": settings.TEMPORAL_FEATURE_DIM,
        "static_feature_dim": settings.STATIC_FEATURE_DIM,
        "thresholds": {
            "hypo_mg_dl": settings.HYPO_THRESH,
            "hyper_mg_dl": settings.HYPER_THRESH
        }
    }
