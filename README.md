# CGM Medical AI Inference Engine

A production-grade, high-performance FastAPI backend for Continuous Glucose Monitoring (CGM) diabetes trajectory prediction and danger risk assessment.

## 🏗️ Project Architecture & Modular Layout

```
cgm_inference_app/
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # Entry point: FastAPI app & router registration
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py            # Global paths, thresholds, and window configuration
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           └── inference.py # REST endpoints for CGM predictions & batch processing
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── cgm.py               # Pydantic data validation models
│   └── services/
│       ├── __init__.py
│       └── predictor.py         # TFLite model singleton & dynamic inference engine
│
├── models/
│   └── hybrid_cgm_brain_quantized.tflite  # Quantized TFLite prediction model
├── requirements.txt
└── README.md
```

---

## 📊 Temporal Feature Pipeline & 30-Minute Window Model

The backend is tailored for models transfer-learned on long-term patient telemetry (e.g., 2 months of CGM data) and deployed for real-time inference on short-term window sequences:

1. **Temporal History Window (3 Hours)**:
   - **Sequence Length**: 6 temporal time steps (taken at 30-minute intervals: $t-150m, t-120m, t-90m, t-60m, t-30m, t$).
   - **Feature Dimension**: 7 features per step:
     1. `CGM` (Continuous Glucose Reading in mg/dL)
     2. `IOB` (Insulin On Board)
     3. `basal` (Basal rate)
     4. `COB` (Carbohydrates On Board)
     5. `time_sin` (Cyclic time feature sin component)
     6. `time_cos` (Cyclic time feature cos component)
     7. `cgm_velocity` (Glucose rate of change in mg/dL/min)
   - **Temporal Tensor Shape**: `(1, 6, 7)`

2. **Static Demographics**:
   - **Feature Dimension**: 2 static features `[scaled_age, scaled_bmi]`
   - **Static Tensor Shape**: `(1, 2)`

3. **Prediction Horizon (Next 30 Minutes)**:
   - Outputs danger risk probability ($0.0 \rightarrow 1.0$).
   - Predicts next 30-minute estimated glucose level ($mg/dL$).
   - Recommends clinical action (`SAFE`, `EVALUATE`, `DANGER`).

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run local Dev Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Interactive API Documentation

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc UI**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🧪 Example REST Request

### Endpoint: `POST /api/v1/inference/`

#### Request Payload:
```json
{
  "temporal_features": [
    [140.0, 1.2, 0.8, 15.0, 0.500, 0.866, 0.20],
    [142.0, 1.0, 0.8, 10.0, 0.707, 0.707, 0.67],
    [145.0, 0.8, 0.8,  5.0, 0.866, 0.500, 1.00],
    [150.0, 0.6, 0.8,  2.0, 0.966, 0.259, 1.67],
    [158.0, 0.4, 0.8,  0.0, 1.000, 0.000, 2.67],
    [168.0, 0.2, 0.8,  0.0, 0.966,-0.259, 3.33]
  ],
  "static_features": [0.45, 0.62]
}
```

#### Response:
```json
{
  "danger_probability": 0.1245,
  "predicted_status": "SAFE",
  "predicted_cgm_next_30m": 178.0,
  "routing_recommendation": "STABLE: Glucose level and 30-minute predicted trajectory remain within safe target boundaries.",
  "inference_engine": "TFLite Model Engine (Quantized)"
}
```
