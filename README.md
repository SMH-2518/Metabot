# Simplified CGM Medical AI FastAPI Backend

A clean, beginner-friendly FastAPI backend for Continuous Glucose Monitoring (CGM) diabetes trajectory prediction and danger risk assessment.

## 📂 Simplified File Layout

```
Metabot/
│
├── main.py                  # Single, clean backend script containing all routes & logic
├── models/
│   └── hybrid_cgm_brain_quantized.tflite  # Quantized TFLite model binary
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

---

## ⚡ How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI server**:
   ```bash
   python -m uvicorn main:app --reload
   ```

3. **Open Interactive Docs**:
   - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - ReDoc UI: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📊 Data Input Format (3 Hours of Data -> Next 30m Prediction)

- **`temporal_features`**: Shape `(6, 7)` representing 6 sequence steps spaced 30 minutes apart (total 3 hours of telemetry). Each step has 7 features:
  `[CGM, IOB, basal, COB, time_sin, time_cos, cgm_velocity]`
- **`static_features`**: Shape `(2,)` representing `[scaled_age, scaled_bmi]`.

### Example JSON Request (`POST /predict`):
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

### Example Response:
```json
{
  "danger_probability": 0.7158,
  "predicted_status": "DANGER",
  "predicted_cgm_next_30m": 265.9,
  "routing_recommendation": "CRITICAL RISK: Impending hypo/hyperglycemia detected within 30 mins. Check glucose & IOB immediately.",
  "inference_engine": "TFLite Model Engine (Quantized)"
}
```
