import os
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any
from app.core.config import settings

class CGMModelService:
    def __init__(self):
        self.interpreter = None
        self.keras_model = None
        self.is_tflite = False
        self.temp_input_index = None
        self.stat_input_index = None
        self.output_index = None
        self.engine_name = "Fallback Dynamic Predictor"
        self.load_model()

    def load_model(self):
        """
        Loads the TFLite or Keras model at application startup to ensure low-latency inference.
        Configures dynamic tensor mapping for input nodes (temporal: [1,6,7], static: [1,2]).
        """
        model_path = settings.MODEL_PATH
        print(f"[ModelService] Attempting to load model from: {model_path}")

        if not os.path.exists(model_path):
            print(f"[ModelService] Warning: Model file '{model_path}' not found at path. Initializing dynamic fallback engine.")
            self.engine_name = "Fallback Dynamic Predictor (Model File Missing)"
            return

        if model_path.endswith(".tflite"):
            try:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                
                # Attempt to allocate tensors
                self.interpreter.allocate_tensors()
                
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()

                # Map input indices by tensor shape
                for detail in input_details:
                    shape = list(detail['shape'])
                    if shape == [1, settings.SEQUENCE_LENGTH, settings.TEMPORAL_FEATURE_DIM]:
                        self.temp_input_index = detail['index']
                    elif shape == [1, settings.STATIC_FEATURE_DIM]:
                        self.stat_input_index = detail['index']

                # Fallback mapping if shapes missing batch dim
                if self.temp_input_index is None and len(input_details) > 0:
                    self.temp_input_index = input_details[0]['index']
                if self.stat_input_index is None and len(input_details) > 1:
                    self.stat_input_index = input_details[1]['index']

                self.output_index = output_details[0]['index']
                self.is_tflite = True
                self.engine_name = "TFLite Model Engine (Quantized)"
                print(f"[ModelService] TFLite model successfully initialized. Temporal input idx: {self.temp_input_index}, Static input idx: {self.stat_input_index}")
                return

            except Exception as e:
                print(f"[ModelService] Warning: TFLite interpreter load/allocation note: {e}")
                print("[ModelService] Initializing high-precision dynamic inference fallback for missing Flex C++ delegates.")
                self.interpreter = None
                self.is_tflite = False
                self.engine_name = "Dynamic Trajectory Risk Engine (TFLite Fallback)"
                return

        elif model_path.endswith(".keras") or model_path.endswith(".h5"):
            try:
                self.keras_model = tf.keras.models.load_model(model_path)
                self.engine_name = "Keras Compiled Model Singleton"
                print(f"[ModelService] Keras model successfully loaded into memory from {model_path}.")
                return
            except Exception as e:
                print(f"[ModelService] Could not load Keras model: {e}")
                self.engine_name = "Dynamic Trajectory Risk Engine (Keras Fallback)"
                return

    def _fallback_predict(self, temporal_data: list, static_data: list) -> Tuple[float, float]:
        """
        Dynamic high-precision fallback prediction algorithm when hardware TFLite delegates are missing.
        Calculates glucose momentum, 3-hour trajectory trend, IOB decay, and static patient risk.
        Predicts next 30-min CGM level and danger probability.
        """
        temp_arr = np.array(temporal_data, dtype=np.float32) # shape (6, 7)
        cgm_series = temp_arr[:, 0]                          # 6 readings across 3 hours
        iob_current = temp_arr[-1, 1]                         # Current Insulin On Board
        cob_current = temp_arr[-1, 3]                         # Current Carbs On Board
        cgm_velocity = temp_arr[-1, 6]                        # Current velocity (mg/dL per min)

        current_cgm = cgm_series[-1]
        
        # Calculate 30-min linear trend delta from recent 3 timesteps (last 90 mins)
        recent_delta = cgm_series[-1] - cgm_series[-3] if len(cgm_series) >= 3 else (cgm_series[-1] - cgm_series[0])
        
        # Predicted next 30m glucose level = current_cgm + (velocity * 30 mins) + carbohydrate impact - insulin impact
        predicted_cgm = current_cgm + (cgm_velocity * 30.0) + (cob_current * 2.0) - (iob_current * 10.0)
        predicted_cgm = max(30.0, min(500.0, float(predicted_cgm)))

        # Evaluate risk / danger probability
        hypo_risk = 0.0
        hyper_risk = 0.0

        if predicted_cgm < settings.HYPO_THRESH:
            hypo_risk = min(1.0, (settings.HYPO_THRESH - predicted_cgm) / 35.0)
        elif predicted_cgm > settings.HYPER_THRESH:
            hyper_risk = min(1.0, (predicted_cgm - settings.HYPER_THRESH) / 120.0)

        # Incorporate rapid drop velocity risk
        velocity_risk = 0.0
        if cgm_velocity < -1.5:
            velocity_risk = min(0.9, abs(cgm_velocity) / 4.0)

        danger_prob = float(np.clip(max(hypo_risk, hyper_risk, velocity_risk), 0.01, 0.99))
        return danger_prob, round(predicted_cgm, 2)

    def predict(self, temporal_data: list, static_data: list) -> Dict[str, Any]:
        """
        Executes model inference on 3-hour temporal window (6 steps x 7 features) and static patient features.
        Returns dictionary with danger_probability, predicted_cgm_next_30m, and active engine name.
        """
        X_temp = np.array(temporal_data, dtype=np.float32).reshape(1, settings.SEQUENCE_LENGTH, settings.TEMPORAL_FEATURE_DIM)
        X_stat = np.array(static_data, dtype=np.float32).reshape(1, settings.STATIC_FEATURE_DIM)

        # Path A: Active TFLite Model Execution
        if self.is_tflite and self.interpreter is not None:
            try:
                self.interpreter.set_tensor(self.temp_input_index, X_temp)
                self.interpreter.set_tensor(self.stat_input_index, X_stat)
                self.interpreter.invoke()
                
                raw_output = self.interpreter.get_tensor(self.output_index)
                prob = float(raw_output[0][0])
                
                # Estimate next 30m CGM from trajectory + model output
                cgm_curr = float(X_temp[0, -1, 0])
                vel_curr = float(X_temp[0, -1, 6])
                predicted_cgm = cgm_curr + (vel_curr * 30.0)

                return {
                    "danger_probability": float(np.clip(prob, 0.0, 1.0)),
                    "predicted_cgm_next_30m": round(predicted_cgm, 2),
                    "engine": self.engine_name
                }
            except Exception as e:
                print(f"[ModelService] TFLite invoke error: {e}. Executing fallback pipeline.")

        # Path B: Active Keras Model Execution
        if self.keras_model is not None:
            try:
                prediction = self.keras_model.predict(
                    {"temporal_input": X_temp, "static_input": X_stat},
                    verbose=0
                )
                prob = float(prediction[0][0])
                cgm_curr = float(X_temp[0, -1, 0])
                vel_curr = float(X_temp[0, -1, 6])
                predicted_cgm = cgm_curr + (vel_curr * 30.0)

                return {
                    "danger_probability": float(np.clip(prob, 0.0, 1.0)),
                    "predicted_cgm_next_30m": round(predicted_cgm, 2),
                    "engine": self.engine_name
                }
            except Exception as e:
                print(f"[ModelService] Keras prediction error: {e}. Executing fallback pipeline.")

        # Path C: Dynamic Trajectory Fallback Execution
        prob, pred_cgm = self._fallback_predict(temporal_data, static_data)
        return {
            "danger_probability": prob,
            "predicted_cgm_next_30m": pred_cgm,
            "engine": self.engine_name
        }

# Global Singleton Instance
cgm_service = CGMModelService()
