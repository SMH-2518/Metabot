import os
import numpy as np
import tensorflow as tf
import config

class CGMModelService:
    def __init__(self):
        self.interpreter = None
        self.is_tflite = False
        self.temp_idx = 0
        self.stat_idx = 1
        self.out_idx = 0
        self.engine_name = "Trajectory Fallback Engine"
        self.load_model()

    def load_model(self):
        """Loads the TFLite model at application startup."""
        if not os.path.exists(config.MODEL_PATH):
            print(f"[ModelService] File '{config.MODEL_PATH}' not found. Running in fallback mode.")
            return

        try:
            self.interpreter = tf.lite.Interpreter(model_path=config.MODEL_PATH)
            self.interpreter.allocate_tensors()

            # Find input tensor indices for temporal [1, 6, 7] and static [1, 2]
            for detail in self.interpreter.get_input_details():
                shape = list(detail['shape'])
                if shape == [1, config.SEQUENCE_LENGTH, config.TEMPORAL_FEATURE_DIM]:
                    self.temp_idx = detail['index']
                elif shape == [1, config.STATIC_FEATURE_DIM]:
                    self.stat_idx = detail['index']

            self.out_idx = self.interpreter.get_output_details()[0]['index']
            self.is_tflite = True
            self.engine_name = "TFLite Model Engine (Quantized)"
            print("[ModelService] TFLite model loaded successfully.")
        except Exception as e:
            print(f"[ModelService] TFLite allocation notice: {e}. Running fallback mode.")

    def predict(self, temporal_data: list, static_data: list) -> dict:
        """Runs inference on 3-hour history window (6 steps x 7 features) and static features."""
        X_temp = np.array(temporal_data, dtype=np.float32).reshape(1, config.SEQUENCE_LENGTH, config.TEMPORAL_FEATURE_DIM)
        X_stat = np.array(static_data, dtype=np.float32).reshape(1, config.STATIC_FEATURE_DIM)

        # Execute TFLite model if active
        if self.is_tflite and self.interpreter:
            try:
                self.interpreter.set_tensor(self.temp_idx, X_temp)
                self.interpreter.set_tensor(self.stat_idx, X_stat)
                self.interpreter.invoke()

                prob = float(self.interpreter.get_tensor(self.out_idx)[0][0])
                cgm_curr = float(X_temp[0, -1, 0])
                vel_curr = float(X_temp[0, -1, 6])
                pred_cgm = cgm_curr + (vel_curr * 30.0)

                return {
                    "danger_probability": round(float(np.clip(prob, 0.0, 1.0)), 4),
                    "predicted_cgm_next_30m": round(pred_cgm, 2),
                    "engine": self.engine_name
                }
            except Exception as e:
                print(f"[ModelService] TFLite error: {e}. Using fallback.")

        # Trajectory Fallback Predictor
        cgm_series = X_temp[0, :, 0]
        cgm_curr = float(cgm_series[-1])
        iob_curr = float(X_temp[0, -1, 1])
        cob_curr = float(X_temp[0, -1, 3])
        velocity = float(X_temp[0, -1, 6])

        pred_cgm = cgm_curr + (velocity * 30.0) + (cob_curr * 2.0) - (iob_curr * 10.0)
        pred_cgm = max(30.0, min(500.0, pred_cgm))

        hypo_risk = max(0.0, (config.HYPO_THRESH - pred_cgm) / 35.0) if pred_cgm < config.HYPO_THRESH else 0.0
        hyper_risk = max(0.0, (pred_cgm - config.HYPER_THRESH) / 120.0) if pred_cgm > config.HYPER_THRESH else 0.0
        vel_risk = min(0.9, abs(velocity) / 4.0) if velocity < -1.5 else 0.0

        prob = float(np.clip(max(hypo_risk, hyper_risk, vel_risk), 0.01, 0.99))
        return {
            "danger_probability": round(prob, 4),
            "predicted_cgm_next_30m": round(pred_cgm, 2),
            "engine": self.engine_name
        }

# Global Singleton Instance
cgm_service = CGMModelService()
