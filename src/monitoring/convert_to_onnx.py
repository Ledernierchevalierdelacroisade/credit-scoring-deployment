from pathlib import Path

import joblib
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "lightgbm_final.joblib")
features = joblib.load(MODEL_DIR / "model_features.joblib")

initial_type = [("float_input", FloatTensorType([None, len(features)]))]

onnx_model = onnxmltools.convert_lightgbm(
    model,
    initial_types=initial_type,
    target_opset=12
)

onnx_path = MODEL_DIR / "lightgbm_final.onnx"

with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Modèle ONNX sauvegardé :", onnx_path)