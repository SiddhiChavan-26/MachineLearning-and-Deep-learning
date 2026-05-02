import onnxruntime as ort
import numpy as np
import os

# ==============================
# STEP 1: Load ONNX Model
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
onnx_path = os.path.join(BASE_DIR, "model", "model.onnx")

session = ort.InferenceSession(onnx_path)

print("✅ ONNX model loaded")

# ==============================
# STEP 2: Prepare Input
# ==============================

# Example input (10 features)
input_data = np.random.rand(1, 10).astype(np.float32)

# Get input name
input_name = session.get_inputs()[0].name

# ==============================
# STEP 3: Run Prediction
# ==============================

result = session.run(None, {input_name: input_data})

print("\nInput:")
print(input_data)

print("\nPrediction:")
print(result[0])