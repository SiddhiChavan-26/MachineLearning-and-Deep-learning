import streamlit as st
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import shap
import os
import matplotlib.pyplot as plt

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Model Deployment with SHAP")

st.title("🔍 Model Prediction with Explainability")

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==============================
# LOAD ONNX MODEL (FOR PREDICTION)
# ==============================
onnx_path = os.path.join(BASE_DIR, "model", "model.onnx")
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name

# ==============================
# LOAD PYTORCH MODEL (FOR SHAP)
# ==============================
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = SimpleModel()
model_path = os.path.join(BASE_DIR, "model", "model.pth")
model.load_state_dict(torch.load(model_path))
model.eval()

# ==============================
# USER INPUT
# ==============================
st.subheader("Enter Feature Values")

inputs = []
for i in range(10):
    val = st.number_input(f"Feature {i}", value=0.0)
    inputs.append(val)

# ==============================
# PREDICT BUTTON
# ==============================
if st.button("Predict"):

    # --------------------------
    # Prepare Input
    # --------------------------
    input_data = np.array(inputs).reshape(1, -1).astype(np.float32)

    # --------------------------
    # ONNX Prediction
    # --------------------------
    result = session.run(None, {input_name: input_data})
    prediction = result[0][0][0]

    st.subheader(f"Prediction: {prediction:.4f}")

    # Optional classification label
    if prediction > 0.5:
        st.success("Class 1")
    else:
        st.error("Class 0")

    # --------------------------
    # SHAP Explanation
    # --------------------------
    st.subheader("SHAP Explanation")

    with st.spinner("Generating explanation..."):

        def model_predict(data):
            data_tensor = torch.tensor(data, dtype=torch.float32)
            with torch.no_grad():
                return model(data_tensor).numpy()

        # Background data
        background = np.random.rand(20, 10)

        # SHAP Explainer
        explainer = shap.KernelExplainer(model_predict, background)

        # SHAP values for user input
        shap_values = explainer.shap_values(input_data)

        shap_val = shap_values[0] if isinstance(shap_values, list) else shap_values
        shap_val_single = np.array(shap_val).flatten()

        # Feature names
        feature_names = [f"Feature {i}" for i in range(10)]

        # Create bar plot
        fig, ax = plt.subplots()
        ax.barh(feature_names, shap_val_single)
        ax.set_title("Feature Contribution (SHAP)")
        ax.set_xlabel("Impact on Prediction")
        ax.invert_yaxis()

        st.pyplot(fig)

    st.success("Explanation generated successfully!")