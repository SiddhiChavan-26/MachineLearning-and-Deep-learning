import torch
import torch.nn as nn
import shap
import numpy as np
import os
import matplotlib.pyplot as plt

# ==============================
# STEP 1: Define Model
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

# ==============================
# STEP 2: Load Model
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "model.pth")

model = SimpleModel()
model.load_state_dict(torch.load(model_path))
model.eval()

print("✅ Model loaded")

# ==============================
# STEP 3: Wrapper Function
# ==============================
def model_predict(data):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        output = model(data_tensor).numpy()
    return output

# ==============================
# STEP 4: Background Data
# ==============================
background = np.random.rand(50, 10)

explainer = shap.KernelExplainer(model_predict, background)

# ==============================
# STEP 5: Explain Input
# ==============================
input_data = np.random.rand(1, 10)

shap_values = explainer.shap_values(input_data)

print("✅ SHAP values calculated")

# ==============================
# FIX SHAPE
# ==============================
shap_val = shap_values[0] if isinstance(shap_values, list) else shap_values

# ==============================
# USE SUMMARY PLOT (STABLE)
# ==============================
shap.summary_plot(
    shap_val,
    input_data,
    show=True
)