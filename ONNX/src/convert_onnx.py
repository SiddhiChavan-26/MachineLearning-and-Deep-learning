import torch
import torch.nn as nn
import os

# ==============================
# STEP 1: Define SAME Model
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
# STEP 2: Load Trained Model
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "model.pth")

model = SimpleModel()
model.load_state_dict(torch.load(model_path))
model.eval()

print("✅ Model loaded successfully")

# ==============================
# STEP 3: Create Dummy Input
# ==============================

dummy_input = torch.randn(1, 10)

# ==============================
# STEP 4: Convert to ONNX
# ==============================

onnx_path = os.path.join(BASE_DIR, "model", "model.onnx")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print(f"✅ Model converted to ONNX at: {onnx_path}")