import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ==============================
# STEP 1: Create Dataset
# ==============================
X = np.random.rand(1000, 10).astype(np.float32)
y = (np.sum(X, axis=1) > 5).astype(np.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train).view(-1, 1)

# ==============================
# STEP 2: Define Model
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

# ==============================
# STEP 3: Loss & Optimizer
# ==============================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==============================
# STEP 4: Training Loop
# ==============================
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# ==============================
# STEP 5: Save Model (FIXED PATH)
# ==============================

# Get root directory (outside src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create model folder if not exists
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

# Save model
model_path = os.path.join(model_dir, "model.pth")
torch.save(model.state_dict(), model_path)

print(f"\n✅ Model trained and saved at: {model_path}")