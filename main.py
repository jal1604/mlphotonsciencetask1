import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Load data
df = pd.read_csv("data1.csv")

X = df.drop(columns=["filename", "hex_x", "hex_z"]).values
y = df[["hex_x", "hex_z"]].values

# Train / validation / test split (70 / 15 / 15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# Normalize predictors
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Tensors
def tensor(x):
    return torch.tensor(x, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(tensor(X_train), tensor(y_train)),
    batch_size=32, shuffle=True
)

X_val, y_val = tensor(X_val), tensor(y_val)
X_test, y_test = tensor(X_test), tensor(y_test)

# Neural network
class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)

model = Net(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training with best-model saving
best_val_loss = float("inf")
train_losses, val_losses = [], []

for epoch in range(200):
    model.train()
    epoch_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)

    train_loss = epoch_loss / len(X_train)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

# Load best model
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model.eval()

# Final test evaluation using best model
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test).item()

print(f"\nBest validation MSE: {best_val_loss:.6f}")
print(f"Test MSE: {test_loss:.6f}")
print(f"Test RMSE: {test_loss**0.5:.6f}")

# Learning curve
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Learning Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()