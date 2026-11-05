import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Load data
df = pd.read_csv("data3.csv")

X = df.drop(columns=["filename", "hex_x", "hex_z"]).values
y = df[["hex_x", "hex_z"]].values

# Train / validation / test split (70 / 15 / 15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# Scale X and y (fit only on training data)
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)
X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train)
y_val = y_scaler.transform(y_val)
y_test_scaled = y_scaler.transform(y_test)

# Tensors
def tensor(x):
    return torch.tensor(x, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(tensor(X_train), tensor(y_train)),
    batch_size=32, shuffle=True
)

X_val, y_val = tensor(X_val), tensor(y_val)
X_test = tensor(X_test)

# Neural network
class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size,128*2),
            nn.ReLU(),
            nn.Linear(128*2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)

model = Net(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.MSELoss()

# Training
best_val_loss = float("inf")
best_epoch = 0
train_losses, val_losses = [], []

for epoch in range(500):
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
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "best_model.pth")

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

# Load best model
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model.eval()

# Test best model (inverse-transform predictions)
with torch.no_grad():
    predictions_scaled = model(X_test).numpy()

predictions = y_scaler.inverse_transform(predictions_scaled)

# Test metrics in original scale
test_mse = ((predictions - y_test) ** 2).mean()
test_rmse = test_mse ** 0.5

print(f"\nBest epoch: {best_epoch}")
print(f"Best validation MSE (scaled): {best_val_loss:.6f}")
print(f"Test MSE (original scale): {test_mse:.6f}")
print(f"Test RMSE (original scale): {test_rmse:.6f}")

# Individual target errors
for i, name in enumerate(["hex_x", "hex_z"]):
    mse = ((predictions[:, i] - y_test[:, i]) ** 2).mean()
    print(f"{name} test MSE: {mse:.6f}")

# Learning curve
plt.figure(figsize=(9, 5))
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.axvline(best_epoch - 1, color="red", linestyle="--",
            label=f"Best epoch ({best_epoch})")
plt.scatter(best_epoch - 1, best_val_loss, color="red", zorder=5)

plt.xlabel("Epoch")
plt.ylabel("Scaled MSE")
plt.title("Learning Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for i, name in enumerate(["hex_x", "hex_z"]):
    axes[i].scatter(y_test[:, i], predictions[:, i], alpha=0.6)
    axes[i].plot(
        [y_test[:, i].min(), y_test[:, i].max()],
        [y_test[:, i].min(), y_test[:, i].max()],
        "r--"
    )
    axes[i].set_title(name)
    axes[i].set_xlabel("Actual")
    axes[i].set_ylabel("Predicted")
    axes[i].grid()

plt.tight_layout()
plt.show()