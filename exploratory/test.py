import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate training data
x_train = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)
y_train = np.tan(x_train)

# Normalize y_train to prevent exploding gradients (tangent grows large)
# y_train = y_train / np.max(np.abs(y_train))

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Initialize model, loss function, and optimizer
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

# Evaluate model
x_test = np.linspace(-np.pi, np.pi, 3000).reshape(-1, 1)
y_test = np.tan(x_test)
# y_test = y_test / np.max(np.abs(y_test))  # Normalize

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_pred_tensor = model(x_test_tensor).detach().numpy()

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(x_test, y_test, label="True tan(x) (scaled)", linestyle="dashed")
plt.plot(x_test, y_pred_tensor, label="MLP Approximation")
plt.legend()
plt.xlabel("x")
plt.ylabel("tan(x)")
plt.title("MLP Approximation of tan(x)")
plt.grid()
plt.show()
plt.savefig('MLP Approximation of tan(x).png')
