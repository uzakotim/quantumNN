import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set up a quantum device with 2 qubits using PennyLane
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# Define a quantum circuit as a variational quantum layer
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# Define a PyTorch quantum layer
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    def forward(self, x):
        x = self.q_layer(x)
        return x.unsqueeze(1)

# Define a hybrid quantum-classical neural network
class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = QuantumLayer(n_qubits, n_layers=2)
        self.fc = nn.Linear(1, 1)  # Classical layer
    
    def forward(self, x):
        x = self.q_layer(x)
        return torch.sigmoid(self.fc(x))

# Generate simple dataset (XOR-like pattern)
np.random.seed(42)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)  # XOR labels

# Convert to PyTorch tensors
X_torch = torch.tensor(X)
y_torch = torch.tensor(y)

# Instantiate and train the model
model = HybridQNN()
optimizer = optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_torch)
    loss = loss_fn(outputs, y_torch)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test the model
print("\nPredictions:")
with torch.no_grad():
    predictions = model(X_torch)
    print(predictions.numpy().round(2))