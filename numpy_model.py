# ===== region note_1 =========
"""
INPUT LAYER
────────────
X  (input features)
│
│   Linear layer 1
│   Z1 = XW1 + b1
▼
Z1
│
│   Activation (ReLU)
│   A1 = ReLU(Z1)
▼
A1  (hidden layer output)
│
│   Linear layer 2
│   Z2 = A1W2 + b2
▼
Z2
│
│   Activation (Sigmoid)
│   A2 = sigmoid(Z2)
▼
A2  (output probability)
│
│   Threshold (predict)
│   if A2 ≥ 0.5 → class 1
│   else → class 0
▼
PREDICTION (0 or 1)

Training loop:

FORWARD PASS
X → Z1 → A1 → Z2 → A2

        ↓

LOSS
compare A2 with y

        ↓

BACKWARD PASS
compute gradients:
dW2, dW1, db2, db1

        ↓

UPDATE
W1, W2, b1, b2 adjusted

        ↓

REPEAT (epochs)

SYMBOLS:

| Variable | Meaning                         |
| -------- | ------------------------------- |
| X        | input data                      |
| W1, W2   | weights (what the model learns) |
| b1, b2   | bias terms                      |
| Z1, Z2   | raw linear outputs              |
| A1       | hidden layer activation         |
| A2       | final probability               |

"""
# ===== endregion ==========

import numpy as np

class SimpleNeuralNet:
    def __init__(self, input_dim, hidden_dim=8, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Trims outliers

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)  
# X -> Z1 -> A1 -> Z2 -> sigmoid -> A2

        return self.A2

    def backward(self, X, y):
        m = X.shape[0]

        dZ2 = self.A2 - y
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = (dZ2 @ self.W2.T) * self.relu_deriv(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2

    def fit(self, X, y):
        for i in range(self.epochs):
            self.forward(X)
            self.backward(X, y)

            if i % 100 == 0:
                loss = -np.mean(
                    y * np.log(self.A2 + 1e-8) +
                    (1 - y) * np.log(1 - self.A2 + 1e-8)
                )
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)
        
