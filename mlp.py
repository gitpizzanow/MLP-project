import numpy as np
import configs
from visualize import visualize_decision_boundary


np.random.seed(42)

"""
(x1,x2) → [ 4 ] → [ 1 ] 
            H1     Out
"""

def sigmoid(a):
    return 1.0 / (1 + np.exp(-a))

def step(a):
    return 1 if a >= 0.5 else 0

def derivative(a):
    return a * (1 - a)

def cross(y, Yhat) -> np.float32:
    Yhat = max(Yhat, configs.EPSILON)
    Yhat = min(Yhat, 1.0 - configs.EPSILON)
    return -(y * np.log(Yhat) + (1 - y) * np.log(1.0 - Yhat))

class Layer:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.uniform(-1, 1, (output_size, input_size))
        self.biases = np.random.uniform(-1, 1, (output_size,))
        self.outputs = np.zeros((output_size,))
        self.deltas = np.zeros((output_size,))
        self.dW = np.zeros_like(self.weights)
        self.dB = np.zeros_like(self.biases)
        self.next = None

def forward(input: np.ndarray, layer: Layer):
    z =  layer.weights @ input
    z += layer.biases.reshape(-1,1)
    layer.outputs = sigmoid(z).flatten()  #1D


def compute_deltas(layers: list[Layer], y, size):
    output_layer = layers[-1]
    yhat = float(output_layer.outputs[0])
    output_layer.deltas[0] = yhat - y

    for l in range(size - 2, -1, -1):
        current = layers[l]
        next_layer = layers[l + 1]

        for j in range(current.output_size):
            sum_error = 0.0
            for i in range(next_layer.output_size):
                sum_error += next_layer.weights[i][j] * next_layer.deltas[i]
            a = current.outputs[j]
            current.deltas[j] = sum_error * derivative(a)

def accumulate(layer: Layer, input):
    input = input.reshape(-1) #flat vector
    for i in range(layer.output_size):
        for j in range(layer.input_size):
            layer.dW[i][j] += layer.deltas[i] * input[j]
        layer.dB[i] += layer.deltas[i]

def apply_gradients(layer: Layer, lr, batch_size):
    for i in range(layer.output_size):
        for j in range(layer.input_size):
            layer.weights[i][j] -= lr * layer.dW[i][j] / batch_size
            layer.dW[i][j] = 0
        layer.biases[i] -= lr * layer.dB[i] / batch_size
        layer.dB[i] = 0

def Train_SGD(layers: list[Layer], X: np.ndarray, y: np.ndarray, epochs: int, lr: float, N_layers: int, rows: int):
    for e in range(epochs):
        loss = 0.0
        correct = 0

        for i in range(rows):
            input = X[i].reshape(-1, 1)

            # Forward pass
            for layer_idx in range(N_layers):
                forward(input, layers[layer_idx])
                input = layers[layer_idx].outputs.reshape(-1, 1)

            yhat = float(layers[-1].outputs[0])
            loss += cross(y[i], yhat)
            if step(yhat) == y[i]:
                correct += 1

            # Backpropagation
            compute_deltas(layers, y[i], N_layers)
            input = X[i].reshape(-1, 1)
            for layer in layers:
                accumulate(layer, input)
                input = layer.outputs.reshape(-1, 1)

            for layer in layers:
                apply_gradients(layer, lr, 1)

        loss /= rows
        accuracy = 100.0 * correct / rows
        print(f"Epoch {e}| Loss = {float(loss):.6f}| Accuracy = {accuracy:.1f}%")


        if correct == rows:
            print(f"Stopping at epoch {e}")
            break

if __name__ == "__main__":
    hidden1 = Layer(configs.COLS, 4)
    output_layer = Layer(4, 1)
    hidden1.next = output_layer


    layers = [hidden1, output_layer]
    N_layers = len(layers)

    X = np.array([[1, 0], [1, 1], [0, 0], [0, 1]], dtype=np.float32)
    y = np.array([1, 0, 0, 1], dtype=np.float32).reshape(-1, 1)

    Train_SGD(layers, X, y, configs.EPOCHS, configs.LR, N_layers, configs.ROWS)

    visualize_decision_boundary(layers, X, y, forward, step)