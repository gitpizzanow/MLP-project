import numpy as np
import configs


"""

(x1,x2) → [ 3 ] → [ 1 ]
            H1     Out

"""


def sigmoid(a):
    return 1.0 / (1 + (np.exp(-a)))


def step(a):
    return 1 if a > 0 else 0


class Layer:
    def __init__(this, input_size: int, output_size: int):
        this.input_size = input_size
        this.output_size = output_size

        this.weights = np.random.uniform(
            -1, 1, (output_size, input_size)
        )  # [input_size * output_size]
        this.biases = np.random.uniform(-1, 1, (output_size,))  # [output_size]
        this.deltas = np.random.uniform(-1, 1, (output_size,))  # [output_size]
        this.outputs = np.random.uniform(
            -1, 1, (output_size,)
        )  # [output_size]

        this.dW = np.zeros_like(this.weights)  # Gradient of weights
        this.dB = np.zeros_like(this.biases)  # Gradient of biases
        this.next = None


def forward(input: np.ndarray, layer: Layer):
    z = (input @ layer.weights.T) + layer.biases
    layer.outputs = sigmoid(z)


if __name__ == "__main__":
    hidden1 = Layer(configs.COLS, 3)
    output_layer = Layer(3, 1)
    hidden1.next = output_layer

    layers: list[Layer] = [hidden1, output_layer]
    N_layers = len(layers)

    X = np.array(
        [[1, 0], [1, 1], [0, 0], [0, 1]], dtype=np.float32  # 1  # 0  # 0  # 1
    )

    y = np.array([1, 0, 0, 1], dtype=np.float32).reshape(-1, 1)
    forward(X, hidden1)
    forward(hidden1.outputs, output_layer)

    for i in output_layer.outputs:
        print(step(i))
