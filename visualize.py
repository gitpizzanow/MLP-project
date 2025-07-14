import numpy as np
import matplotlib.pyplot as plt

def visualize_decision_boundary(layers, X, y, forward_fn, step_fn):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = []
    for i in range(xx.shape[0]):
        row = []
        for j in range(xx.shape[1]):
            inp = np.array([xx[i, j], yy[i, j]]).reshape(-1, 1)
            for layer in layers:
                forward_fn(inp, layer)
                inp = layer.outputs.reshape(-1, 1)
            row.append(step_fn(layer.outputs[0]))
        Z.append(row)

    Z = np.array(Z)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.binary, edgecolors='k')
    plt.title("Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
<<<<<<< HEAD
    plt.show()
=======
    plt.show()
>>>>>>> aaf335260e10b546d90b63da3f6e7f48e21652dc
