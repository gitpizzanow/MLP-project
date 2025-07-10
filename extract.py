from PIL import Image
import numpy as np
import os

X = []
y = []


base_path = "data/train"


for label_name, label_value in [('cats', 1), ('dogs', 0)]:
    folder = os.path.join(base_path, label_name)
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            path = os.path.join(folder, filename)
            img = Image.open(path).convert('L').resize((28, 28))
            arr = np.asarray(img).flatten() / 255.0
            X.append(arr)
            y.append(label_value)

X = np.array(X)
y = np.array(y)

np.savetxt('X_data.txt', X, delimiter=",")
np.savetxt('y_data.txt', y, fmt="%d", delimiter=",")
