import numpy as np

class MSE():
    def forward(self, x, y):
        return np.square(y - x)

    def backward(self, x, y):
        return 2 * (y.reshape(1, -1) - x.reshape(1, -1))
