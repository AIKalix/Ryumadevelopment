import numpy as np

class nn():
    def __init__(self, v, learn):
        self.w = np.random.randn(v, v)
        self.prev_x = None
        self.prev_sig = None
        self.upg_w = np.zeros((v, v))
        self.learn = learn

    def forward(self, x):
        self.prev_x = x.reshape(-1, 1)
        tmp = np.dot(self.w, self.prev_x)
        self.prev_sig = 1/(1 + np.exp(-1 * tmp))
        return self.prev_sig

    def backward(self):
        ans = (self.prev_sig * (1 - self.prev_sig)).reshape(1 ,-1)
        self.upg_x = np.dot(self.prev_x.reshape(-1, 1), ans).reshape(self.w.shape)
        return np.dot(ans, self.w)

    def update(self):
        self.w -= self.upg_x * self.learn

    def topback(self, ans):
        ans = (ans *(1 - ans)).reshape(1, -1)
        self.upg_x = np.dot(self.prev_x.reshape(-1, 1), ans).reshape(self.w.shape)
        return np.dot(ans, self.w)
