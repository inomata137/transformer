from .np import np

class SGD:
    '''
    確率的勾配降下法（Stochastic Gradient Descent）
    '''
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, max_grad=np.inf):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        self.max_grad = max_grad

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            grad = np.clip(grads[i], -self.max_grad, self.max_grad)
            self.m[i] += (1 - self.beta1) * (grad - self.m[i])
            self.v[i] += (1 - self.beta2) * (grad**2 - self.v[i])
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
