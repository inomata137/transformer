from common.np import np
from common.layers import BaseLayer, Dropout

class LayerNorm(BaseLayer):
    def __init__(self, positionwise=False, eps=1e-10):
        super.__init__()
        self.mu = None
        self.sigma = None
        self.x = None
        self.pw = positionwise
        self.eps = eps

    def forward(self, x: np.ndarray):
        '''x: N x n x d_m tensor'''
        N, n, d_m = x.shape
        if self.pw:
            x = x.reshape((N * n, 1, d_m))
            norm = d_m
        else:
            norm = n * d_m
        mu: np.ndarray = np.sum(x, axis=(1, 2), keepdims=True) / norm
        # mu: M x 1 x 1, M = N * n if pw else N
        sqsum: np.ndarray = np.sum((x - mu) ** 2 / norm, axis=(1, 2), keepdims=True)
        # sqsum: M x 1 x 1
        sigma = np.sqrt(sqsum) + self.eps
        # sigma: M x 1 x 1
        self.mu = mu
        self.sigma = sigma
        self.x = x
        out = (x - mu) / sigma
        if self.pw:
            out = out.reshape((N, n, d_m))
        return out

    def backward(self, dout: np.ndarray):
        mu = self.mu
        # mu: M x 1 x 1
        sigma = self.sigma
        # sigma: M x 1 x 1
        x = self.x
        # x: M x m x d_m, m = 1 if pw else n
        shape = dout.shape
        d_m = shape[-1]
        if self.pw:
            N, n = shape[0] * shape[1], 1
            dout = dout.reshape((N, n, d_m))
        else:
            N, n = shape[:2]
        norm = n * d_m
        j1 = (np.identity(norm) * norm - 1) / (norm * sigma)
        j2 = np.matmul(
            x.reshape((N, norm, 1)) - mu,
            x.reshape((N, 1, norm)) - mu
        ) / (norm * sigma**3)
        j = j1 - j2
        dx: np.ndarray = np.matmul(dout.reshape((N, 1, norm)), j)
        return dx.reshape(shape)

class ResidualConnection(BaseLayer):
    def __init__(self, layer: BaseLayer, p_drop: float, norm_positionwise: bool):
        super.__init__()
        self.layer = layer
        self.layer_norm = LayerNorm(norm_positionwise)
        self.dropout = Dropout(p_drop)
        self.params += layer.params
        self.grads += layer.grads
    
    def forward(self, *args, **kwargs):
        train_flg = kwargs.get('train_flg', True)
        y: np.ndarray = self.layer.forward(*args)
        y = self.dropout.forward(y, train_flg)
        s: np.ndarray = args[0] + y
        out = self.layer_norm.forward(s)
        return out

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        dx1 = self.layer_norm.backward(dout)
        dx2 = self.dropout.backward(dx1)
        dx2 = self.layer.backward(dx2)
        if type(dx2) == tuple:
            dx1 += dx2[0]
            return dx1, *(dx2[1:])
        else:
            return dx1 + dx2
