from .common.np import np
from .common.layers import BaseLayer, Dropout
from typing import Union

class LayerNorm(BaseLayer):
    def __init__(self, positionwise=False):
        super().__init__()
        self.sigma = None
        self.y = None
        self.pw = positionwise

    def forward(self, x: np.ndarray):
        N, n, d_m = x.shape
        if not self.pw:
            x = x.reshape((N, n * d_m))
        mu = x.mean(axis=-1, keepdims=True)
        sigma = x.std(axis=-1, keepdims=True)
        self.y = (x - mu) / sigma
        self.sigma = sigma
        return self.y.reshape((N, n, d_m))

    def backward(self, dout: np.ndarray):
        '''https://zenn.dev/taro137/articles/2c639b39a32166'''
        N, n, d_m = dout.shape
        if not self.pw:
            dout = dout.reshape((N, n * d_m))
        H = dout.shape[-1]
        a = dout.sum(axis=-1, keepdims=True)
        b = np.einsum('...i,...i->...', self.y, dout)[..., None]
        dx = (dout - (a + b * self.y) / H) / self.sigma
        return dx.reshape((N, n, d_m))


class ResidualConnection(BaseLayer):
    def __init__(self, layer: BaseLayer, p_drop: float, norm_positionwise: bool):
        super().__init__()
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

    def backward(self, dout: np.ndarray) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
        dx1 = self.layer_norm.backward(dout)
        dx2 = self.dropout.backward(dx1)
        dx2 = self.layer.backward(dx2)
        if type(dx2) == tuple:
            dx1 += dx2[0]
            return dx1, *(dx2[1:])
        else:
            return dx1 + dx2
