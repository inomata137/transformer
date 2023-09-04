from .common.np import np
from .common.layers import BaseLayer, Affine, Relu

class PositionWiseFfn(BaseLayer):
    def __init__(self, d_m: int, d_ff: int, b1_scale=1.,
                 b2_scale=1., rn=np.random.randn) -> None:
        super().__init__()
        self.layers: list[Affine | Relu] = [
            Affine(d_m, d_ff, b1_scale, rn),
            Relu(),
            Affine(d_ff, d_m, b2_scale, rn)
        ]
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
