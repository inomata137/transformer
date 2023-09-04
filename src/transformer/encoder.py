from .ffn import PositionWiseFfn
from .residual_connection import ResidualConnection
from .attention import MultiHeadSelfAttention
from .common.np import np
from .common.layers import BaseLayer

rn = np.random.randn

class Encoder(BaseLayer):
    def __init__(self, d_m: int, h: int, d_ff: int, repeat_num: int,
                 p_drop: float, norm_positionwise=False, rn=rn):
        super().__init__()
        assert d_m % h == 0
        self.layers = [[
            ResidualConnection(MultiHeadSelfAttention(d_m, h, False, rn), p_drop, norm_positionwise),
            # ResidualConnection(PositionWiseFfn(d_m, d_ff, 0.1, 0.1, rn), p_drop, norm_positionwise)
        ] for _ in range(repeat_num)]
        for layer in self.layers:
            for sublayer in layer:
                self.params += sublayer.params
                self.grads += sublayer.grads

    def forward(self, x: np.ndarray, train_flg=True):
        for layer in self.layers:
            for sublayer in layer:
                x = sublayer.forward(x, train_flg=train_flg)
        return x

    def backward(self, dx: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            for sublayer in reversed(layer):
                dx = sublayer.backward(dx)
        return dx
