from .ffn import PositionWiseFfn
from .residual_connection import ResidualConnection
from .attention import MultiHeadCrossAttention, MultiHeadSelfAttention
from .common.np import np
from .common.layers import BaseLayer

rn = np.random.randn

class Decoder(BaseLayer):
    def __init__(self, d_m, h, d_ff, repeat_num: int, p_drop: float, rn=rn) -> None:
        assert d_m % h == 0
        super().__init__()
        self.layers = [[
            ResidualConnection(MultiHeadSelfAttention(d_m, h, True, rn), p_drop, False),
            ResidualConnection(MultiHeadCrossAttention(d_m, h, rn), p_drop, False),
            ResidualConnection(PositionWiseFfn(d_m, d_ff, 0.1, 0.1, rn), p_drop, False)
        ] for _ in range(repeat_num)]
        for layer in self.layers:
            for sublayer in layer:
                self.params += sublayer.params
                self.grads += sublayer.grads
        
    def forward(self, x: np.ndarray, hs: np.ndarray, train_flg=True):
        for layer in self.layers:
            sa, at, pf = layer
            x = sa.forward(x, train_flg=train_flg)
            x = at.forward(x, hs, train_flg=train_flg)
            x = pf.forward(x, train_flg=train_flg)
        return x
    
    def backward(self, dx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dhs = 0.
        for layer in reversed(self.layers):
            sa, at, pf = layer
            dx = pf.backward(dx)
            dx, _dhs = at.backward(dx)
            dx = sa.backward(dx)
            dhs += _dhs
        return dx, dhs
