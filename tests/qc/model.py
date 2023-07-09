from transformer.common.np import np
from transformer.common import BaseLayer, MatMul, Softmax, Embedding, BaseModel
from transformer.attention import MultiHeadSelfAttention
from transformer.ffn import PositionWiseFfn
from transformer.positional_encoding import positional_encoding
from transformer.residual_connection import ResidualConnection

class CircuitSimulator(BaseModel):
    def __init__(self, m: int, d_m: int, h: int, d_ff: int,
                 rn=np.random.randn) -> None:
        super().__init__()
        self.embed = Embedding(m + 1, d_m, rn) # +1はinit用
        '''0, 1, ..., m-1は測定値、mはinit'''
        self.layers: list[BaseLayer] = [
            ResidualConnection(MultiHeadSelfAttention(d_m, h, True, rn), 0., True),
            ResidualConnection(PositionWiseFfn(d_m, d_ff, 0.1, 0.1, rn), 0., True),
            MatMul((d_m, m), rn),
        ]
        self.softmax = Softmax()
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
        self.m = m
    
    def forward(self, batch: int, n: int, p_e: np.ndarray):
        '''
        batch: number of samples
        n: number of qubits
        p_e: ndarray(size=(m, m, ...)) represents P^(e)(a)
        '''
        assert p_e.shape == tuple(self.m for _ in range(len(p_e.shape)))
        rng = np.random.default_rng(0)
        x = np.zeros((batch, n), dtype=int)
        x[:, 0] = self.m
        for qubit_idx in range(n):
            y = self.embed.forward(x)
            y = positional_encoding(y, 10)
            for layer in self.layers:
                y = layer.forward(y)
            y = self.softmax.forward(y)
            for batch_idx in range(batch):
                x[batch_idx, (qubit_idx + 1) % n, ...] = rng.choice(self.m, p=y[batch_idx, qubit_idx])

        self.a = a = np.roll(x, shift=-1, axis=-1)
        '''
        size (batch, N)
        a[i, j] represents a_j of i-th sample
        '''
        self.y = y
        '''
        size (batch, N, m)
        y[i, j, k] represents P(k|a_1a_2...) of j-th qubit of i-th sample
        '''
        p = np.ones(batch)
        for pos in range(n):
            p *= np.take_along_axis(y[:, pos], a[:, pos, None], axis=1)[:, 0]
        f = np.empty(batch)
        for batch_idx in range(batch):
            tmp = p_e
            for pos in range(n):
                tmp = p[a[batch_idx, pos]]
            f[batch_idx] = tmp
        f /= p
        self.f = f
        loss = np.dot(f, np.log(f)) / batch
        return loss

    def backward(self, dout=1.):
        a = self.a
        y = self.y
        f = self.f
        batch, n = a.shape

        t = np.zeros((batch * n, self.m))
        for i in range(batch * n):
            t[i, a.flatten()[i]] = 1.
        t = t.reshape((batch, n, self.m))
        
        dout = (y - t) * f.reshape((f.size, 1, 1)) / batch
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        dout = self.embed.backward(dout)
        return dout
