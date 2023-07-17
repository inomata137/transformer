from transformer.common.np import np
from transformer.common import BaseLayer, MatMul, Softmax, Embedding, BaseModel, RandomChoiceGenerator
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
        self.rng = np.random.default_rng()
        # self.rcg = RandomChoiceGenerator()

    def forward(self, batch: int, n: int, p_e: np.ndarray):
        '''
        batch: number of samples
        n: number of qubits
        p_e: ndarray(size=(m, m, ...)) represents P^(e)(a)
        '''
        assert p_e.shape == tuple(self.m for _ in range(len(p_e.shape)))
        x = np.zeros((batch, n), dtype=int)
        x[:, 0] = self.m
        for qubit_idx in range(n):
            y = self.embed.forward(x)
            y = positional_encoding(y, 10)
            for layer in self.layers:
                y = layer.forward(y)
            y = self.softmax.forward(y)
            rns = self.rng.random(batch)
            b1 = rns >= y[:, qubit_idx, 0]
            b2 = rns >= y[:, qubit_idx, 0] + y[:, qubit_idx, 1]
            b3 = rns >= 1 - y[:, qubit_idx, 3]
            results = b1.astype(int) + b2.astype(int) + b3.astype(int)
            x[:, (qubit_idx + 1) % n] = np.array(results)

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
        f = p_e[tuple(qubit for qubit in a.T)]
        f /= p
        self.f = f

        # kl_div = (f * np.log(f)).mean().item()

        l1_norm = np.abs(f - 1).mean().item()

        # accurate KL div for 2-qubit
        errs = np.full((self.m, self.m), np.inf)
        for batch_idx in range(batch):
            a1, a2 = a[batch_idx]
            if errs[a1, a2] == np.inf:
                p_i = p_e[a1, a2]
                p_theta = p[batch_idx]
                errs[a1, a2] = p_i * np.log(p_i / p_theta)
                if np.all(errs != np.inf):
                    break
        kl_div_accurate = errs.sum().item()
        if kl_div_accurate == np.inf:
            print('KL div is infinity')
        return kl_div_accurate, l1_norm

    def backward(self, dout=1.):
        a = self.a
        y = self.y
        f = self.f
        batch, n = a.shape

        t = np.zeros((batch * n, self.m))
        t[np.arange(batch * n), a.flatten()] = 1.
        t = t.reshape((batch, n, self.m))

        dout = (y - t) * f.reshape((f.size, 1, 1)) / batch
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        dout = self.embed.backward(dout)
        return dout
