from .common.np import np
from .common.layers import BaseLayer, MatMul, SimpleMatMul, Softmax

class AttentionHead(BaseLayer):
    def __init__(self, d_m: int, d_k: int,
                 mask: bool, rn=np.random.randn) -> None:
        super().__init__()
        self.wq = MatMul((d_m, d_k), rn)
        self.wk = MatMul((d_m, d_k), rn)
        self.wv = MatMul((d_m, d_k), rn)
        self.matmul1 = SimpleMatMul()
        self.softmax = Softmax()
        self.matmul2 = SimpleMatMul()
        self.params = [
            *self.wq.params,
            *self.wk.params,
            *self.wv.params,
            *self.matmul1.params,
            *self.softmax.params,
            *self.matmul2.params
        ]
        self.grads = [
            *self.wq.grads,
            *self.wk.grads,
            *self.wv.grads,
            *self.matmul1.grads,
            *self.softmax.grads,
            *self.matmul2.grads
        ]
        self.d_k = d_k
        self.mask = mask
        self.attention_weight = None

    def forward(self, x_q: np.ndarray, x_kv: np.ndarray):
        q = self.wq.forward(x_q)
        k = self.wk.forward(x_kv)
        v = self.wv.forward(x_kv)
        x = self.matmul1.forward(q, k.swapaxes(-2, -1)) / np.sqrt(self.d_k)
        if self.mask:
            x -= np.triu(np.full_like(x, np.inf), 1)
        x = self.softmax.forward(x)
        self.attention_weight = x.copy()
        x = self.matmul2.forward(x, v)
        return x

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dv = self.matmul2.backward(dout)
        dx = self.softmax.backward(dx)
        if self.mask:
            dx = np.tril(dx)
        dq, dkT = self.matmul1.backward(dx / np.sqrt(self.d_k))
        dx_q = self.wq.backward(dq)
        dx_k = self.wk.backward(dkT.swapaxes(-2, -1))
        dx_v = self.wv.backward(dv)
        dx_kv = dx_k + dx_v
        return dx_q, dx_kv


class MultiHeadAttention(BaseLayer):
    def __init__(self, d_m: int, h: int, mask: bool, rn=np.random.randn):
        super().__init__()
        assert d_m % h == 0, 'd_m/h should be an integer'
        d_v = d_k = d_m // h
        self.heads = [AttentionHead(d_m, d_k, mask, rn) for _ in range(h)]
        self.wo = MatMul((d_m, d_m), rn)
        for head in self.heads:
            self.params += head.params
            self.grads += head.grads
        self.params += self.wo.params
        self.grads += self.wo.grads
        self.h = h
        self.d_v = d_v

    def forward(self, x_q: np.ndarray, x_kv: np.ndarray):
        '''
        x: (N, n, d_m)
        '''
        # result: list[futures.Future] = []
        # with futures.ThreadPoolExecutor() as executor:
        #     for head in self.heads:
        #         result.append(executor.submit(head.forward, x_q, x_kv))
        # result = tuple(f.result() for f in result)
        result = tuple(
            head.forward(x_q, x_kv) for head in self.heads
        )
        x = np.dstack(result)
        x = self.wo.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        dout: (N, n, d_m)
        '''
        dout = self.wo.backward(dout) # (N, n, h*d_v)
        # result = []
        # with futures.ThreadPoolExecutor() as executor:
        #     for i, head in enumerate(self.heads):
        #         datt = dout[:, :, i*self.d_v:(i+1)*self.d_v]
        #         result.append(executor.submit(head.backward, datt))
        # result = [f.result() for f in result]
        result = [head.backward(
            dout[:, :, i * self.d_v : (i + 1) * self.d_v]
        ) for i, head in enumerate(self.heads)]
        dx_q = 0
        dx_kv = 0
        for r in result:
            dx_q += r[0]
            dx_kv += r[1]
        return dx_q, dx_kv


class MultiHeadSelfAttention(BaseLayer):
    def __init__(self, d_m: int, h: int, mask: bool, rn=np.random.randn):
        super().__init__()
        self.mha = MultiHeadAttention(d_m, h, mask, rn)
        self.params += self.mha.params
        self.grads += self.mha.grads

    def forward(self, x: np.ndarray):
        return self.mha.forward(x, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx_q, dx_kv = self.mha.backward(dout)
        return dx_q + dx_kv


class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, d_m: int, h: int, rn=np.random.randn):
        super().__init__(d_m, h, False, rn)
        self.mha = self
