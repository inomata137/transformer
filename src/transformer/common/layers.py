from .np import np
from .functions import softmax, cross_entropy_error
from ..config import gpu_config

if gpu_config():
    import cupyx

class BaseLayer:
    def __init__(self) -> None:
        self.params: list[np.ndarray] = []
        self.grads: list[np.ndarray] = []

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError


class MatMul(BaseLayer):
    def __init__(self, shape: tuple[int, int], rn=np.random.randn) -> None:
        super().__init__()
        W: np.ndarray = rn(*shape) / np.sqrt(shape[0])
        self.params.append(W)
        self.grads.append(np.zeros_like(W))
        self.x = None

    def forward(self, x: np.ndarray):
        W, = self.params
        out: np.ndarray = np.matmul(x, W)
        self.x = x
        return out
    
    def backward(self, dout: np.ndarray):
        W, = self.params
        dx: np.ndarray = np.matmul(dout, W.T)
        dW = np.sum(np.matmul(self.x.swapaxes(-2, -1), dout), axis=0)
        self.grads[0][...] = dW
        return dx


class Affine(BaseLayer):
    def __init__(self, d_in: int, d_out: int, b_scale=1., rn=np.random.randn) -> None:
        super().__init__()
        self.mm = MatMul((d_in, d_out), rn)
        b = rn(1, 1, d_out) * b_scale
        self.params = [
            b,
            *self.mm.params
        ]
        self.grads = [
            np.zeros_like(b),
            *self.mm.grads
        ]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.mm.forward(x) + self.params[0]

    def backward(self, dout: np.ndarray):
        self.grads[0][...] = dout.sum(axis=(0, 1))
        return self.mm.backward(dout)


class SimpleMatMul(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.a = None
        self.b = None
    
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.a, self.b = a, b
        return np.matmul(a, b)

    def backward(self, dout: np.ndarray):
        da: np.ndarray = np.matmul(dout, self.b.swapaxes(-2, -1))
        db: np.ndarray = np.matmul(self.a.swapaxes(-2, -1), dout)
        return da, db


class Softmax(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.out = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = softmax(x)
        return self.out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = self.out * dout
        sumdx = np.sum(dx, axis=-1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss(BaseLayer):
    def __init__(self, e_ls = 0.) -> None:
        super().__init__()
        self.y = None
        self.t = None
        self.e_ls = e_ls
    
    def forward(self, x: np.ndarray, t: np.ndarray):
        '''
        t: one-hot, same shape as x
        '''
        vocab_size = t.shape[-1]
        e_ls = self.e_ls
        self.t = t * (1 - e_ls * vocab_size / (vocab_size - 1)) + e_ls / (vocab_size - 1)
        self.y = softmax(x)
        return cross_entropy_error(self.y, self.t)

    def backward(self, dout=1.) -> np.ndarray:
        y = self.y
        t = self.t
        batch_size = t.shape[0]
        return (y - t) * dout / batch_size


class Dropout(BaseLayer):
    '''
    http://arxiv.org/abs/1207.0580
    '''
    def __init__(self, dropout_ratio=0.1, seed=None):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.rng = np.random.default_rng(seed)

    def forward(self, x: np.ndarray, train_flg=True) -> np.ndarray:
        if train_flg:
            self.mask = self.rng.random(x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask


class Embedding(BaseLayer):
    def __init__(self, vocab_size: int, d_m: int, rn=np.random.randn):
        super().__init__()
        W: np.ndarray = rn(vocab_size, d_m)
        self.params.append(W)
        self.grads.append(np.zeros_like(W))
        self.idx = None

    def forward(self, idx: np.ndarray):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout: np.ndarray):
        dW, = self.grads
        dW[...] = 0
        if gpu_config():
            cupyx.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)
        return None


class Relu(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.cache = None
    
    def forward(self, x: np.ndarray):
        r = np.maximum(x, 0)
        self.cache = np.heaviside(x, 0.)
        return r
    
    def backward(self, dx: np.ndarray) -> np.ndarray:
        return self.cache * dx
