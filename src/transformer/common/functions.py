from .np import np

def softmax(x: np.ndarray):
    x -= x.max(axis=-1, keepdims=True)
    y: np.ndarray = np.exp(x)
    y /= y.sum(axis=-1, keepdims=True)
    return y


def cross_entropy_error(y: np.ndarray, t: np.ndarray, eps=1e-9) -> float:
    batch_size = y.shape[0]
    return -np.dot(np.log(y + eps), t) / batch_size

