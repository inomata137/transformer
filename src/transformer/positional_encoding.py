from .common.np import np

def positional_encoding(x: np.ndarray, T=1e4):
    b, r, c = x.shape
    pos = np.arange(r).reshape((r, 1))
    dim = np.arange(c).reshape((1, c))
    pe_even = np.sin(pos / T**(dim / c)) * ((dim + 1) % 2)
    pe_odd = np.cos(pos / T**((dim - 1) / c)) * (dim % 2)
    pe = (pe_even + pe_odd)
    return x + pe.reshape(1, r, c)
