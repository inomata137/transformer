from transformer.common.np import np
from constants import M, T_INV, T_INV_M, HADAMARD, CX, PAULI_X, PAULI_Y, PAULI_Z
from model import CircuitSimulator

def p2rho(p: np.ndarray):
    '''convert probability distribution to density matrix
    ## Parameters
    p: ndarray(4, 4, ..., 4)
    ## Returns
    rho: ndarray(2^Nq, 2^Nq)
    '''
    assert p.shape == tuple(4 for _ in range(p.ndim))
    rho = p
    for _ in range(p.ndim):
        rho = np.tensordot(rho, T_INV_M, (0, 0))
    # assert rho.shape == tuple(2 for _ in range(p.ndim * 2))
    target = np.arange(p.ndim, dtype=int)
    src = target * 2
    rho = np.moveaxis(rho, src, target)
    rho_size = 2 ** p.ndim
    rho = rho.reshape((rho_size, rho_size))
    '''rho.shape = (2^Nq, 2^Nq)'''
    return rho


def rho2p(rho: np.ndarray, Nq: int):
    '''convert density matrix to probability distribution
    ## Parameters
    rho: ndarray(2^Nq, 2^Nq)
    Nq: number of qubits
    ## Returns
    p: probability distribution
    '''
    assert rho.ndim == 2
    assert rho.shape[0] == 2 ** Nq
    assert rho.shape[1] == 2 ** Nq
    p = rho.reshape(tuple(2 for _ in range(2 * Nq)))
    for i in range(Nq):
        p = np.tensordot(p, M, ([0, Nq - i], [2, 1]))
    assert p.shape == tuple(4 for _ in range(Nq))
    return p.real


def one_qubit_stocastic(u: np.ndarray, pos: int, p_prev: np.ndarray):
    assert u.shape == (2, 2)
    um = np.tensordot(u, M, (1, 1)) # (2, 4, 2)
    um_cc = np.tensordot(u.conj().transpose(), M, (1, 1))
    weight = np.tensordot(um, um_cc, ([0, 2], [2, 0])) # (4, 4)
    o = np.tensordot(weight, T_INV, (0, 0)) # (4, 4)
    p_next = np.tensordot(o, p_prev, (1, pos))
    p_next = np.moveaxis(p_next, 0, pos)
    return p_next.real


def two_qubit_stocastic(u: np.ndarray, pos1: int, pos2: int, p_prev: np.ndarray):
    assert u.shape == (2, 2, 2, 2) # qubit1, qubit2, qubit1', qubit2'
    Nq = p_prev.ndim
    assert pos1 >= 0 and pos1 < Nq
    assert pos2 >= 0 and pos2 < Nq
    mm = np.tensordot(M, M, 0) # 4, 2, 2 / 4, 2, 2
    um = np.tensordot(u, mm, ([2, 3], [1, 4])) # 2, 2 / 4, 4
    u_cc = np.moveaxis(u.conj(), (0, 1), (2, 3))
    umu_cc = np.tensordot(um, u_cc, ([3, 5], [0, 1]))
    o = np.tensordot(umu_cc, mm, ([0, 1, 4, 5], [2, 5, 1, 4])).real
    o = np.tensordot(o, T_INV, (0, 0))
    o = np.tensordot(o, T_INV, (0, 0))
    assert o.shape == (4, 4, 4, 4)
    p_next = np.tensordot(o, p_prev, ([2, 3], [pos1, pos2]))
    p_next = np.moveaxis(p_next, [0, 1], [pos1, pos2])
    return p_next


def one_qubit_p2p(u: np.ndarray, pos: int, a: np.ndarray, cs: CircuitSimulator):
    assert u.shape == (2, 2)
    assert a.ndim == 1
    Nq = len(a)
    samples = np.empty((4, Nq), dtype=int)
    samples[:] = a
    samples[:, pos] = np.arange(4)
    p = cs.forward(samples)


if __name__ == '__main__':
    rho_prev = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    p_prev = rho2p(rho_prev, 2)
    p_next = one_qubit_stocastic(HADAMARD, 0, p_prev)
    # p_next = two_qubit_stocastic(CX, 0, 1, p_next)
    rho_next = p2rho(p_next)

    import matplotlib.pyplot as plt
    plt.colorbar(plt.subplot(221, title='p_prev').imshow(p_prev))
    plt.colorbar(plt.subplot(222, title='rho_prev').imshow(rho_prev))
    plt.colorbar(plt.subplot(223, title='rho_next').imshow(rho_next.real))
    plt.colorbar(plt.subplot(224, title='p_next').imshow(p_next))
    plt.show()
