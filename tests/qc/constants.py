from transformer.common.np import np


M0 = np.array([
    [1, 0],
    [0, 0]
]) / 3.

M1 = np.array([
    [1, 1],
    [1, 1]
]) / 6.

M2 = np.array([
    [1, -1j],
    [1j, 1]
]) / 6.

M3 = np.identity(2) - M0 - M1 - M2

M = np.array([M0, M1, M2, M3])
'''(4, 2, 2)'''


T = np.array([
    [2, 1, 1, 2],
    [1, 2, 1, 2],
    [1, 1, 2, 2],
    [2, 2, 2, 12]
]) / 18.

T_INV = np.array([
    [14, -4, -4, -1],
    [-4, 14, -4, -1],
    [-4, -4, 14, -1],
    [-1, -1, -1, 2]
])


T_INV_M = np.tensordot(T_INV, M, axes=(1, 0))
'''(4, 2, 2)'''


PAULI_X = np.array([
    [0, 1],
    [1, 0]
])

PAULI_Y = np.array([
    [0, -1j],
    [1j, 0]
])

PAULI_Z = np.array([
    [1, 0],
    [0, -1]
])

HADAMARD = np.array([
    [1, 1],
    [1, -1]
]) / np.sqrt(2)
'''(2, 2)'''

CX = np.array([
    [
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]]
    ],
    [
        [[0, 0], [0, 1]],
        [[0, 0], [1, 0]]
    ]
])
'''(2, 2, 2, 2)'''

if __name__ == '__main__':
    print(M3)
    # mumu = np.einsum('acd,de,bef,fc->ab', M, HADAMARD, M, HADAMARD)
    # O: np.ndarray = T_INV @ mumu
    # O = O.real
    # import matplotlib.pyplot as plt
    # plt.matshow(O)
    # plt.colorbar()
    # plt.xlabel(r'$a$')
    # plt.ylabel(r"$a'$")
    # for i in range(4):
    #     for j in range(4):
    #         c = 'black' if O[i, j] > 0.5 else 'w'
    #         plt.text(j, i, round(O[i, j] + 1e-5, 2), ha='center', va='center', color=c)
    # plt.show()