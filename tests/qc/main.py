import matplotlib.pyplot as plt
from transformer.common.np import np
from transformer.common import Adam
from model import CircuitSimulator
from params import *
import time


np.random.seed(seed)
model = CircuitSimulator(m=m, d_m=d_m, h=h, d_ff=d_ff, rn=np.random.randn)
opt = Adam(lr=lr, max_grad=max_grad)

# p_e = np.exp(np.random.randn(*[m for _ in range(Nq)]))
# p_e /= np.sum(p_e)

p_e = np.array([[1/6], [1/3], [1/6], [1/3]]) * np.array([[1/3, 1/6, 1/6, 1/3]])
print(p_e)

print('-' * 10)
print(f'{Nq=}\n{Ns=}\n{seed=}\n{d_m=}\n{h=}\n{d_ff=}\n{lr=}\n{max_epoch=}')
print('-' * 10)

kl_div_list: list[float] = []
l1_norm_list: list[float] = []
classical_fidelity_list: list[float] = []
t1 = time.time()
for epoch in range(max_epoch):
    t2 = time.time()
    kl_div, l1_norm, classical_fidelity = model.forward(batch=Ns, n=Nq, p_e=p_e)
    kl_div_list.append(kl_div)
    l1_norm_list.append(l1_norm)
    classical_fidelity_list.append(1 - classical_fidelity)
    model.backward()
    opt.update(model.params, model.grads)
    t3 = time.time()
    if epoch % 20 == 19:
        print(f'epoch {epoch + 1: >3} | KL div {kl_div: > 4.2e} | L1 norm {l1_norm:4.2e} | {t3 - t2:4.1f} [s] (total {round(t3 - t1)} [s])')

ax = plt.axes()
ax.set_axisbelow(True)
ax.grid()
ax.set_title(f'{Ns=}')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_yscale('log')
ax.scatter(range(max_epoch), kl_div_list, label='KL div', s=8)
ax.scatter(range(max_epoch), l1_norm_list, label='L1 norm', s=8)
ax.scatter(range(max_epoch), classical_fidelity_list, label='1 - Fc', s=8)
ax.legend()
plt.show()

# print(kl_div_list)
# print(l1_norm_list)
if input('save?(yes/no): ') == 'yes':
    filename = input('filename: ') or None
    model.save_params(filename)
