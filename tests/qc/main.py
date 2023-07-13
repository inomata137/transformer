import matplotlib.pyplot as plt
from transformer.common.np import np
from transformer.common import Adam
from model import CircuitSimulator
from params import *
import time


np.random.seed(seed)
model = CircuitSimulator(m=m, d_m=d_m, h=h, d_ff=d_ff, rn=np.random.randn)
opt = Adam(lr=lr)

# p_e = np.exp(np.random.randn(*[m for _ in range(Nq)]))
# p_e /= np.sum(p_e)

p_e = np.array([[1/6], [1/3], [1/6], [1/3]]) * np.array([[1/3, 1/6, 1/6, 1/3]])
print(p_e)

loss_list: list[float] = []
t1 = time.time()
for epoch in range(max_epoch):
    t2 = time.time()
    loss = model.forward(batch=Ns, n=Nq, p_e=p_e)
    loss_list.append(loss)
    model.backward()
    opt.update(model.params, model.grads)
    t3 = time.time()
    print(f'epoch {epoch + 1: 3} | loss {loss:6.4f} | {t3 - t2:5.2f} [s] (total {round(t3 - t1)} [s])')

plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yscale('log')
plt.scatter(range(max_epoch), loss_list)
plt.show()

print(loss_list)
if input('save?(yes/no): ') == 'yes':
    filename = input('filename: ') or None
    model.save_params(filename)
