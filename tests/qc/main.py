import matplotlib.pyplot as plt
from transformer.common.np import np
from transformer.common import Adam
from model import CircuitSimulator
from params import *

np.set_printoptions(precision=2, sign=' ', floatmode='fixed')
np.random.seed(seed)
model = CircuitSimulator(m=m, d_m=d_m, h=h, d_ff=d_ff, rn=np.random.randn)
opt = Adam(lr=lr)

# p_e = np.exp(np.random.randn(*[m for _ in range(Nq)]))
# p_e /= np.sum(p_e)

p_e = np.array([[1/6], [1/3], [1/6], [1/3]]) * np.array([[1/3, 1/6, 1/6, 1/3]])
print(p_e)

loss_list: list[float] = []

for epoch in range(max_epoch):
    loss = model.forward(batch=Ns, n=Nq, p_e=p_e)
    loss_list.append(loss)
    model.backward()
    opt.update(model.params, model.grads)
    print(f'epoch {epoch + 1} | loss {loss}')

plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yscale('log')
plt.plot(loss_list)
plt.show()

print(loss_list)
model.save_params()
