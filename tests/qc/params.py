from transformer.config import gpu_config
gpu_config(False)

m = 4
'''number of POVM elements'''
Nq = 2
'''number of qubits'''
Ns = 2**9
'''number of samples'''
seed = 0
'''random seed for parameter initialization'''
d_m = 16
'''d_model of Transformer'''
h = 2
'''number of heads in Attention layer'''
d_ff = 8
'''size of hidden layer of Position-wise FFN'''
lr = 0.01
'''initial learning rate'''
max_grad = 10.
'''maximum gradient'''
max_epoch = 800
'''maximum epoch'''
