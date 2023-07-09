from transformer.config import gpu_config
gpu_config(False)

m = 4
'''number of POVM elements'''
Nq = 2
'''number of qubits'''
Ns = 4096
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
max_epoch = 50
'''maximum epoch'''
