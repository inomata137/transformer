d_m = 16
'''d_model'''
wordvec_size = d_m
'''alias of d_model'''
d_ff = 8
'''size of FFN hidden layer'''
h = 2
'''number of attention heads'''
enc_rep = 1
'''number of encoder blocks'''
dec_rep = 1
'''number of decoder blocks'''
max_epoch = 30
'''maximum epochs'''
p_drop_embed = 0.05
'''dropout ratio at embedding'''
p_drop_sublayer = 0.1
'''dropout ratio at each sublayer'''
pe_interval = 30
'''positional encoding interval'''
lr = 0.005
'''initial learning rate'''
max_grad = 10.0
'''maximum gradient'''
batch_size = 128
'''batch size'''
seed = 2023
'''random seed of params initialization'''
