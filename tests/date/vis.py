from transformer.transformer import Transformer
from transformer.common.np import np
from data import load_data, get_vocab
from params import *
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_data()
char_to_id, id_to_char = get_vocab()
vocab_size = len(char_to_id)

model = Transformer(d_m, h, d_ff, vocab_size, enc_rep, dec_rep, p_drop_embed,
                    p_drop_sublayer, pe_interval, np.random.randn)

model.load_params('log/202309021105.pkl')

while True:
    idx = int(input())
    x = x_test[idx:idx+1]
    y = model.generate(x, 14, 11)
    x_str = ''.join([id_to_char[c] for c in x[0]])
    y_str = ''.join([id_to_char[c] for c in y[0]])[1:] + ' '
    print(x_str)
    print(y_str)
    layer = model.dec.layers[0][1].layer
    for i, head in enumerate(layer.heads):
        weight = head.attention_weight[0]
        ax = plt.subplot(211 + i, title=r'head$_{}$'.format(i+1))
        ax.imshow(weight)
        ax.set_xticks(range(29))
        ax.set_xticklabels([*x_str])
        ax.set_yticks(range(11))
        ax.set_yticklabels([*y_str], rotation=-90, va='center')
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.show()