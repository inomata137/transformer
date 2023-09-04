from transformer.transformer import Transformer
from transformer.common import Adam, Trainer
from transformer.common.np import np
from transformer.common.util import eval_seq2seq
from data import load_data, get_vocab
from params import *

(x_train, t_train), (x_test, t_test) = load_data()
char_to_id, id_to_char = get_vocab()
vocab_size = len(char_to_id)
'''number of unique characters'''

np.random.seed(seed)

model = Transformer(d_m, h, d_ff, vocab_size, enc_rep, dec_rep, p_drop_embed,
                    p_drop_sublayer, pe_interval, np.random.randn)

optimizer = Adam(lr)
trainer = Trainer(model, optimizer)

print('-' * 10)
print(f'seed = {seed}')
print(f'd_m = {d_m}')
print(f'd_ff = {d_ff}')
print(f'h = {h}')
print(f'enc_rep = {enc_rep}')
print(f'dec_rep = {dec_rep}')
print(f'batch_size = {batch_size}')
print(f'max_grad = {max_grad}')
print(f'p_drop_embed = {p_drop_embed}')
print(f'p_drop_sublayer = {p_drop_sublayer}')
print(f'lr = {lr}')
print('-' * 10)

acc_list = []

for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = eval_seq2seq(model, x_test, t_test, id_to_char)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print(f'val acc {round(acc * 100, 3)}%')

print(acc_list)


if input('save?(yes/no): ') == 'yes':
    filename = input('filename: ') or None
    model.save_params(filename)