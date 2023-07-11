from .encoder import Encoder
from .decoder import Decoder
from .positional_encoding import positional_encoding
from .common.np import np
from .common.layers import SoftmaxWithLoss, MatMul, Embedding, Dropout
from .common.base_model import BaseModel

rn = np.random.randn

class Transformer(BaseModel):
    def __init__(self, d_m: int, h: int, d_ff: int, vocab_size: int,
                 enc_rep: int, dec_rep: int, p_drop_embed: float,
                 p_drop_sublayer: float, pe_interval: float,
                 norm_positionwise=False, rn=rn):
        super().__init__()
        self.embed = Embedding(vocab_size, d_m, rn)
        self.dropout_enc = Dropout(p_drop_embed)
        self.dropout_dec = Dropout(p_drop_embed)
        self.enc = Encoder(d_m, h, d_ff, enc_rep, p_drop_sublayer, norm_positionwise, rn)
        self.dec = Decoder(d_m, h, d_ff, dec_rep, p_drop_sublayer, norm_positionwise, rn)
        self.matmul = MatMul((d_m, vocab_size), rn)
        self.softmax = SoftmaxWithLoss(e_ls=0.01)

        self.vocab_size = vocab_size
        self.pe_interval = pe_interval

        self.params += self.embed.params + self.enc.params + self.dec.params + self.matmul.params
        self.grads += self.embed.grads + self.enc.grads + self.dec.grads + self.matmul.grads
    
    def forward(self, x_enc: np.ndarray, x_dec: np.ndarray):
        '''
        x_enc: N x n
        x_dec: N x m
        '''
        vs = self.vocab_size
        N, n = x_enc.shape
        _, m = x_dec.shape
        self.N = N
        self.m = m
        x_embedded = self.embed.forward(np.hstack((x_enc, x_dec)))
        x_enc_embedded = x_embedded[:, :n, :]
        x_dec_embedded = x_embedded[:, n:, :]
        x_enc_embedded = positional_encoding(x_enc_embedded, self.pe_interval)
        x_dec_embedded = positional_encoding(x_dec_embedded, self.pe_interval)
        x_enc_embedded = self.dropout_enc.forward(x_enc_embedded)
        x_dec_embedded = self.dropout_dec.forward(x_dec_embedded)
        hs = self.enc.forward(x_enc_embedded)
        y = self.dec.forward(x_dec_embedded, hs)
        y = self.matmul.forward(y)
        loss = self.softmax.forward(
            y.reshape((N * m, vs)),
            np.roll(np.eye(vs)[x_dec], -1, 1).reshape((N * m, vs))
        )
        correct_count: np.ndarray = (
            y.argmax(-1) == np.roll(np.asarray(x_dec), -1, 1)
        ).all(axis=-1).sum()
        return loss, correct_count.item()
    
    def backward(self, dout=None):
        N = self.N
        m = self.m
        vs = self.vocab_size
        dout = self.softmax.backward().reshape((N, m, vs))
        dout = self.matmul.backward(dout)
        dx_dec, dhs = self.dec.backward(dout)
        dx_enc = self.enc.backward(dhs)
        dx_enc = self.dropout_enc.backward(dx_enc)
        dx_dec = self.dropout_dec.backward(dx_dec)
        dx = np.hstack((dx_enc, dx_dec))
        self.embed.backward(dx)
        return None
    
    def generate(self, x_enc: np.ndarray, start_id: int, length: int):
        '''
        x_enc: N x n array[int]
        '''
        N, n = x_enc.shape
        x_dec = np.full((N, length), start_id, dtype=int)
        
        for i in range(length - 1):
            x_encoded = self.embed.forward(np.hstack((x_enc, x_dec)))
            x_enc_encoded = x_encoded[:, :n, :]
            x_dec_encoded = x_encoded[:, n:, :]
            x_enc_encoded = positional_encoding(x_enc_encoded, self.pe_interval)
            x_dec_encoded = positional_encoding(x_dec_encoded, self.pe_interval)
            x_enc_encoded = self.dropout_enc.forward(x_enc_encoded, False)
            x_dec_encoded = self.dropout_dec.forward(x_dec_encoded, False)
            hs = self.enc.forward(x_enc_encoded, False)
            y = self.dec.forward(x_dec_encoded, hs, False)
            y = self.matmul.forward(y)
            x_dec[:, i + 1] = np.argmax(y[:, i], axis=-1)

        return x_dec
