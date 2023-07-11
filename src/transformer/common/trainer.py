import numpy
import time
from .np import np
from .util import clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20, epoch=0):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for _ in range(max_epoch):
            # シャッフル
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x[idx]
            t = t[idx]
            train_acc = 0

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # 勾配を求め、パラメータを更新
                res = model.forward(batch_x, batch_t)
                loss, correct_count = res if type(res) == tuple else (res, 0)
                train_acc += correct_count
                model.backward()
                params, grads = model.params, model.grads
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 評価
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print(
                        f'| epoch {self.current_epoch + 1} | iter {iters + 1} / {max_iters}',
                        f'| time {round(elapsed_time, 2)}[s] | loss {round(avg_loss, 3)}',
                        f'| acc. {correct_count} / {batch_size} |')
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            print(f'train data accuracy: {round(train_acc / data_size * 100, 3)}%')
            self.current_epoch += 1
