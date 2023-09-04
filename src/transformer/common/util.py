import os
from .np import np

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def ids_to_str(ids: np.ndarray, dic: dict[int, str]):
    return ''.join([dic[int(c)] for c in ids])


def eval_seq2seq(model, questions: np.ndarray,
                 answers: np.ndarray, id_to_char: dict[int, str],
                 verbose=10):
    assert questions.shape[0] == answers.shape[0]
    # 頭の区切り文字
    start_id = answers[0, 0]
    guesses = model.generate(questions, start_id, len(answers[0]))

    # 文字列へ変換
    questions = [ids_to_str(q, id_to_char) for q in questions]
    answers = [ids_to_str(a, id_to_char) for a in answers]
    guesses = [ids_to_str(g, id_to_char) for g in guesses]

    colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
    for i in range(verbose):
        q = questions[i]
        a = answers[i]
        g = guesses[i]
        print('Q', q)
        print('T', a)

        is_windows = os.name == 'nt'
        if a == g:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'O'
            print(mark + ' ' + g)
        else:
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
            print(mark + ' ' + g)
        print('---')

    correct_count = len(list(filter(lambda x: x[0] == x[1], zip(answers, guesses))))

    return correct_count
