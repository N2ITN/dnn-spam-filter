#%%

import random

import keras
import numpy as np
from model_ import define_model
from keras.preprocessing.text import Tokenizer
import os
from KFoldNet import kF
import to_json

_in = to_json.get_json('INBOX')
_rc = to_json.get_json('RCRT')

num_classes = 2
num_words = 1000


def words():
    inbox = [[1, j[0]] for j in _in]
    recruit = [[0, j[0]] for j in _rc]

    texts_combo = inbox + recruit
    rng = np.random.RandomState(666)
    rng.shuffle(texts_combo)

    _y = [t[0] for t in texts_combo]
    _x = [t[1] for t in texts_combo]
    T = Tokenizer(num_words=num_words, lower=True, split=" ")
    T.fit_on_texts(_x)

    wcounts = list(T.word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts][:num_words]
    sorted_ct = [wc[1] for wc in wcounts][:num_words]

    print(wcounts[0])

    tsv_ = ['name' + '\t' + 'rank' + '\t' + 'count' + '\n'] + [
        ''.join([c[0], '\t', str(n + 1), '\t', str(c[1]), '\n'])
        for n, c in enumerate(wcounts)
    ]

    # tsv_ = [''.join([i[0], '\t', str(i[1]), '\n']) for i in wcounts]

    X = T.texts_to_matrix(_x, mode='tfidf')

    y = keras.utils.to_categorical(_y, num_classes)
    # print(sorted(T.word_counts.items(), key=lambda x: x[1])[-100:])
    return X, y, tsv_[:num_words + 1]


def miniTokenizer(_x, k='subject'):
    T = Tokenizer(num_words=num_words)
    text = [d[k] for d in _x]
    T.fit_on_texts(text)
    X = T.texts_to_matrix(text, mode='tfidf')
    return X


def metadata():
    inbox = [(1, i) for i in _in]
    recruit = [(0, i) for i in _rc]
    texts_combo = inbox + recruit
    _y = [t[0] for t in texts_combo]

    _x = [t[1] for t in texts_combo]
    tok = miniTokenizer(_x)

    X = np.array((
        list(
            zip([d['hour'] for d in _x], [d['weekday'] for d in _x])
        )
    ))

    X = np.concatenate((tok, X), axis=1)

    y = keras.utils.to_categorical(_y, num_classes)
    return np.asarray(X), y


X, y, tsv_ = words()

# print(X.shape)
X_tr = X[10:]
X_te = X[:10]

y_tr = y[10:]
y_te = y[:10]

print(X.shape[-1], y.shape[0])

model = define_model(X_tr.shape[-1], num_words)

if __name__ == '__main__':
    w = previous_model = kF(
        X_tr,
        y_tr,
        model,
        tsv_=tsv_,
        newModel=True,
        kf=False,
        meta_=False,
        predict=(y_te, X_te),
        train=True,
        iterations=20
    )

#%%
# dir(w)
#%%
# x = w.layers[0].embeddings
#%%
# dir(x)

#%%
# x.name

# emb = next(y for y in w.layers if y.name == 'embedding_1')
# print(dir(emb), '\n' * 3, vars(emb))
# print(emb.get_weights()[0])
