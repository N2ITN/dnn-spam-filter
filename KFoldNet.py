import time
from random import shuffle

import numpy as np
# from get_vecs import load_pickle, names, one_pic
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Activation, Convolution1D, Dense, Dropout, Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.regularizers import l2
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
import os


class kF:

    def __new__(
        self,
        X,
        y,
        model,
        tsv_,
        meta_=False,
        k=5,
        predict=None,
        newModel=False,
        kf=True,
        train=True,
        iterations=20
    ):
        kF.newModel = newModel
        kF.tsv_ = tsv_
        kF.X = X
        kF.y = y
        kF.model = model
        kF.k = k
        kF.predict = predict
        kF.iterations = iterations
        if meta_:
            return meta()
        if train == False:
            return predictor()
        if kf:
            return run_kfold()
        else:
            return train_model()


def kFold():
    k = kF.k

    n = kF.y.shape[0]
    split = n // k
    resid = n % k

    chunks = [j for j in range(k)]
    shuffle(chunks)
    for i in chunks:
        start = i * split
        end = (i + 1) * split
        yield start, end


def subdivide(start, end, X, y):

    X_test = X[start:end]
    first = X[:start]
    second = X[end:]
    X_train = np.vstack((first, second))
    y_test = y[start:end]
    first = y[:start]
    second = y[end:]
    y_train = np.vstack((first, second))
    return X_train, y_train, X_test, y_test


def run_kfold():
    for start, end in kFold():
        X_train, y_train, X_test, y_test = subdivide(
            start, end, kF.X, kF.y
        )
        train_model(X_train, y_train, X_test, y_test, kF.model)


def train_model():
    import os
    import tensorflow as tf

    LOG_DIR = './logs'
    filepath = "weights.best.hdf5"
    if kF.newModel == False:
        if not os.path.exists(filepath):
            print(print("model not found, writing new"))
        else:
            try:
                kF.model.load_weights(filepath)
                print("model loaded")
            except ValueError:
                print(
                    'model mismatch. did a dimension change? overwriting'
                )
    now = time.strftime("%c")
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        mode='auto'
    )
    tensorboard = TensorBoard(
        log_dir='./logs/' + now,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        embeddings_freq=1,
        # embeddings_layer_names=['embedding_1', 'dense_1'],
        embeddings_metadata='./logs/' + now + '/metadata.tsv'
    )

    callbacks_list = [checkpoint, tensorboard]
    kF.model.fit(
        kF.X,
        kF.y,
        epochs=kF.iterations,
        batch_size=10,
        shuffle='batch',
        validation_split=0.2,
        callbacks=callbacks_list,
        verbose=1
    )

    with open('./logs/' + now + '/metadata.tsv', 'w') as m:

        [m.write(t) for t in kF.tsv_]
    predictor()


def meta():

    from keras import backend as K
    kF.model.load_weights("weights.best.hdf5")
    # all new operations will be in test mode from now on
    K.set_learning_phase(0)

    # serialize the model and get its weights, for quick re-building
    config = kF.model.get_config()
    weights = kF.model.get_weights()

    # re-build a model where the learning phase is now hard-coded to 0
    from keras.models import model_from_config, Sequential

    from tensorflow.contrib.tensorboard.plugins import projector
    import tensorflow as tf
    new_model = Sequential.from_config(config)
    new_model.set_weights(weights)
    # embedding_var = tf.Variable(
    #     tf.random_normal(kF.y.shape), name='word_embedding')
    embed = next(
        filter(lambda x: x.name == 'dense_1', kF.model.layers)
    )
    tensor_name = embed.name

    config = projector.ProjectorConfig()
    saver = tf.train.Saver(sharded=True)
    LOG_DIR = './logs'
    # [print(i) for i in dir(K)]
    # print()
    # [print(i) for i in dir(embed.embeddings)]
    # exit()
    print(kF.model.layers[0].get_weights()[0][0])

    embed2 = embed.embeddings

    saver.save(K.get_session(), os.path.join(LOG_DIR, "model.ckpt"))

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = tensor_name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    # return config 

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(LOG_DIR, embedding)

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)


def predictor():

    w = kF.model.load_weights("weights.best.hdf5")
    if not kF.predict:
        y = kF.y
        _x_test = kF.X
    else:
        y, _x_test = kF.predict

    y = np.rot90(y)[0]
    pred = kF.model.predict_classes(_x_test).tolist()
    print(pred)
    assert len(pred) == len(y)

    res = dict.fromkeys(['falsePos', 'falseNeg', 'correct'], 0)
    for i, x in enumerate(y):
        print(pred[i], y[i])
        if pred[i] != y[i]:
            if pred[i] == 0:
                res['falsePos'] += 1
            else:
                res['falseNeg'] += 1
        else:
            res['correct'] += 1
    print(res['correct'] / len(y))
    print(res)
    # meta()
    return kF.model
