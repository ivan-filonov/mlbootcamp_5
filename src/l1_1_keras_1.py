#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

import state
public_score = 0.5445531
state = state.State('l1_1_keras_1')
import data_raw as data

def tf_force_cpu(f):
    def ff(*args, **kwargs):
        import tensorflow as tf
        with tf.device('/cpu:0'):
            return f(*args, **kwargs)
    return ff

def build_keras_fit_callbacks(model_path):
    from keras import callbacks
    return [
           callbacks.EarlyStopping(
                  monitor='val_loss',
                  patience=20
                  #verbose=1
                   ),
           callbacks.ModelCheckpoint(
                  model_path,
                  monitor='val_loss',
                  save_best_only=True,
                  save_weights_only=True,
                  verbose=0
                   ),
           callbacks.ReduceLROnPlateau(
                  monitor='val_loss',
                  min_lr=1e-7,
                  factor=0.2,
                  verbose=0
                   )
           ]

def keras_limit_mem():
    from keras import backend as K
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))#@tf_force_cpu

def run(train, y, test, v, z):
    #cname = sys._getframe().f_code.co_name
    from keras import layers
    from keras import models
    from keras import optimizers
    cname = 'p'
    train.drop('id', axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)
    num_splits = 9
    scaler = preprocessing.RobustScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    input_dims = train.shape[1]
    def build_model():
        input_ = layers.Input(shape=(input_dims,))
        model = layers.Dense(256, kernel_initializer='Orthogonal')(input_)
        #model = layers.BatchNormalization()(model)
        #model = layers.advanced_activations.PReLU()(model)
        model = layers.Activation('selu')(model)
        #model = layers.Dropout(0.7)(model)

        model = layers.Dense(64, kernel_initializer='Orthogonal')(model)
        #model = layers.BatchNormalization()(model)
        model = layers.Activation('selu')(model)
        #model = layers.advanced_activations.PReLU()(model)
        #model = layers.Dropout(0.9)(model)

        model = layers.Dense(16, kernel_initializer='Orthogonal')(model)
        #model = layers.BatchNormalization()(model)
        model = layers.Activation('selu')(model)
        #model = layers.advanced_activations.PReLU()(model)

        model = layers.Dense(1, activation='sigmoid')(model)

        model = models.Model(input_, model)
        model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Nadam())
        #print(model.summary(line_length=120))
        return model
    batch_size = 128
    np.random.seed(1234)
    build_model().summary(line_length=120)
    ss = model_selection.ShuffleSplit(n_splits=num_splits, random_state=11, test_size=1/num_splits)
    scores = list()
    model_path = state.temp_name('keras_mlp_weights')
    v[cname] = 0
    z[cname] = 0
    for n, (itrain, ival) in enumerate(ss.split(train, y)):
        xtrain, xval = train[itrain], train[ival]
        ytrain, yval = y[itrain], y[ival]
        model = build_model()
        model.fit(
                xtrain, ytrain,
                batch_size = batch_size,
                epochs = 10000,
                validation_data = (xval, yval),
                verbose = 0,
                callbacks = build_keras_fit_callbacks(model_path),
                shuffle = True
            )
        model.load_weights(model_path)
        p = model.predict(xval)
        v.loc[ival, cname] += p.ravel()
        score = metrics.log_loss(y[ival], p)
        print(cname, 'fold %d: '%(n+1), score, state.now())
        scores.append(score)
        z[cname] += model.predict(test).ravel()
        del model
        for i in range(3): gc.collect(i)
    state.drop_temp(model_path)
    cv=np.mean(scores)
    z[cname] /= num_splits
    z['y'] = z[cname]

    return cv, None

def predict():
    saved = state.load('model')
    #saved = None
    if saved == None:
        train, y, test, _ = data.get()
        z = pd.DataFrame()
        z['id'] = test.id
        z['y'] = 0

        v = pd.DataFrame()
        v['id'] = train.id
        v['y'] = y
        cv, _ = run(train, y, test, v, z)
        state.save('model', (v, z, cv, None))
    else:
        v, z, cv, _ = saved
    return v, z, cv, _

if '__main__' == __name__:
    print('starting', state.now())
    v, z, cv, _ = predict()
    state.save_model(v, z, cv)
    if public_score == None:
        state.save_predicts(z)
    else:
        import os
        if os.path.exists('../model_scores.csv'):
            mdf = pd.read_csv('../model_scores.csv')
        else:
            mdf = pd.DataFrame(columns=['timestamp', 'model', 'cv', 'cv std', 'public score'])
        idx = mdf.model == state.base_name_
        if np.sum(idx) == 0:
            mdf.loc[len(mdf), 'model'] = state.base_name_
            idx = mdf.model == state.base_name_
        if (mdf.ix[idx, 'public score'] != public_score).bool():
            mdf.ix[idx, 'public score'] = public_score
            mdf.ix[idx, 'timestamp'] = state.now()
            mdf.ix[idx, 'cv'] =  np.mean(cv)
            mdf.ix[idx, 'cv std'] = np.std(cv)
        mdf.to_csv('../model_scores.csv', index=None)
    print('done.', state.now())
