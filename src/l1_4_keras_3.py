#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

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

import model_base

import data_clean_1 as data
import features_text_2 as fea_1
import features_misc_2 as fea_2

class model(model_base.Model):
    def __init__(self):
        name = 'l1_4_keras_3'
        debug = False
        public_score = None
        super().__init__(name, data, [fea_1, fea_2], debug, public_score)

    def model(self):
        #cname = sys._getframe().f_code.co_name
        cname = 'p'
        train, y, test = self.train_, self.y_, self.test_

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
            model = layers.BatchNormalization()(model)
            #model = layers.advanced_activations.PReLU()(model)
            model = layers.Activation('selu')(model)
            #model = layers.Dropout(0.7)(model)

            model = layers.Dense(256, kernel_initializer='Orthogonal')(model)
            #model = layers.BatchNormalization()(model)
            model = layers.Activation('selu')(model)
            #model = layers.advanced_activations.PReLU()(model)
            #model = layers.Dropout(0.9)(model)

            model = layers.Dense(128, kernel_initializer='Orthogonal')(model)
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
        ss = model_selection.StratifiedKFold(n_splits=num_splits, random_state=11, shuffle=True)
        scores = list()
        model_path = self.temp_name('keras_mlp_weights')
        v, z = self.v_, self.z_
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
            if score != score:
                raise Exception('NaN score!!!')
            print(cname, 'fold %d: '%(n+1), score, self.now())
            scores.append(score)
            z[cname] += model.predict(test).ravel()
            del model
            for i in range(3): gc.collect(i)
        print('scores:', scores, np.mean(scores), np.std(scores))
        self.drop_temp(model_path)
        cv=np.mean(scores)
        z[cname] /= num_splits
        z['y'] = z[cname]

        return cv, None

def predict():
    return model().predict()

if '__main__' == __name__:
    m = model()
    print('starting', m.now())
    m.main()
    print('done.', m.now())