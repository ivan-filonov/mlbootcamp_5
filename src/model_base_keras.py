#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc

import numpy as np

from keras import backend as K
from keras import callbacks

from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

def tf_force_cpu(f):
    def ff(*args, **kwargs):
        with K.tf.device('/cpu:0'):
            return f(*args, **kwargs)
    return ff

def build_keras_fit_callbacks(model_path):
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
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))#@tf_force_cpu

import model_base

class KerasModelBase(model_base.Model):
    def __init__(self, name,
                 data_source,
                 features,
                 debug = False,
                 public_score = None,
                 batch_size = 128,
                 num_splits = 9):
        super().__init__(name, data_source, features, debug, public_score)
        self.batch_size_ = batch_size
        self.num_splits_ = num_splits

    def build_keras_model(self):
        ''' #example:
        from keras import layers
        from keras import models
        from keras import optimizers
        input_ = layers.Input(shape=(self.input_dims_,))
        model = layers.Dense(256, kernel_initializer='Orthogonal')(input_)
        #model = layers.BatchNormalization()(model)
        model = layers.Activation('selu')(model)
        #model = layers.noise.AlphaDropout(0.2, seed=1)(model)
        #model = layers.advanced_activations.PReLU()(model)
        #model = layers.Dropout(0.4)(model)

        model = layers.Dense(64, kernel_initializer='Orthogonal')(model)
        #model = layers.BatchNormalization()(model)
        model = layers.Activation('selu')(model)
        #model = layers.noise.AlphaDropout(0.1, seed=1)(model)
        #model = layers.advanced_activations.PReLU()(model)
        #model = layers.Dropout(0.4)(model)

        model = layers.Dense(16, kernel_initializer='Orthogonal')(model)
        #model = layers.BatchNormalization()(model)
        model = layers.Activation('selu')(model)
        #model = layers.advanced_activations.PReLU()(model)

        model = layers.Dense(1, activation='sigmoid')(model)

        model = models.Model(input_, model)
        model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Nadam())
        #print(model.summary(line_length=120))
        return model
        '''
        raise Exception('implement this!')

    #@tf_force_cpu
    def model(self):
        #cname = sys._getframe().f_code.co_name
        cname = 'keras'
        train, y, test = self.train_, self.y_, self.test_

        np.random.seed(1234)
        train.drop('id', axis=1, inplace=True)
        test.drop('id', axis=1, inplace=True)

        from sklearn import pipeline
        pipe = pipeline.make_pipeline(preprocessing.Imputer(),
                                      preprocessing.RobustScaler())

        train = pipe.fit_transform(train)
        test = pipe.transform(test)

        self.input_dims_ = train.shape[1]
        def build_model():
            return self.build_keras_model()
        batch_size = self.batch_size_
        build_model().summary(line_length=120)
        ss = model_selection.StratifiedKFold(n_splits = self.num_splits_,
                                             random_state = 11,
                                             shuffle = True)
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
        z[cname] /= self.num_splits_
        z['y'] = z[cname]

        return cv, None

if '__main__' == __name__:
    pass