#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 20:26:49 2017

@author: user
"""
import datetime
import os
import sys

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from sklearn import cluster
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import preprocessing

from keras import backend as K
from keras import callbacks
from keras import layers
from keras import models
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor

def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def csv_name_suffix():
    script_name = os.path.basename(sys.argv[0])
    script_name = script_name[script_name.index('_'):-3]
    return script_name + '.csv'

def weighted_average_dict(data, col, real_target, alpha=10):
    '''
    data - DataFrame with source data

    col - feature to process

    real_target - target to compute weighted average per selected feature value

    alpha - averaging coeff

    returns Series with weighted average of real target
    '''
    g = data.groupby(col)
    KK = g.size()
    mean_rt = g[real_target].mean()
    global_mean = data[real_target].mean()
    c = (mean_rt * KK + global_mean * alpha)/(KK + alpha)
    d = c.to_dict()
    del c, g, KK, mean_rt, global_mean
    return d

PREDICT_LOG_BIAS = 1e-5
def pconvert(c):
    return np.log(c + PREDICT_LOG_BIAS)

def prestore(c):
    return np.exp(c) - PREDICT_LOG_BIAS

def cleanup_and_generate(train, y, test, use_hand_labels = False):
    ntrain = len(train)
    df_all = pd.concat([train, test]).reset_index(drop=True)

    for c in ['active', 'alco', 'smoke']:
        replacement = test.ix[test[c] != 'None', c].astype('float32').mean()
        df_all.ix[df_all[c]=='None', c] = replacement
        df_all[c] = df_all[c].astype('float32')
    del c, replacement

    base_columns = list(set(df_all.columns) - {'id', 'y'})

    df_all['bad_height'] = (df_all.height < 130).values * 1
    df_all['bad_weight'] = (df_all.weight + 120 < df_all.height).values * 1

    # now cleanup height and weight
    df_all['r_height'] = df_all.height
    df_all['r_weight'] = df_all.weight
    df_all.ix[df_all.height < 95, 'r_height'] += 100
    df_all.ix[df_all.bad_weight > 0, 'r_weight'] += 100
    df_all.ix[(df_all.height < 100).values * (df_all.weight > 90).values, 'r_height'] += 100

    df_all['BWI'] = df_all.weight / (df_all.height / 100) / (df_all.height / 100)
    df_all['bad_bwi'] = (df_all.BWI > 60).values * 1 + (df_all.BWI < 10).values * 1
    df_all['r_BWI'] = df_all.r_weight / (df_all.r_height / 100) / (df_all.r_height / 100)

    df_all['bad_ap_hi'] = 0
    df_all.ix[(df_all.ap_hi < 80).values + (df_all.ap_hi > 220).values, 'bad_ap_hi'] = 1

    df_all['bad_ap_lo'] = 0
    df_all.ix[(df_all.ap_lo < 40).values + (df_all.ap_lo > 200).values, 'bad_ap_lo'] = 1

    df_all['r_ap_hi'] = np.abs(df_all.ap_hi)
    df_all['r_ap_lo'] = np.abs(df_all.ap_lo)

    # now cleanup ap_hi, ap_lo
    df_all['c1'] = (df_all.r_ap_hi > 250)
    df_all.ix[df_all.r_ap_hi > 250, 'r_ap_hi'] /= 10

    df_all['c2'] = (df_all.r_ap_hi > 250)
    df_all.ix[df_all.r_ap_hi > 250, 'r_ap_hi'] /= 10

    df_all['c3'] = (df_all.r_ap_hi < 25)
    df_all.ix[df_all.r_ap_hi < 25, 'r_ap_hi'] *= 10

    df_all['c4'] = (df_all.r_ap_hi < 25)
    df_all.ix[df_all.r_ap_hi < 25, 'r_ap_hi'] *= 10

    df_all['c5'] = (df_all.r_ap_lo > 250)
    df_all.ix[df_all.r_ap_lo > 250, 'r_ap_lo'] /= 10

    df_all['c6'] = (df_all.r_ap_lo > 250)
    df_all.ix[df_all.r_ap_lo > 250, 'r_ap_lo'] /= 10

    df_all['c7'] = (df_all.r_ap_lo < 25)
    df_all.ix[df_all.r_ap_lo < 25, 'r_ap_lo'] *= 10

    df_all['c8'] = (df_all.r_ap_lo > 10000)
    df_all.ix[df_all.r_ap_lo > 10000, 'r_ap_lo'] /= 100

    df_all['c9'] = df_all.r_ap_hi - df_all.r_ap_lo < 10
    df_all['t1'] = np.maximum(df_all.r_ap_hi, df_all.r_ap_lo)
    df_all['t2'] = np.minimum(df_all.r_ap_hi, df_all.r_ap_lo)
    df_all.r_ap_hi = df_all.t1
    df_all.r_ap_lo = df_all.t2
    df_all.drop(['t1', 't2'], axis=1, inplace=True)

    df_all['ap_diff'] = df_all.ap_hi - df_all.ap_lo
    df_all['r_ap_diff'] = df_all.r_ap_hi - df_all.r_ap_lo

    df_all['bad_data_count'] = (df_all.bad_bwi + df_all.bad_height + df_all.bad_weight + df_all.bad_ap_hi + df_all.bad_ap_lo).values
    df_all['has_bad_data'] = df_all.bad_data_count > 0

    df_all['w_div_h'] = df_all.weight / df_all.height
    df_all['h_sub_w'] = df_all.height - df_all.weight
    df_all['ap_hi_sub_w'] = df_all.ap_hi - df_all.weight

    df_all['r_w_div_h'] = df_all.r_weight / df_all.r_height
    df_all['r_h_sub_w'] = df_all.r_height - df_all.r_weight
    df_all['r_ap_hi_sub_w'] = df_all.r_ap_hi - df_all.r_weight

    df_all['age_months'] = df_all.age // 30
    df_all['age_years'] = df_all.age // 365

    for ncl in range(2, 30):
        clf = cluster.KMeans(n_clusters=ncl, random_state=12)
        clf.fit(train[base_columns].values)
        df_all['kmeans_%d'%(ncl)] = clf.predict(df_all[base_columns].values)
        print('added kmeans_%d'%(ncl))

    train2 = df_all[:ntrain].reindex()
    test2 = df_all[ntrain:].reindex()

    #'''
    tr = pd.concat([train2, y], axis=1)
    print('tr:', tr.shape)
    for c in base_columns:
        skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
        cname = c + '_wa'
        train2[cname] = 0
        test2[cname] = 0
        for itrain, ival in skf.split(tr, y):
            wd = weighted_average_dict(tr.ix[itrain], c, 'cardio')
            train2.ix[ival, cname] = tr.ix[ival, c].map(wd)
            test2[cname] += test2[c].map(wd)
        test2[cname] /= skf.n_splits
        train2[cname].fillna(y.mean(), inplace=True)
        test2[cname].fillna(y.mean(), inplace=True)
    #'''
    print('train2.shape:', train2.shape, 'test2 shape:', test2.shape)

    return train2, y, test2

def keras1(train2, y, test2, v, z):
    cname = sys._getframe().f_code.co_name
    v[cname], z[cname] = 0, 0
    scores = list()
    scaler = preprocessing.RobustScaler()
    train3 = scaler.fit_transform(train2)
    test3 = scaler.transform(test2)
    input_dims = train3.shape[1]
    def build_model():
        input_ = layers.Input(shape=(input_dims,))
        model = layers.Dense(int(input_dims * 3.33),
                             kernel_initializer='Orthogonal',
                             activation=layers.advanced_activations.PReLU())(input_)
        model = layers.BatchNormalization()(model)
        #model = layers.Dropout(0.7)(model)
        model = layers.Dense(int(input_dims * 1.35),
                             kernel_initializer='Orthogonal',
                             activation=layers.advanced_activations.PReLU())(model)
        model = layers.BatchNormalization()(model)
        #model = layers.Dropout(0.9)(model)
        model = layers.Dense(int(input_dims * 0.51),
                             kernel_initializer='Orthogonal',
                             activation=layers.advanced_activations.PReLU())(model)
        model = layers.BatchNormalization()(model)
        model = layers.Dense(1,
                             activation='sigmoid')(model)
        model = models.Model(input_, model)
        model.compile(loss = 'binary_crossentropy',
                      optimizer = optimizers.Nadam(),
                      metrics=["accuracy"])
        #print(model.summary(line_length=120))
        return model
    np.random.seed(1234)
    est = KerasRegressor(build_fn=build_model,
                         nb_epoch=10000,
                         batch_size=32,
                         #verbose=2
                        )
    print(build_model().summary(line_length=120))
    model_path = '../data/working/' + csv_name_suffix()
    model_path = model_path[:-4] + '_keras_model.h5'
    kcb = [
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
                  verbose=1
                   )
           ]
    num_splits = 5
    ss = model_selection.ShuffleSplit(n_splits=num_splits, random_state=11)
    for n, (itrain, ival) in enumerate(ss.split(train3, y)):
        xtrain, xval = train3[itrain], train3[ival]
        ytrain, yval = y[itrain], y[ival]
        est.fit(
                xtrain, ytrain,
                epochs=10000,
                validation_data=(xval, yval),
                verbose=0,
                callbacks=kcb,
                shuffle=True
            )
        est.model.load_weights(model_path)
        p = est.predict(xval)
        v.loc[ival, cname] += pconvert(p)
        score = metrics.log_loss(y[ival], p)
        print(cname, 'fold %d: '%(n+1), score, now())
        scores.append(score)
        z[cname] += pconvert(est.predict(test3))
    os.remove(model_path)

    cv=np.array(scores)
    print(cv, cv.mean(), cv.std())
    z[cname] /= num_splits

def xgb1(train2, y, test2, v, z):
    cname = sys._getframe().f_code.co_name
    v[cname], z[cname] = 0, 0
    N_splits = 7
    N_seeds = 7
    scores = []
    skf = model_selection.StratifiedKFold(n_splits=N_splits, shuffle=True)
    xgb_params = dict(
            max_depth = 6,
            learning_rate = 0.05,
            objective = 'binary:logistic',
            eval_metric = 'logloss',
            seed = 1,
            silent = 1
        )
    for s in range(N_seeds):
        xgb_params['seed'] = s + 4242
        for n, (itrain, ival) in enumerate(skf.split(train2, y)):
            print('step %d of %d'%(n+1, skf.n_splits), now())
            dtrain = xgb.DMatrix(train2.ix[itrain], y[itrain])
            dvalid = xgb.DMatrix(train2.ix[ival], y[ival])
            dtest = xgb.DMatrix(test2)
            watch = [(dtrain, 'train'), (dvalid, 'valid')]
            clf = xgb.train(xgb_params, dtrain, 10000, watch, early_stopping_rounds=100, verbose_eval=1000)

            p = clf.predict(dvalid)
            v.loc[ival, cname] += pconvert(p)
            score = metrics.log_loss(y[ival], p)
            z[cname]  += pconvert(clf.predict(dtest))
            print(cname, 'seed %d step %d: '%(xgb_params['seed'], n+1), score, now())
            scores.append(score)

    print('validation loss: ', metrics.log_loss(y, v[cname]))
    cv=np.array(scores)
    print(cv, cv.mean(), cv.std())
    z[cname] /= N_splits * N_seeds
    v[cname] /= N_seeds

def xgb2(train2, y, test2, v, z):
    cname = sys._getframe().f_code.co_name
    v[cname], z[cname] = 0, 0
    N_splits = 7
    N_seeds = 7
    scores = []
    skf = model_selection.StratifiedKFold(n_splits=N_splits, shuffle=True)
    xgb_params = dict(
            max_depth = 5,
            learning_rate = 0.05,
            subsample = 0.7,
            #colsample_bytree = 0.8,
            objective = 'binary:logistic',
            eval_metric = 'logloss',
            seed = 1,
            silent = 1
        )
    dtest = xgb.DMatrix(test2)
    for s in range(N_seeds):
        xgb_params['seed'] = s + 4242
        for n, (itrain, ival) in enumerate(skf.split(train2, y)):
            print('step %d of %d'%(n+1, skf.n_splits), now())
            dtrain = xgb.DMatrix(train2.ix[itrain], y[itrain])
            dvalid = xgb.DMatrix(train2.ix[ival], y[ival])
            watch = [(dtrain, 'train'), (dvalid, 'valid')]
            clf = xgb.train(xgb_params, dtrain, 10000, watch, early_stopping_rounds=100, verbose_eval=1000)

            p = clf.predict(dvalid)
            v.loc[ival, cname] += pconvert(p)
            score = metrics.log_loss(y[ival], p)
            z[cname]  += pconvert(clf.predict(dtest))
            print(cname, 'seed %d step %d: '%(xgb_params['seed'], n+1), score, now())
            scores.append(score)

    print('validation loss: ', metrics.log_loss(y, v[cname]))
    cv=np.array(scores)
    print(cv, cv.mean(), cv.std())
    z[cname] /= N_splits * N_seeds
    v[cname] /= N_seeds

def rf1(train2, y, test2, v, z):
    cname = sys._getframe().f_code.co_name
    v[cname], z[cname] = 0, 0
    scores = list()
    num_seeds = 7
    num_splits = 7
    base_seed = 13
    ss = model_selection.ShuffleSplit(n_splits=num_splits)
    for seed in range(base_seed, base_seed + num_seeds):
        ss = model_selection.ShuffleSplit(n_splits=num_splits, random_state=seed)
        for n, (itrain, ival) in enumerate(ss.split(train2, y)):
            reg = ensemble.RandomForestClassifier(max_depth=9,
                                               random_state=seed,
                                               n_estimators=500,
                                               n_jobs=-2)
            reg.fit(train2[itrain], y[itrain])
            p = reg.predict_proba(train2[ival])[:,1]
            v.loc[ival, cname] += pconvert(p)
            score = metrics.log_loss(y[ival], p)
            print(cname, 'seed %d step %d: '%(seed, n+1), score, now())
            scores.append(score)
            z[cname] += pconvert(reg.predict_proba(test2)[:,1])

    cv=np.array(scores)
    print(cv, cv.mean(), cv.std())
    z[cname] /= num_splits * num_seeds
    v[cname] /= num_seeds

def et1(train2, y, test2, v, z):
    cname = sys._getframe().f_code.co_name
    v[cname], z[cname] = 0, 0
    scores = list()
    num_seeds = 7
    num_splits = 7
    base_seed = 13
    ss = model_selection.ShuffleSplit(n_splits=num_splits)
    for seed in range(base_seed, base_seed + num_seeds):
        ss = model_selection.ShuffleSplit(n_splits=num_splits, random_state=seed)
        for n, (itrain, ival) in enumerate(ss.split(train2, y)):
            reg = ensemble.ExtraTreesClassifier(max_depth=6,
                                               random_state=seed,
                                               n_estimators=500,
                                               n_jobs=-2)
            reg.fit(train2[itrain], y[itrain])
            p = reg.predict_proba(train2[ival])[:,1]
            v.loc[ival, cname] += pconvert(p)
            score = metrics.log_loss(y[ival], p)
            print(cname, 'seed %d step %d: '%(seed, n+1), score, now())
            scores.append(score)
            z[cname] += pconvert(reg.predict_proba(test2)[:,1])

    cv=np.array(scores)
    print(cv, cv.mean(), cv.std())
    z[cname] /= num_splits * num_seeds
    v[cname] /= num_seeds

def save_results(v, z):
    pred_path = '../submissions/p' + csv_name_suffix()
    all_data_path = '../data/output/model' + csv_name_suffix()

    z[['y']].to_csv(pred_path, header=None, index=False)
    print(z.head(20))

    v['train'] = 1
    z['train'] = 0

    q = pd.concat([v, z], axis=0)
    q.to_csv(all_data_path, index=False, compression='gzip')
    print('saved', pred_path, all_data_path)

if '__main__' == __name__:
    print('starting', now())
    np.random.seed(1234)

    train = pd.read_csv('../data/input/train.csv', delimiter=';')
    test = pd.read_csv('../data/input/test.csv', delimiter=';')

    y = train.cardio
    train.drop('cardio', axis=1, inplace=True)

    z = pd.DataFrame()
    z['id'] = test.id
    z['y'] = 0

    v = pd.DataFrame()
    v['y'] = y

    train2, y, test2 = cleanup_and_generate(train, y, test)

    keras1(train2, y, test2, v, z)
    rf1(train2.values, y, test2.values, v, z)
    xgb1(train2, y, test2, v, z)
    xgb2(train2, y, test2, v, z)
    et1(train2.values, y, test2.values, v, z)

    z.y = z.xgb1 * 0.35 + z.xgb2 * 0.35 + (z.keras1 + z.rf1 + z.et1) * 0.1
    z.y = prestore(z.y)
    save_results(v, z)

    print('done: %s.'%(now()))

    #'''
    clf = xgb.XGBClassifier(n_estimators=1000, learning_rate=.005)
    clf.fit(train2, y)
    for c in ['weight', 'gain', 'cover']:
        xgb.plot_importance(clf, title = 'Feature ' + c, importance_type=c)
    #'''
