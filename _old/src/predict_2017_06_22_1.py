#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: user
"""
import datetime
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import model_selection
from sklearn import metrics

def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def base_data_name():
    script_name = os.path.basename(sys.argv[0])
    script_name = script_name[script_name.index('_'): script_name.rindex('.')]
    return script_name

def save_state(name, state):
    path = '../data/working/' + name + base_data_name() + '.pickle.gz'
    with gzip.open(path, 'wb') as f:
        pickle.dump(state, f)

def load_state(name):
    path = '../data/working/' + name + base_data_name() + '.pickle.gz'
    try:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def drop_state(*args):
    for name in args:
        path = '../data/working/' + name + base_data_name() + '.pickle.gz'
        if os.path.exists(path):
            os.remove(path)

def save_results(v, z):
    pred_path = '../submissions/p' + base_data_name() + '.csv'
    all_data_path = '../data/output/model' + base_data_name() + '.csv.gz'

    z[['y']].to_csv(pred_path, header=None, index=False)
    print(z.head(20))

    v['train'] = 1
    z['train'] = 0

    q = pd.concat([v, z], axis=0)
    q.to_csv(all_data_path, index=False, compression='gzip')
    print('saved', pred_path, all_data_path)

PREDICT_LOG_BIAS = 1e-5
def pconvert(c):
    return np.log(c + PREDICT_LOG_BIAS)

def prestore(c):
    return np.exp(c) - PREDICT_LOG_BIAS

def cleanup_and_generate(train, y, test, random_state=1234):
    ntrain = len(train)
    #base_columns = list(set(train.columns) - {'id', 'y'})

    for c in ['active', 'alco', 'smoke']:
        replacement = test.ix[test[c] != 'None', c].astype('float32').mean()
        test.ix[test[c]=='None', c] = -1
        test[c] = test[c].astype('float32')
    del c, replacement

    df_all = pd.concat([train, test]).reset_index(drop=True)

    idx = (df_all.height == 157).values * (df_all.weight == 10).values
    df_all.ix[idx, 'weight'] = 40

    idx = (df_all.height == 169).values * (df_all.weight == 10).values
    df_all.ix[idx, 'weight'] = 70

    idx = (df_all.height == 165).values * (df_all.weight == 10).values
    df_all.ix[idx, ['weight', 'ap_lo']] = 100, 110

    idx = (df_all.height == 178).values * (df_all.weight == 11).values
    df_all.ix[idx, 'weight'] = 110

    idx = (df_all.height == 183).values * (df_all.weight == 13).values
    df_all.ix[idx, 'weight'] = 130

    idx = (df_all.height == 169).values * (df_all.weight == 16.3).values
    df_all.ix[idx, 'weight'] = 63

    idx = (df_all.height == 170).values * (df_all.weight == 20).values
    df_all.ix[idx, 'weight'] = 80

    idx = (df_all.height == 162).values * (df_all.weight == 21).values
    df_all.ix[idx, 'weight'] = 71

    idx = (df_all.height == 177).values * (df_all.weight == 22).values
    df_all.ix[idx, 'weight'] = 80

    idx = (df_all.height == 157).values * (df_all.weight == 23).values
    df_all.ix[idx, 'weight'] = 43

    idx = (df_all.height == 171).values * (df_all.weight == 29).values
    df_all.ix[idx, 'weight'] = 49

    idx = (df_all.weight == 200).values
    df_all.ix[idx, 'weight'] = 100

    idx = (df_all.height == 180).values * (df_all.weight == 183).values
    df_all.ix[idx, 'weight'] = 83

    idx = (df_all.height == 58).values * (df_all.weight == 183).values
    df_all.ix[idx, ['height', 'weight', 'ap_lo']] = 158, 83, 100

    idx = (df_all.height == 161).values * (df_all.weight == 181).values
    df_all.ix[idx, ['weight', 'ap_lo']] = 121, 110

    idx = (df_all.height == 250).values * (df_all.weight == 86).values
    df_all.ix[idx, 'height'] = 150

    idx = (df_all.height == 165).values * (df_all.weight == 180).values
    df_all.ix[idx, 'weight'] = 80

    idx = (df_all.height == 180).values * (df_all.weight == 180).values
    df_all.ix[idx, 'weight'] = 130

    idx = (df_all.weight == 180).values
    df_all.ix[idx, 'weight'] = 130

    idx = (df_all.weight == 178).values * (df_all.height == 80).values
    df_all.ix[idx, 'height'] = 180

    idx = (df_all.weight == 178).values
    df_all.ix[idx, 'weight'] = 78

    idx = (df_all.weight == 177).values * (df_all.height == 177).values
    df_all.ix[idx, 'weight'] = 77

    #--------------------1----------------------------

    df_all['age_months'] = df_all.age // 30
    df_all['age_years'] = df_all.age // 365

    train2 = df_all[:ntrain].reindex()
    test2 = df_all[ntrain:].reindex()
    print('train2.shape:', train2.shape, 'test2 shape:', test2.shape)

    return train2, y, test2

def xgb1(train2, y, test2, v, z):
    cname = sys._getframe().f_code.co_name
    v[cname], z[cname] = 0, 0
    N_splits = 7
    N_seeds = 2
    scores = []
    skf = model_selection.StratifiedKFold(n_splits=N_splits, shuffle=True)
    xgb_params = dict(
            max_depth = 5,
            learning_rate = 0.04,
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
    N_seeds = 2
    scores = []
    skf = model_selection.StratifiedKFold(n_splits=N_splits, shuffle=True)
    xgb_params = dict(
            max_depth = 4,
            learning_rate = 0.04,
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

def xgb3(train2, y, test2, v, z):
    cname = sys._getframe().f_code.co_name
    v[cname], z[cname] = 0, 0
    N_splits = 7
    N_seeds = 2
    scores = []
    skf = model_selection.StratifiedKFold(n_splits=N_splits, shuffle=True)
    xgb_params = dict(
            max_depth = 4,
            learning_rate = 0.04,
            subsample = 0.8,
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

if '__main__' == __name__:
    print('starting', now())
    np.random.seed(1234)

    with gzip.open('../data/input/all.pickle.gz', 'rb') as f:
        train, y, test = pickle.load(f)
    z = pd.DataFrame()
    z['id'] = test.id
    z['y'] = 0

    v = pd.DataFrame()
    v['y'] = y

    data1 = load_state('features')
    #data1 = None
    if data1 == None:
        train2, y, test2 = cleanup_and_generate(train, y, test)
        save_state('features', (train2, y, test2))
        drop_state('model_predicts')
    else:
        train2, y, test2 = data1
    del data1

    data2 = load_state('model_predicts')
    if data2 == None:
        save_state('model_predicts', (v, z))
    else:
        v, z = data2
    del data2

    if 'xgb1' not in v.columns:
        xgb1(train2, y, test2, v, z)
        save_state('model_predicts', (v, z))
    if 'xgb2' not in v.columns:
        xgb2(train2, y, test2, v, z)
        save_state('model_predicts', (v, z))
    if 'xgb3' not in v.columns:
        xgb3(train2, y, test2, v, z)
        save_state('model_predicts', (v, z))

    z.y = (
            (z.xgb1 + z.xgb2 + z.xgb3) * (1.0 / 3)
        )
    z.y = prestore(z.y)
    save_results(v, z)

    print('done.', now())
    drop_state('features', 'model_predicts')