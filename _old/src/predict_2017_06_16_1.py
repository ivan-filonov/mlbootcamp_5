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

from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def csv_name_suffix():
    script_name = os.path.basename(sys.argv[0])
    script_name = script_name[script_name.index('_'):-3]
    return script_name + '.csv'

def cleanup(train, test, use_hand_labels = True):
    train.ix[train.ap_hi < 0, 'ap_hi'] *= -1
    train.ix[train.ap_hi < 50, 'ap_hi'] *= 10
    train.ix[train.ap_hi > 5000, 'ap_hi'] *= 1e-2
    train.ix[train.ap_hi > 500, 'ap_hi'] *= 1e-1

    if use_hand_labels:
        test.ix[1929, ['ap_hi', 'ap_lo']] = 110, 99
        test.ix[15510, ['ap_hi', 'ap_lo']] = 120, 88
        test.ix[12852, ['ap_hi', 'ap_lo']] = 130, np.NaN

    test.ix[test.ap_hi < 0, 'ap_hi'] *= -1
    test.ix[(test.ap_hi < 25).values, 'ap_hi'] *= 10
    test.ix[test.ap_hi > 5000, 'ap_hi'] *= 1e-2
    test.ix[test.ap_hi > 500, 'ap_hi'] *= 1e-1

    if use_hand_labels:
        train.ix[8757, ['ap_hi', 'ap_lo']] = 120, 88
        train.ix[2014, ['ap_hi', 'ap_lo']] = 90, 60
        train.ix[17381, ['ap_hi', 'ap_lo']] = 130, 80
        train.ix[31783, ['ap_hi', 'ap_lo']] = 90, 70
        train.ix[38370, ['ap_hi', 'ap_lo']] = 140, 80
        train.ix[41505, ['ap_hi', 'ap_lo']] = 80, 60
        train.ix[42397, ['ap_hi', 'ap_lo']] = 90, 60
        train.ix[43922, ['ap_hi', 'ap_lo']] = 100, 80
        train.ix[63787, ['ap_hi', 'ap_lo']] = 110, 70
        train.ix[68663, ['ap_hi', 'ap_lo']] = 90, 60

    train.ix[train.ap_lo < 0, 'ap_lo'] *= -1
    train.ix[train.ap_lo > 2500, 'ap_lo'] *= 1e-2
    train.ix[train.ap_lo > 250, 'ap_lo'] *= 1e-1
    train.ix[train.ap_lo < 10, 'ap_lo'] *= 10
    train.ix[train.ap_lo < 10, 'ap_lo'] = np.NAN
    train.ix[(train.ap_lo < 49).values * (train.cardio==0).values, 'ap_lo'] = np.NAN

    test.ix[test.ap_lo < 0, 'ap_lo'] *= -1
    test.ix[test.ap_lo > 2500, 'ap_lo'] *= 1e-2
    test.ix[test.ap_lo > 250, 'ap_lo'] *= 1e-1
    test.ix[test.ap_lo < 10, 'ap_lo'] *= 10

    x = train.ix[train.ap_lo >= train.ap_hi, ['ap_lo', 'ap_hi']]
    x.columns = ['ap hi', 'ap lo']
    train.ix[train.ap_lo >= train.ap_hi, ['ap_hi', 'ap_lo']] = x

    train.ix[train.height < 90, 'height'] += 100
    test.ix[test.height < 90, 'height'] += 100

def gen_features(train, y, test):
    for c in ['active', 'alco', 'smoke']:
        le = preprocessing.LabelEncoder()
        le.fit(train[c].values.tolist() + test[c].values.tolist())
        train[c] = le.transform(train[c])
        test[c] = le.transform(test[c])

    train['ap_dif'] = train.ap_hi - train.ap_lo
    test['ap_dif'] = test.ap_hi - test.ap_lo

    h = train['height'] / 100
    train['BWI'] = train['weight'] / (h * h)
    h = test['height'] / 100
    test['BWI'] = test['weight'] / (h * h)

    imp = preprocessing.Imputer()
    train = imp.fit_transform(train)
    test = imp.transform(test)

    return train, y, test

if '__main__' == __name__:
    print('starting', now())
    np.random.seed(1234)

    train = pd.read_csv('../data/input/train.csv', delimiter=';')
    test = pd.read_csv('../data/input/test.csv', delimiter=';')

    cleanup(train, test)

    y = train.cardio
    train.drop('cardio', axis=1, inplace=True)

    train, y, test = gen_features(train, y, test)

    xgb_params = dict(
            learning_rate = 0.005,
            objective = 'binary:logistic',
            eval_metric = 'logloss',
            seed = 1,
            silent = 1
        )

    N_splits = 500
    z = np.zeros((30000, ))
    v = np.zeros((70000, ))
    skf = model_selection.StratifiedKFold(n_splits=N_splits, shuffle=True)
    for n, (itrain, ival) in enumerate(skf.split(train, y)):
        print('step %d of %d'%(n+1, skf.n_splits), now())
        dtrain = xgb.DMatrix(train[itrain], y[itrain])
        dvalid = xgb.DMatrix(train[ival], y[ival])
        dtest = xgb.DMatrix(test)
        watch = [(dtrain, 'train'), (dvalid, 'valid')]
        clf = xgb.train(xgb_params, dtrain, 10000, watch, early_stopping_rounds=100, verbose_eval=100)
        v[ival] = clf.predict(dvalid)
        z  += np.log1p(clf.predict(dtest))

    print('validation loss: ', metrics.log_loss(y, v))

    z = np.expm1(z / skf.n_splits)
    np.savetxt('../submissions/p' + csv_name_suffix(), z, fmt='%g')

    print('done', now())
