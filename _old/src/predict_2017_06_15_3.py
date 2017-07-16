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

def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def csv_name_suffix():
    script_name = os.path.basename(sys.argv[0])
    script_name = script_name[script_name.index('_'):-3]
    return script_name + '.csv'

def cleanup(train, test):
    train.ix[train.ap_hi < 0, 'ap_hi'] *= -1
    train.ix[train.ap_hi < 50, 'ap_hi'] *= 10
    train.ix[train.ap_hi > 5000, 'ap_hi'] *= 1e-2
    train.ix[train.ap_hi > 500, 'ap_hi'] *= 1e-1

    test.ix[1929, ['ap_hi', 'ap_lo']] = 110, 99
    test.ix[15510, ['ap_hi', 'ap_lo']] = 120, 88
    test.ix[12852, ['ap_hi', 'ap_lo']] = 130, np.NaN

    test.ix[test.ap_hi < 0, 'ap_hi'] *= -1
    test.ix[(test.ap_hi < 25).values, 'ap_hi'] *= 10
    test.ix[test.ap_hi > 5000, 'ap_hi'] *= 1e-2
    test.ix[test.ap_hi > 500, 'ap_hi'] *= 1e-1

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

from sklearn import model_selection
from sklearn import metrics
import xgboost as xgb

if '__main__' == __name__:
    print('starting', now())
    np.random.seed(1234)

    train = pd.read_csv('../data/input/train.csv', delimiter=';')
    test = pd.read_csv('../data/input/test.csv', delimiter=';')

    cleanup(train, test)

    for c in ['active', 'alco', 'smoke']:
        test.ix[test[c] == 'None', c] = test.ix[test[c] != 'None', c].mean()

    from sklearn.preprocessing import Imputer

    y = train.cardio
    train.drop('cardio', axis=1, inplace=True)

    imp = Imputer()
    train = imp.fit_transform(train)
    test = imp.transform(test)

    clf = xgb.XGBClassifier(n_estimators=100)
    loss = metrics.make_scorer(metrics.log_loss, greater_is_better=False)
    cv = model_selection.cross_val_score(clf, train, y=y.values, scoring=loss, cv=10)
    print(cv, cv.mean(), cv.std())

    clf.fit(train, y)
    z = clf.predict_proba(test)[:, 1]

    np.savetxt('../submissions/p' + csv_name_suffix(), z, fmt='%g')

    print('done', now())
