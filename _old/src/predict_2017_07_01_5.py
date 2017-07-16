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

def restore_missing(df, N_splits = 10):
    xgb_params = dict(
            max_depth = 4,
            learning_rate = 0.005,
            subsample = 0.7,
            gamma = 1,
            alpha = 0.01,
            #colsample_bytree = 0.8,
            objective = 'binary:logistic',
            eval_metric = 'logloss',
            seed = 1,
            silent = 1
        )
    #{'gamma': 0.0, 'seed': 1, 'eval_metric': 'logloss', 'objective': 'binary:logistic', 'subsample': 0.6, 'min_child_weight': 1, 'colsample_bytree': 0.9, 'silent': 1, 'n_estimators': 10000, 'reg_alpha': 0.05, 'learning_rate': 0.005, 'max_depth': 2}
    df.ix[df.active == -1, 'active'] = 1
    df.ix[df.alco == -1, 'alco'] = 0

    label = 'smoke'
    print('before', label, '{{{', df[label].value_counts(), '}}}')
    xtrain = df[df[label] > -1].copy()
    ytrain = xtrain[label].astype('int32').values
    xtrain = xtrain.drop(label, axis=1)
    #print(label, ytrain.value_counts())

    xpred = df[df[label] == -1].copy()
    ypred = xpred[label] * 0
    xpred = xpred.drop(label, axis=1)

    dpred = xgb.DMatrix(xpred)
    dtrain = xgb.DMatrix(xtrain, label=ytrain)

    cv = xgb.cv(params=xgb_params,
                dtrain=dtrain,
                num_boost_round=10000,
                early_stopping_rounds=100,
                nfold=10,
                metrics='error',
                stratified=True)
    print(label, 'num_boost_rounds =', len(cv))
    bst = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=len(cv))
    ypred += bst.predict(dpred)
    df.ix[df[label] == -1, label] = (ypred > 0.5) * 1
    print('restored', label, '{{{', df[label].value_counts(), '}}}')

def cleanup(df_all):
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
    idx = (df_all.height == 162).values * (df_all.weight == 175).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.weight == 175).values
    df_all.ix[idx, 'weight'] = 75

    idx = (df_all.height == 87).values * (df_all.weight == 173).values
    df_all.ix[idx, ['height', 'weight']] = 187, 113

    idx = (df_all.height == 169).values * (df_all.weight == 172).values
    df_all.ix[idx, 'weight'] = 72

    idx = (df_all.height == 170).values * (df_all.weight == 171).values
    df_all.ix[idx, 'weight'] = 71

    idx = (df_all.weight == 170).values * (df_all.gender == 2).values
    df_all.ix[idx, 'weight'] = 70

    idx = (df_all.weight == 170).values * (df_all.height == 97).values
    df_all.ix[idx, 'height'] = 197

    idx = (df_all.weight == 170).values
    df_all.ix[idx, 'weight'] = 140

    idx = (df_all.height == 165).values * (df_all.weight == 169).values
    df_all.ix[idx, 'weight'] = 70

    idx = (df_all.height < 100).values * (df_all.weight == 168).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.weight == 168).values
    df_all.ix[idx, 'weight'] -= 100

    idx = (df_all.height == 104).values * (df_all.weight == 159).values
    df_all.ix[idx, 'height'] = 164

    idx = (df_all.height == 159).values * (df_all.weight == 165).values
    df_all.ix[idx, 'weight'] -= 100

    idx = (df_all.height == 159).values * (df_all.weight == 159).values
    df_all.ix[idx, 'weight'] -= 100

    idx = (df_all.ap_lo == 1100).values
    df_all.ix[idx, 'ap_lo'] /= 10

    idx = (df_all.height == 75).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 68).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 65).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.ap_lo == 1000).values
    df_all.ix[idx, 'ap_lo'] /= 10

    idx = (df_all.height == 60).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 110).values
    df_all.ix[idx, 'height'] = 170

    idx = (df_all.height == 57).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 59).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 76).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 66).values
    df_all.ix[idx, ['height', 'ap_hi']] = 166, 120

    idx = (df_all.height == 81).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 55).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 71).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 74).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 70).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 52).values
    df_all.ix[idx, 'height'] += 100
    #--------------------2----------------------------
    df_all.ap_hi = np.abs(df_all.ap_hi)
    df_all.ap_lo = np.abs(df_all.ap_lo)

    idx = (df_all.height == 50).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 56).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 72).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 62).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 64).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 67).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 91).values
    df_all.ix[idx, ['height', 'weight']] += 155, 91

    idx = (df_all.height == 96).values
    df_all.ix[idx, 'height'] = 156

    idx = (df_all.height == 98).values
    df_all.ix[idx, ['height', 'ap_hi']] = 158, 120

    idx = (df_all.height == 99).values
    df_all.ix[idx, 'height'] = 159

    idx = (df_all.height == 100).values
    df_all.ix[idx, 'height'] = 160

    idx = (df_all.height == 112).values
    df_all.ix[idx, 'height'] = 172

    idx = (df_all.height == 122).values * (df_all.weight == 161).values
    df_all.ix[idx, ['height', 'weight']] = 172, 61

    idx = (df_all.height == 122).values
    df_all.ix[idx, 'height'] = 142

    idx = (df_all.height == 102).values
    df_all.ix[idx, 'height'] = 162

    idx = (df_all.ap_lo == 10000).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_hi == 16020).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.ap_hi == 14020).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.ap_hi == 11500).values
    df_all.ix[idx, 'ap_hi'] = 115

    idx = (df_all.ap_hi == 11020).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.ap_hi == 13010).values
    df_all.ix[idx, 'ap_hi'] = 110

    idx = (df_all.ap_hi == 12008).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.ap_hi == 14900).values
    df_all.ix[idx, 'ap_hi'] = 140

    idx = (df_all.ap_hi == 12080).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.ap_hi == 1420).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.ap_hi == 1300).values
    df_all.ix[idx, 'ap_hi'] = 130

    idx = (df_all.ap_hi == 1500).values
    df_all.ix[idx, 'ap_hi'] = 150

    idx = (df_all.ap_hi == 906).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 90, 60
    #--------------------3----------------------------
    idx = (df_all.ap_hi == 1620).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.ap_lo == 8044).values
    df_all.ix[idx, 'ap_lo'] = 110

    idx = (df_all.ap_hi == 1400).values
    df_all.ix[idx, 'ap_hi'] = 140

    idx = (df_all.ap_lo == 9100).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo == 8000).values
    df_all.ix[idx, 'ap_lo'] = 80

    idx = (df_all.ap_lo == 8100).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo == 1200).values
    df_all.ix[idx, 'ap_lo'] = 120

    idx = (df_all.ap_lo == 9011).values
    df_all.ix[idx, 'ap_lo'] = 110

    idx = (df_all.ap_lo == 800).values
    df_all.ix[idx, 'ap_lo'] = 80

    idx = (df_all.ap_lo == 8500).values
    df_all.ix[idx, 'ap_lo'] = 85

    idx = (df_all.ap_lo == 8099).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo == 8079).values
    df_all.ix[idx, 'ap_lo'] = 80

    idx = (df_all.ap_lo == 1110).values
    df_all.ix[idx, 'ap_lo'] = 110

    idx = (df_all.ap_lo == 710).values
    df_all.ix[idx, 'ap_lo'] = 70

    idx = (df_all.ap_lo == 7100).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo == 2088).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 120, 80

    idx = (df_all.ap_hi == 1).values * (df_all.ap_lo > 1000).values
    df_all.ix[idx, 'ap_hi'] = 110
    df_all.ix[idx, 'ap_lo'] -= 1008

    idx = (df_all.ap_lo == 4100).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo == 570).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 115, 70

    idx = (df_all.ap_lo == 11000).values
    df_all.ix[idx, 'ap_lo'] = 110

    idx = (df_all.ap_lo == 1033).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo == 4700).values
    df_all.ix[idx, 'ap_lo'] = 70

    idx = (df_all.ap_lo == 5700).values
    df_all.ix[idx, 'ap_lo'] = 70

    idx = (df_all.ap_lo == 6800).values
    df_all.ix[idx, 'ap_lo'] = 80

    idx = (df_all.ap_lo == 8200).values
    df_all.ix[idx, 'ap_lo'] = 80

    idx = (df_all.ap_lo == 9800).values
    df_all.ix[idx, 'ap_lo'] = 80

    idx = (df_all.ap_lo == 7099).values
    df_all.ix[idx, 'ap_lo'] = 110

    idx = (df_all.ap_lo > 7000).values
    df_all.ix[idx, 'ap_lo'] /= 100
    #--------------------4----------------------------
    idx = (df_all.ap_lo == 1900).values
    df_all.ix[idx, 'ap_lo'] = 90

    idx = (df_all.ap_lo == 1001).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo == 1400).values
    df_all.ix[idx, 'ap_lo'] = 140

    idx = (df_all.ap_lo == 1044).values
    df_all.ix[idx, 'ap_lo'] = 104

    idx = (df_all.ap_lo == 1008).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo == 1022).values
    df_all.ix[idx, 'ap_lo'] = 102

    idx = (df_all.ap_lo == 1177).values
    df_all.ix[idx, 'ap_lo'] = 117

    idx = (df_all.ap_lo == 1011).values
    df_all.ix[idx, 'ap_lo'] = 111

    idx = (df_all.ap_lo == 1111).values
    df_all.ix[idx, 'ap_lo'] = 111

    idx = (df_all.ap_lo == 1120).values
    df_all.ix[idx, 'ap_lo'] = 120

    idx = (df_all.ap_lo == 900).values
    df_all.ix[idx, 'ap_lo'] = 90

    idx = (df_all.ap_lo == 170).values * (df_all.ap_hi == 200).values
    df_all.ix[idx, 'ap_lo'] = 120

    idx = (df_all.ap_lo == 170).values * (df_all.ap_hi == 20).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 200, 120

    idx = (df_all.ap_lo == 1130).values
    df_all.ix[idx, 'ap_lo'] = 130

    idx = (df_all.ap_lo == 1300).values
    df_all.ix[idx, 'ap_lo'] = 130

    idx = (df_all.ap_lo == 1125).values
    df_all.ix[idx, 'ap_lo'] = 125

    idx = (df_all.ap_lo == 1007).values
    df_all.ix[idx, 'ap_lo'] = 107

    idx = (df_all.ap_lo == 1066).values
    df_all.ix[idx, 'ap_lo'] = 107

    idx = (df_all.ap_lo == 1077).values
    df_all.ix[idx, 'ap_lo'] = 107

    idx = (df_all.ap_lo == 1088).values
    df_all.ix[idx, 'ap_lo'] = 108

    idx = (df_all.ap_lo == 1099).values
    df_all.ix[idx, 'ap_lo'] = 110

    idx = (df_all.ap_hi == 1502).values
    df_all.ix[idx, 'ap_hi'] = 150

    idx = (df_all.ap_hi == 902).values
    df_all.ix[idx, 'ap_hi'] = 90

    idx = (df_all.ap_hi == 1608).values
    df_all.ix[idx, 'ap_hi'] = 160

    idx = (df_all.ap_hi == 1130).values
    df_all.ix[idx, 'ap_hi'] = 130
    #--------------------4----------------------------
    idx = (df_all.ap_hi == 907).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 90, 70

    idx = (df_all.ap_hi == 11).values * (df_all.ap_lo == 70).values
    df_all.ix[idx, 'ap_hi'] = 110

    idx = (df_all.ap_hi == 11).values * (df_all.ap_lo == 71).values
    df_all.ix[idx, 'ap_hi'] = 110

    idx = (df_all.ap_hi == 11).values * (df_all.ap_lo == 60).values
    df_all.ix[idx, 'ap_hi'] = 110

    idx = (df_all.ap_hi == 11).values * (df_all.ap_lo == 80).values
    df_all.ix[idx, 'ap_hi'] = 110

    idx = (df_all.ap_hi == 11).values * (df_all.ap_lo == 120).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 110, 70

    idx = (df_all.ap_hi == 400).values
    df_all.ix[idx, 'ap_hi'] = 100

    idx = (df_all.ap_hi == 10).values * (df_all.ap_lo == 160).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 100, 60

    idx = (df_all.ap_hi == 10).values
    df_all.ix[idx, 'ap_hi'] = 100

    idx = (df_all.ap_hi == 12).values * (df_all.ap_lo < 100).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.ap_hi == 12).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 140, 120

    idx = (df_all.ap_hi == 1202).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.ap_hi == 909).values
    df_all.ix[idx, 'ap_hi'] = 90

    idx = (df_all.ap_hi == 13).values
    df_all.ix[idx, 'ap_hi'] = 130

    idx = (df_all.ap_hi == 14).values
    df_all.ix[idx, 'ap_hi'] = 140

    idx = (df_all.ap_hi == 15).values
    df_all.ix[idx, 'ap_hi'] = 150

    idx = (df_all.ap_hi == 16).values * (df_all.ap_lo == 10).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 160, 100

    idx = (df_all.ap_hi == 16).values
    df_all.ix[idx, 'ap_hi'] = 160

    idx = (df_all.ap_hi == 957).values
    df_all.ix[idx, 'ap_hi'] = 95

    idx = (df_all.ap_hi == 2000).values
    df_all.ix[idx, 'ap_hi'] = 200

    idx = (df_all.ap_hi == 1407).values
    df_all.ix[idx, 'ap_hi'] = 140

    idx = (df_all.ap_hi == 1409).values
    df_all.ix[idx, 'ap_hi'] = 140

    idx = (df_all.ap_hi == 509).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 150, 90

    idx = (df_all.ap_hi == 17).values
    df_all.ix[idx, 'ap_hi'] = 170

    idx = (df_all.ap_hi == 1).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 130, 90

    idx = (df_all.ap_hi == 806).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 80, 60

    idx = (df_all.ap_lo == 850).values
    df_all.ix[idx, 'ap_lo'] = 85
    #--------------------5----------------------------
    idx = (df_all.ap_hi == 1205).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.ap_hi == 1110).values
    df_all.ix[idx, 'ap_hi'] = 110

    idx = (df_all.ap_hi == 960).values
    df_all.ix[idx, 'ap_hi'] = 90

    idx = (df_all.ap_hi == 701).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 110, 70

    idx = (df_all.ap_lo == 902).values
    df_all.ix[idx, 'ap_lo'] = 90

    idx = (df_all.ap_lo == 1003).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo == 801).values
    df_all.ix[idx, 'ap_lo'] = 80

    idx = (df_all.ap_lo == 1101).values
    df_all.ix[idx, 'ap_lo'] = 110

    idx = (df_all.ap_lo == 1002).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo == 1139).values
    df_all.ix[idx, 'ap_lo'] = 140

    idx = (df_all.ap_lo == 809).values
    df_all.ix[idx, 'ap_lo'] = 80

    idx = (df_all.weight == 165).values * (df_all.height < 190).values
    df_all.ix[idx, 'weight'] = 65

    idx = (df_all.ap_lo == 1004).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo == 1009).values
    df_all.ix[idx, 'ap_lo'] = 110

    idx = (df_all.ap_lo == 1211).values
    df_all.ix[idx, 'ap_lo'] = 120

    idx = (df_all.ap_lo == 910).values
    df_all.ix[idx, 'ap_lo'] = 90

    idx = (df_all.ap_lo == 1140).values
    df_all.ix[idx, 'ap_lo'] = 140

    idx = (df_all.ap_lo > 800).values
    df_all.ix[idx, 'ap_lo'] /= 10

    idx = (df_all.ap_hi == 160).values * (df_all.ap_lo == 708).values
    df_all.ix[idx, 'ap_lo'] = 108

    idx = (df_all.ap_lo > 600).values
    df_all.ix[idx, 'ap_lo'] /= 10

    idx = (df_all.ap_lo == 585).values
    df_all.ix[idx, 'ap_lo'] = 85

    idx = (df_all.ap_hi == 309).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 130, 90

    idx = (df_all.ap_lo == 0).values * (df_all.ap_hi == 108).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 100, 80

    idx = (df_all.ap_lo == 0).values * (df_all.ap_hi == 117).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 110, 70

    idx = (df_all.ap_lo == 0).values * (df_all.ap_hi == 118).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 110, 80

    idx = (df_all.ap_lo == 0).values * (df_all.ap_hi == 138).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 130, 80

    idx = (df_all.ap_lo == 0).values * (df_all.ap_hi == 148).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 140, 80

    idx = (df_all.ap_lo == 0).values * (df_all.ap_hi == 149).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 140, 90
    #--------------------6----------------------------
    idx = (df_all.ap_hi == 90).values * (df_all.ap_lo == 160).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 90, 60

    idx = (df_all.ap_hi == 100).values * (df_all.ap_lo == 160).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 100, 60

    idx = (df_all.ap_hi == 80).values * (df_all.ap_lo == 170).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 80, 70

    idx = (df_all.ap_hi == 90).values * (df_all.ap_lo == 170).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 90, 70

    idx = (df_all.ap_hi == 95).values * (df_all.ap_lo == 180).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 95, 80

    idx = (df_all.ap_hi == 95).values * (df_all.ap_lo == 170).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 95, 70

    idx = (df_all.ap_hi == 95).values * (df_all.ap_lo == 160).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 95, 60

    idx = (df_all.ap_hi == 130).values * (df_all.ap_lo == 190).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 130, 90

    idx = (df_all.ap_hi == 140).values * (df_all.ap_lo == 190).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 140, 90

    idx = (df_all.ap_hi == 110).values * (df_all.ap_lo == 170).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 110, 70

    idx = (df_all.ap_hi == 150).values * (df_all.ap_lo == 180).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 150, 80

    idx = (df_all.ap_hi == 170).values * (df_all.ap_lo == 190).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 170, 90

    for p_h, p_l in [ (120, 80), (130, 80), (130, 90), (140, 90), (100, 70), (125, 80),
                      (110, 70), (140, 80), (150, 90), (110, 80), (120, 70), (100, 60),
                      (110, 60), (100, 80), (100, 90), (120, 90), (150, 100),(115, 80),
                      (130, 95), (150, 120),(140, 95), (130, 85), (135, 95), (105, 80),
                      (150, 95),
                    ]:
        idx = (df_all.ap_hi == p_l).values * (df_all.ap_lo == p_h).values
        df_all.ix[idx, ['ap_hi', 'ap_lo']] = p_h, p_l

    idx = (df_all.ap_hi == 20).values * (df_all.ap_lo == 80).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 120, 80

    idx = (df_all.ap_hi == 20).values * (df_all.ap_lo == 90).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 120, 90

    idx = (df_all.ap_hi == 172).values * (df_all.ap_lo == 190).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 172, 90

    idx = (df_all.ap_hi == 401).values
    df_all.ix[idx, 'ap_hi'] = 101

    idx = (df_all.ap_hi == 7).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.ap_lo == 12).values
    df_all.ix[idx, 'ap_lo'] = 120

    idx = (df_all.ap_lo == 10).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo < 10).values
    df_all.ix[idx, 'ap_lo'] *= 10

    idx = (df_all.ap_lo == 30).values
    df_all.ix[idx, 'ap_lo'] = 90

    idx = (df_all.ap_hi == 116).values * (df_all.ap_lo == 120).values
    df_all.ix[idx, 'ap_hi'] = 146

    idx = (df_all.ap_hi == 100).values * (df_all.ap_lo == 0).values
    df_all.ix[idx, 'ap_lo'] = 60

    idx = (df_all.ap_hi == 120).values * (df_all.ap_lo == 0).values
    df_all.ix[idx, 'ap_lo'] = 80

    idx = (df_all.ap_hi == 130).values * (df_all.ap_lo == 0).values
    df_all.ix[idx, 'ap_lo'] = 90 - 10 * df_all.ix[idx, 'active']

    idx = (df_all.ap_lo == 0).values
    df_all.ix[idx, 'ap_lo'] = 90
    #--------------------7----------------------------
    idx = (df_all.ap_lo == 10).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo == 15).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 150, 70

    idx = (df_all.ap_lo == 19).values
    df_all.ix[idx, 'ap_lo'] = 90

    idx = (df_all.ap_lo == 20).values * (df_all.ap_hi > 99).values * (df_all.ap_hi < 131).values
    df_all.ix[idx, 'ap_lo'] = 70

    idx = (df_all.ap_lo == 20).values * (df_all.ap_hi == 180).values
    df_all.ix[idx, 'ap_lo'] = 120

    idx = (df_all.ap_lo == 20).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 94, 70 #!!!!!!!!!!!!!!!!!!!!!!!!!!!

    idx = (df_all.ap_hi == 19).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.height == 246).values
    df_all.ix[idx, 'height'] = 176

    idx = (df_all.height == 154).values * (df_all.weight == 162).values
    df_all.ix[idx, 'weight'] = 62

    idx = (df_all.height == 180).values * (df_all.weight == 164).values
    df_all.ix[idx, 'weight'] = 104

    idx = (df_all.height == 164).values * (df_all.weight == 164).values
    df_all.ix[idx, 'weight'] = 104

    idx = (df_all.height == 160).values * (df_all.weight == 155).values
    df_all.ix[idx, 'weight'] = 55

    idx = (df_all.height == 125).values * (df_all.weight == 167).values
    df_all.ix[idx, ['height', 'weight']] = 145, 107
    #--------------------7----------------------------

def cleanup_and_generate(train, y, test, random_state=1234):
    ntrain = len(train)
    #base_columns = list(set(train.columns) - {'id', 'y'})

    for c in ['active', 'alco', 'smoke']:
        replacement = test.ix[test[c] != 'None', c].astype('float32').mean()
        test.ix[test[c]=='None', c] = -1
        test[c] = test[c].astype('float32')
    del c, replacement

    df_all = pd.concat([train, test]).reset_index(drop=True)

    cleanup(df_all)

    restore_missing(df_all)

    df_all['age_months'] = (df_all.age // 30) % 12
    df_all['age_years'] = df_all.age // 365

    h = df_all.height / 100
    w = df_all.weight
    df_all['bwi'] = w / (h * h)

    df_all['ap_diff'] = df_all.ap_hi - df_all.ap_lo

    df_all['w_div_h'] = w / h
    df_all['aph_div_h'] = df_all.ap_hi / h;
    df_all['apl_div_h'] = df_all.ap_lo / h;
    df_all['apd_div_h'] = df_all.ap_diff / h;

    df_all['apd_div_w'] = df_all.ap_diff / w;
    df_all['h_sub_w'] = df_all.height - w
    df_all['aph_div_w'] = df_all.ap_hi / w;
    df_all['apl_div_w'] = df_all.ap_lo / w;

    train2 = df_all[:ntrain].reindex()
    test2 = df_all[ntrain:].reindex()
    print('train2.shape:', train2.shape, 'test2 shape:', test2.shape)

    return train2, y, test2

def xgb_base(train2, y, test2, v, z, xgb_params, N_splits, N_seeds, cname, base_seed=42):
    v[cname], z[cname] = 0, 0
    scores = []

    dtest = xgb.DMatrix(test2)
    for s in range(N_seeds):
        xgb_params['seed'] = s + base_seed
        skf = model_selection.StratifiedKFold(n_splits=N_splits, shuffle=True, random_state=s + base_seed)
        for n, (itrain, ival) in enumerate(skf.split(train2, y)):
            dtrain = xgb.DMatrix(train2.ix[itrain], y[itrain])
            dvalid = xgb.DMatrix(train2.ix[ival], y[ival])
            watch = [(dtrain, 'train'), (dvalid, 'valid')]
            clf = xgb.train(xgb_params, dtrain, 10000, watch, early_stopping_rounds=100, verbose_eval=False)

            p = clf.predict(dvalid)
            v.loc[ival, cname] += pconvert(p)
            score = metrics.log_loss(y[ival], p)
            z[cname]  += pconvert(clf.predict(dtest))
            print(cname, 'seed %d step %d of %d: '%(xgb_params['seed'], n+1, skf.n_splits), score, now())
            scores.append(score)

    z[cname] /= N_splits * N_seeds
    v[cname] /= N_seeds
    print('validation loss: ', metrics.log_loss(y, prestore(v[cname])))
    cv=np.array(scores)
    print(cv, cv.mean(), cv.std())

def xgb1(train2, y, test2, v, z):
    cname = sys._getframe().f_code.co_name
    N_splits = 9
    N_seeds = 1
    xgb_params = dict(
            max_depth = 5,
            learning_rate = 0.02,
            objective = 'binary:logistic',
            eval_metric = 'logloss',
            seed = 1,
            silent = 1
        )
    xgb_base(train2, y, test2, v, z, xgb_params, N_splits, N_seeds, cname)

def tune_xgb(train2, y, test2, v, z):
    cname = sys._getframe().f_code.co_name
    xgb_params = dict(
            learning_rate = 0.02,
            objective = 'binary:logistic',
            eval_metric = 'logloss',
            silent = 1,
            nthread = 1,
            seed = 42
        )
    dtrain = xgb.DMatrix(train2, y)
    cv = xgb.cv(xgb_params, dtrain, nfold=10, num_boost_round=10000, early_stopping_rounds=100, seed=42)
    print(cname, 'num_boost_round =', len(cv))
    del xgb_params['eval_metric']
    xgb_params['n_estimators'] = len(cv)
    grid_params = dict(
            max_depth = range(2, 5),
            min_child_weight = range(1, 4)
        )
    scorer = metrics.make_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True)
    gs1 = model_selection.GridSearchCV(xgb.XGBClassifier(**xgb_params),
                                       param_grid = grid_params,
                                       scoring = scorer,
                                       n_jobs = -2,
                                       cv = 8,
                                       verbose = 0
                                       )
    gs1.fit(train2, y)
    print(cname, gs1.best_params_, gs1.best_score_)
    for k in gs1.best_params_:
        xgb_params[k] = gs1.best_params_[k]
    grid_params2 = dict(
            gamma = [i / 10 for i in range(0, 14, 2)]
        )
    gs2 = model_selection.GridSearchCV(xgb.XGBClassifier(**xgb_params),
                                       param_grid = grid_params2,
                                       scoring = scorer,
                                       n_jobs = -2,
                                       cv = 8,
                                       verbose = 0
                                       )
    gs2.fit(train2, y)
    print(cname, gs2.best_params_, gs2.best_score_)
    xgb_params['gamma'] = gs2.best_params_['gamma']
    grid_params3 = dict(
            subsample = [i/10 for i in range(6, 11)],
            colsample_bytree = [i/10 for i in range(6, 11)]
        )
    gs3 = model_selection.GridSearchCV(xgb.XGBClassifier(**xgb_params),
                                       param_grid = grid_params3,
                                       scoring = scorer,
                                       n_jobs = -2,
                                       cv = 8,
                                       verbose = 0
                                       )
    gs3.fit(train2, y)
    print(cname, gs3.best_params_, gs3.best_score_)
    for k in gs3.best_params_:
        xgb_params[k] = gs3.best_params_[k]

    grid_params = dict(
            reg_alpha = [1e-5, 1e-2, 0.05, 0.1, 0.15, 1, 100]
        )
    gs = model_selection.GridSearchCV(xgb.XGBClassifier(**xgb_params),
                                       param_grid = grid_params,
                                       scoring = scorer,
                                       n_jobs = -2,
                                       cv = 8,
                                       verbose = 0
                                       )
    gs.fit(train2, y)
    for k in gs.best_params_:
        xgb_params[k] = gs.best_params_[k]
    print(cname, gs.best_params_, gs.best_score_)

    xgb_params['nthread'] = -2
    xgb_params['learning_rate'] = 0.002
    xgb_params['eval_metric'] = 'logloss'
    print(cname, xgb_params)

    N_splits = 9
    N_seeds = 10
    xgb_base(train2, y, test2, v, z, xgb_params, N_splits, N_seeds, cname)

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

    if 'tune_xgb' not in v.columns:
        tune_xgb(train2, y, test2, v, z)
        save_state('model_predicts', (v, z))
    '''
    if 'xgb1' not in v.columns:
        xgb1(train2, y, test2, v, z)
        save_state('model_predicts', (v, z))
    #'''

    z.y = z.tune_xgb
    z.y = prestore(z.y)
    save_results(v, z)

    print('done.', now())
    drop_state('features', 'model_predicts')

    #'''
    clf = xgb.XGBClassifier(n_estimators=1000, learning_rate=.005)
    clf.fit(train2, y)
    for c in ['weight', 'gain', 'cover']:
        xgb.plot_importance(clf, title = 'Feature ' + c, importance_type=c)
    #'''
