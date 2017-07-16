#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import gc
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing

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

    z.y = np.clip(z.y.values, 1e-5, 1-1e-5)
    z[['y']].to_csv(pred_path, header=None, index=False)

    v['train'] = 1
    z['train'] = 0

    q = pd.concat([v, z], axis=0)
    q.to_csv(all_data_path, index=False, compression='gzip')

    for c in z.columns:
        if c in {'id', 'train', 'y'}: continue
        z[c] = prestore(z[c])
    print(z.head(20))
    print('saved', pred_path, all_data_path)

PREDICT_LOG_BIAS = 1e-5
def pconvert(c):
    return np.log(c + PREDICT_LOG_BIAS)

def prestore(c):
    return np.exp(c) - PREDICT_LOG_BIAS

def restore_missing(df, N_splits = 10):
    xgb_params = dict(
            max_depth = 5,
            learning_rate = 0.005,
            gamma = 1,
            alpha = 0.01,
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

    idx = (df_all.ap_hi == 19).values
    df_all.ix[idx, 'ap_hi'] = 190

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

    h = df_all.height / 100
    w = df_all.weight
    df_all['bwi'] = w / (h * h)

    df_all['ap_diff'] = df_all.ap_hi - df_all.ap_lo
    df_all['ap_k'] = df_all.ap_hi / df_all.ap_lo

    df_all['w_div_h'] = w / h
    df_all['aph_div_h'] = df_all.ap_hi / h;
    df_all['apl_div_h'] = df_all.ap_lo / h;
    df_all['apd_div_h'] = df_all.ap_diff / h;

    df_all['apl_div_w'] = df_all.ap_lo / w;
    df_all['apd_div_w'] = df_all.ap_diff / w;
    df_all['h_sub_w'] = df_all.height - w
    df_all['aph_div_w'] = df_all.ap_hi / w;

    df_all['log_age'] = np.log1p(df_all['age'])
    df_all['sqrt_age'] = np.sqrt(df_all['age'])
    df_all['age_root3'] = np.power(df_all['age'], 1/3)

    df_all.drop(['id'], axis=1, inplace=True)

    from sklearn import cluster
    cc = df_all.columns
    for k in range(2, 15):
        clf = cluster.KMeans(k)
        cname = 'kmeans_' + str(k)
        df_all[cname] = clf.fit_predict(df_all[cc].values)
        '''
        cdf = pd.get_dummies(df_all[cname], prefix=cname, drop_first=True)
        for c in cdf.columns:
            df_all[c] = cdf[c]
        print('added ', cname, cdf.columns)
        del cdf, c, clf, cname
        #'''

    train2 = df_all[:ntrain].reindex()
    test2 = df_all[ntrain:].reindex()
    print('train2.shape:', train2.shape, 'test2 shape:', test2.shape)

    return train2, y, test2

def xgb_common(train2, y, test2, v, z, N_seeds, N_splits, cname, xgb_params):
    scores = []
    skf = model_selection.StratifiedKFold(n_splits=N_splits, shuffle=True)
    dtest = xgb.DMatrix(test2)
    for s in range(N_seeds):
        cname2 = cname + str(s)
        v[cname2], z[cname2] = 0, 0
        xgb_params['seed'] = s + 4242
        for n, (itrain, ival) in enumerate(skf.split(train2, y)):
            dtrain = xgb.DMatrix(train2.ix[itrain], y[itrain])
            dvalid = xgb.DMatrix(train2.ix[ival], y[ival])
            watch = [(dtrain, 'train'), (dvalid, 'valid')]
            clf = xgb.train(xgb_params, dtrain, 10000, watch, early_stopping_rounds=100, verbose_eval=False)

            p = clf.predict(dvalid)
            v.loc[ival, cname2] += pconvert(p)
            score = metrics.log_loss(y[ival], p)
            z[cname2]  += pconvert(clf.predict(dtest))
            print(cname, 'seed %d step %d of %d: '%(xgb_params['seed'], n+1, skf.n_splits), score, now())
            scores.append(score)
        z[cname2] /= N_splits

    vloss = [metrics.log_loss(y, prestore(v[cname + str(i)])) for i in range(N_seeds)]
    print('validation loss: ', vloss, np.mean(vloss), np.std(vloss))
    cv=np.array(scores)
    print(cv, cv.mean(), cv.std())

def xgb1(train2, y, test2, v, z):
    cname = sys._getframe().f_code.co_name
    N_splits = 9
    N_seeds = 2
    xgb_params = dict(
            max_depth = 5,
            learning_rate = 0.02,
            gamma = 1,
            objective = 'binary:logistic',
            eval_metric = 'logloss',
            seed = 1,
            silent = 1
        )
    xgb_common(train2, y, test2, v, z, N_seeds, N_splits, cname, xgb_params)

def xgb2(train2, y, test2, v, z):
    cname = sys._getframe().f_code.co_name
    N_splits = 9
    N_seeds = 2
    xgb_params = dict(
            max_depth = 5,
            learning_rate = 0.02,
            subsample = 0.8,
            #colsample_bytree = 0.8,
            objective = 'binary:logistic',
            eval_metric = 'logloss',
            seed = 1,
            silent = 1
        )
    xgb_common(train2, y, test2, v, z, N_seeds, N_splits, cname, xgb_params)

def xgb3(train2, y, test2, v, z):
    cname = sys._getframe().f_code.co_name
    N_splits = 9
    N_seeds = 2
    xgb_params = dict(
            max_depth = 5,
            learning_rate = 0.02,
            subsample = 0.8,
            colsample_bytree = 0.8,
            objective = 'binary:logistic',
            eval_metric = 'logloss',
            seed = 1,
            silent = 1
        )
    xgb_common(train2, y, test2, v, z, N_seeds, N_splits, cname, xgb_params)

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

def keras_common(train3, y, test3, v, z, num_splits, cname, build_model, seed = 1234, batch_size = 128):
    v[cname], z[cname] = 0, 0
    np.random.seed(seed)
    build_model().summary(line_length=120)
    model_path = '../data/working/' + cname + '_keras_model.h5'
    ss = model_selection.ShuffleSplit(n_splits=num_splits, random_state=11, test_size=1/num_splits)
    scores = list()
    for n, (itrain, ival) in enumerate(ss.split(train3, y)):
        xtrain, xval = train3[itrain], train3[ival]
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
        v.loc[ival, cname] += pconvert(p).ravel()
        score = metrics.log_loss(y[ival], p)
        print(cname, 'fold %d: '%(n+1), score, now())
        scores.append(score)
        z[cname] += pconvert(model.predict(test3)).ravel()
        del model
        for i in range(3): gc.collect(i)
    os.remove(model_path)

    cv=np.array(scores)
    print(cv, cv.mean(), cv.std())
    z[cname] /= num_splits

def keras_mlp1(train2, y, test2, v, z):
    from keras import layers
    from keras import models
    from keras import optimizers
    cname = sys._getframe().f_code.co_name
    num_splits = 9
    scaler = preprocessing.RobustScaler()
    train3 = scaler.fit_transform(train2)
    test3 = scaler.transform(test2)
    input_dims = train3.shape[1]
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
    keras_common(train3, y, test3, v, z, num_splits, cname, build_model)

def keras_mlp2(train2, y, test2, v, z):
    from keras import layers
    from keras import models
    from keras import optimizers
    cname = sys._getframe().f_code.co_name
    num_splits = 9
    scaler = preprocessing.RobustScaler()
    train3 = scaler.fit_transform(train2)
    test3 = scaler.transform(test2)
    input_dims = train3.shape[1]
    def build_model():
        input_ = layers.Input(shape=(input_dims,))
        model = layers.Dense(1024, kernel_initializer='Orthogonal')(input_)
        model = layers.Activation('selu')(model)

        model = layers.Dense(128, kernel_initializer='Orthogonal')(model)
        model = layers.Activation('selu')(model)

        model = layers.Dense(16, kernel_initializer='Orthogonal')(model)
        model = layers.Activation('selu')(model)

        model = layers.Dense(1, activation='sigmoid')(model)

        model = models.Model(input_, model)
        model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop())
        #print(model.summary(line_length=120))
        return model
    keras_common(train3, y, test3, v, z, num_splits, cname, build_model)

def keras_mlp3(train2, y, test2, v, z):
    from keras import layers
    from keras import models
    from keras import optimizers
    cname = sys._getframe().f_code.co_name
    num_splits = 9
    scaler = preprocessing.RobustScaler()
    train3 = scaler.fit_transform(train2)
    test3 = scaler.transform(test2)
    input_dims = train3.shape[1]
    def build_model():
        input_ = layers.Input(shape=(input_dims,))
        model = layers.Dense(512, kernel_initializer='Orthogonal')(input_)
        model = layers.Activation('selu')(model)

        model = layers.Dense(256, kernel_initializer='Orthogonal')(model)
        model = layers.Activation('selu')(model)

        model = layers.Dense(32, kernel_initializer='Orthogonal')(model)
        model = layers.Activation('selu')(model)

        model = layers.Dense(1, activation='sigmoid')(model)

        model = models.Model(input_, model)
        model.compile(loss = 'binary_crossentropy', optimizer = optimizers.SGD(nesterov=True))
        #print(model.summary(line_length=120))
        return model
    keras_common(train3, y, test3, v, z, num_splits, cname, build_model)

if '__main__' == __name__:
    print('starting', now())
    np.random.seed(1234)

    with gzip.open('../data/input/all.pickle.gz', 'rb') as f:
        train, y, test = pickle.load(f)
    z = pd.DataFrame()
    z['id'] = test.id
    z['y'] = 0

    v = pd.DataFrame()
    v['id'] = train.id
    v['y'] = y

    models = set()

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
        save_state('model_predicts', (v, z, models))
    else:
        v, z, models = data2
    del data2

    for part in ['xgb1', 'xgb2', 'xgb3',
                 #'keras_mlp1', 'keras_mlp2', 'keras_mlp3'
                 ]:
        if part in models: continue
        print('running', part)
        f = globals()[part]
        f(train2, y, test2, v, z)
        models.add(part)
        save_state('model_predicts', (v, z, models))

    v1 = prestore(v.drop(['id', 'y'], axis=1).values)
    z1 = prestore(z.drop(['id', 'y'], axis=1).values)

    lr = linear_model.BayesianRidge()
    cv = model_selection.cross_val_score(lr, v1, v.y, cv=10, scoring=metrics.make_scorer(metrics.log_loss))
    print('stacking cv:', cv, cv.mean(), cv.std())

    lr.fit(v1, v['y'])
    print('stacking coeffs:', lr.coef_, np.sum(lr.coef_))
    z['y'] = np.clip(lr.predict(z1), 1e-5, 1-1e-5)

#    z.y = (z.keras_resnet1 + z.keras_mlp1 + z.xgb1 + z.xgb2 + z.xgb3) / 5
#    z.y = prestore(z.y)
    save_results(v, z)

    print('done.', now())
    drop_state('features', 'model_predicts')

    '''
    clf = xgb.XGBClassifier(n_estimators=1000, learning_rate=.005)
    clf.fit(train2, y)
    for c in ['weight', 'gain', 'cover']:
        xgb.plot_importance(clf, title = 'Feature ' + c, importance_type=c)
    #'''
