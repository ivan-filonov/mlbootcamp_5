#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb

import data_raw as data_src

import data_source_base
class dataset(data_source_base.DataSource):
    def __init__(self):
        name = 'data_raw'
        save_path = '../data/features/restored_smoke.pickle.gz'
        super().__init__(name, save_path)

    def build(self):
        train, y, test, _ = data_src.get()
        xgb_params = dict(
                max_depth = 5,
                learning_rate = 0.005,
                subsample = 0.7,
                gamma = 5,
                alpha = 0.01,
                #colsample_bytree = 0.8,
                objective = 'binary:logistic',
                eval_metric = 'logloss',
                seed = 1,
                silent = 1
            )

        idx = (test.smoke > 0).values * (test.smoke < 1).values
        print('values to restore:', np.sum(idx))
        xtrain = pd.concat([train, test[~idx]])
        ytrain = xtrain['smoke']
        xtrain.drop('smoke', axis=1, inplace=True)
        print(xtrain.shape, ytrain.shape, test[idx].shape)

        dtrain = xgb.DMatrix(xtrain.values, ytrain.values)
        dpred = xgb.DMatrix(test[idx].drop('smoke', axis=1).values)

        cv = xgb.cv(params=xgb_params,
                    dtrain=dtrain,
                    num_boost_round=10000,
                    early_stopping_rounds=50,
                    nfold=10,
                    seed=1,
                    metrics='error',
                    stratified=True)
        print('smoke num_boost_rounds =', len(cv))
        bst = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=len(cv))
        test.ix[idx, 'smoke'] = bst.predict(dpred)
        test['smoke'] = (test['smoke'] > 0.5) * 1
        return train, y, test, None

d = dataset()
def get():
    return d.get_data()

if '__main__' == __name__:
    d.main()
