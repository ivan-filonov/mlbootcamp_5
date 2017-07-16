#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import decomposition

import data_source_base

import features_text_2 as data

class features(data_source_base.FeatureSource):
    def __init__(self):
        name = 'features_test_2_pca16'
        save_path = '../data/features/features_text_2_pca16.pickle.gz'
        super().__init__(name, save_path)

    def build(self):
        train, test, _ = data.get()

        to_drop = train.columns

        pca = decomposition.PCA(n_components=16, random_state=1)
        train_ = pca.fit_transform(train)
        test_ = pca.transform(test)

        for c in range(train_.shape[1]):
            cname = 'f_t_pca16_' + str(c)
            train[cname] = train_[: ,c]
            test[cname] = test_[: ,c]

        return train.drop(to_drop, axis=1), test.drop(to_drop, axis=1), None


f = features()
def get():
    '''
    returns - tuple(train=DataFrame, test=DataFrame, other)
    -- columns to add to dataset
    '''
    return f.get_features()

if '__main__' == __name__:
    f.main()