#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn import cluster
from sklearn import model_selection

import data_source_base

import data_raw as data

class features(data_source_base.FeatureSource):
    def __init__(self):
        name = 'features_kmeans_1'
        save_path = '../data/features/features_kmeans_1.pickle.gz'
        super().__init__(name, save_path)

    def build(self):
        train, y, test, _ = data.get()

        ntrain = len(train)
        df = pd.concat([train, test], axis=0)
        to_drop = df.columns

        dcn = []
        for n in [2, 5, 10, 15, 25]:
            cname = 'kmeans_' + str(n)
            dcn.append(cname)
            df[cname] = cluster.KMeans(n_clusters=n).fit_predict(df)

        df = pd.get_dummies(df, columns=dcn)

        df = df.drop(to_drop, axis=1)
        train = df[:ntrain]
        test = df[ntrain:].copy()

        return train.astype('int32'), test.astype('int32'), None


f = features()
def get():
    '''
    returns - tuple(train=DataFrame, test=DataFrame, other)
    -- columns to add to dataset
    '''
    return f.get_features()

if '__main__' == __name__:
    f.main()
