#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn import preprocessing

import data_source_base

import data_clean_2 as data
import features_misc_2 as fea_1

def squish(df, src, dst, ratio = 10):
    le = preprocessing.LabelEncoder()
    df[dst] = le.fit_transform((df[src] / ratio).astype('int32'))

def wa_dict(data, col, real_target = 'cardio', global_avg = 0.5, alpha = 10):
    g = data.groupby(col)
    KK = g.size()
    mean_rt = g[real_target].mean()
    c = (mean_rt * KK + global_avg * alpha) / (KK + alpha)
    d = c.to_dict()
    return d

class features(data_source_base.FeatureSource):
    def __init__(self):
        name = 'features_wa_3'
        save_path = '../data/features/features_wa_3.pickle.gz'
        super().__init__(name, save_path)

    def build(self):
        train, y, test, _ = data.get()
        ftrain1, ftest1, _ = fea_1.get()

        train = pd.concat([train, ftrain1], axis=1)
        test = pd.concat([test, ftest1], axis=1)

        #
        ntrain = len(train)
        df = pd.concat([train, test], axis=0)
        to_drop = df.columns

        for c in train.columns:
            if c == 'id':
                continue
            if c in {'gluc', 'cholesterol'}:
                squish(df, c, c + 'S', 1)
            else:
                squish(df, c, c + 'S', 10)

        df = df.drop(to_drop, axis=1)
        train = df[:ntrain]
        test = df[ntrain:].copy()
        #
        train = pd.concat([train, y], axis=1)

        global_avg = y.mean()
        for c in test.columns:
            d = wa_dict(train, c, 'cardio', global_avg, 10)
            test[c] = test[c].map(d).astype('float32')
        test.fillna(global_avg, inplace=True)

        kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
        for itrain, iset in kf.split(train, y):
            for c in test.columns:
                d = wa_dict(train.ix[itrain], c, 'cardio', global_avg, 10)
                train.ix[iset, c] = train.ix[iset, c].map(d)
        train.drop('cardio', axis=1, inplace=True)
        train.fillna(global_avg, inplace=True)

        return train.astype('float32'), test.astype('float32'), None


f = features()
def get():
    '''
    returns - tuple(train=DataFrame, test=DataFrame, other)
    -- columns to add to dataset
    '''
    return f.get_features()

if '__main__' == __name__:
    f.main()
