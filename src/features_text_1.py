#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

import data_source_base

import data_raw as data

class features(data_source_base.FeatureSource):
    def __init__(self):
        name = 'features_text_1'
        save_path = '../data/features/features_text_1.pickle.gz'
        super().__init__(name, save_path)

    def build(self):
        train, _, test, _ = data.get()
        cset = []
        ntrain = len(train)
        df = pd.concat([train, test], axis=0)
        to_drop = df.columns
        for sc in ['height', 'weight', 'ap_hi', 'ap_lo']:
            tc = df[sc].apply(str)
            maxc = tc.apply(len).max()
            for n in range(maxc):
                df['ft_l_'+sc+'_'+str(n)] = tc.apply(lambda s:ord(s[n])  if n < len(s) else -1)
                df['ft_r_'+sc+'_'+str(n)] = tc.apply(lambda s:ord(s[-n]) if n < len(s) else -1)
                cset.append('ft_l_'+sc+'_'+str(n))
                cset.append('ft_r_'+sc+'_'+str(n))

        df = df.drop(to_drop, axis=1)
        self.train_= df[:ntrain]
        self.test_ = df[ntrain:]
        return self.train_, self.test_, None


f = features()
def get():
    '''
    returns - tuple(train=DataFrame, test=DataFrame, other)
    -- columns to add to dataset
    '''
    return f.get_features()

if '__main__' == __name__:
    f.main()