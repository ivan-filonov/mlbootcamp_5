#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

import data_source_base

import data_raw as data

class features(data_source_base.FeatureSource):
    def __init__(self):
        name = 'features_misc_2'
        save_path = '../data/features/features_misc_2.pickle.gz'
        super().__init__(name, save_path)

    def build(self):
        train, _, test, _ = data.get()
        #
        ntrain = len(train)
        df = pd.concat([train, test], axis=0)
        to_drop = df.columns

        h = df.height / 100
        df['bwi'] = df['weight'] / (h * h)
        df['ap_p'] = df['ap_hi'] - df['ap_lo']
        df['ap_m4'] = (df['ap_hi'] + 3 * df['ap_lo']) / 4
        df['ap_m3'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
        df['ap_m2'] = (df['ap_hi'] + df['ap_lo']) / 2

        male = df['gender'] == 2
        df.ix[male, 'ap_hi_e'] = 109 + 0.5 * df.ix[male, 'age'] / 365.25 + 0.1 * df.ix[male, 'weight']
        df.ix[male, 'ap_lo_e'] = 74 + 0.1 * df.ix[male, 'age'] / 365.25 + 0.15 * df.ix[male, 'weight']
        df.ix[~male, 'ap_hi_e'] = 102 + 0.7 * df.ix[~male, 'age'] / 365.25 + 0.15 * df.ix[~male, 'weight']
        df.ix[~male, 'ap_lo_e'] = 78 + 0.17 * df.ix[~male, 'age'] / 365.25 + 0.1 * df.ix[~male, 'weight']

        df.ix[male, 'weight_ah'] = (df.ix[male, 'ap_hi'] - 109 - 0.5 * df.ix[male, 'age'] / 365.25) / 0.1
        df.ix[male, 'weight_al'] = (df.ix[male, 'ap_lo'] - 74 - 0.1 * df.ix[male, 'age'] / 365.25) / 0.15
        df.ix[~male, 'weight_ah'] = (df.ix[~male, 'ap_hi'] - 102 - 0.7 * df.ix[~male, 'age'] / 365.25) / 0.15
        df.ix[~male, 'weight_al'] = (df.ix[~male, 'ap_lo'] - 78 - 0.17 * df.ix[~male, 'age'] / 365.25) / 0.1

        df['ap_hi_ed'] = df['ap_hi'] - df['ap_hi_e']
        df['ap_lo_ed'] = df['ap_lo'] - df['ap_lo_e']
        df['dw_ah'] = df['weight'] - df['weight_ah']
        df['dw_al'] = df['weight'] - df['weight_al']

        df = df.drop(to_drop, axis=1)
        ftrain = df[:ntrain]
        ftest = df[ntrain:]
        return ftrain, ftest, None


f = features()
def get():
    '''
    returns - tuple(train=DataFrame, test=DataFrame, other)
    -- columns to add to dataset
    '''
    return f.get_features()

if '__main__' == __name__:
    f.main()