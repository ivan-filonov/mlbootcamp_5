#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def cleanup(df_all):
    idx = (df_all.ap_hi > 240).values + (df_all.ap_hi < 70).values
    df_all.ix[idx, 'ap_hi'] = np.NaN

    idx = (df_all.ap_lo > 170).values + (df_all.ap_lo < 40).values
    df_all.ix[idx, 'ap_lo'] = np.NaN

    idx = (df_all.weight < 30).values
    df_all.ix[idx, 'weight'] = np.NaN

    idx = (df_all.height > 240).values + (df_all.height < 110).values
    df_all.ix[idx, 'height'] = np.NaN

import data_clean_extreme_ap as data_src
import data_source_base
class dataset(data_source_base.DataSource):
    def __init__(self):
        name = 'data_clean_3'
        save_path = None
        super().__init__(name, save_path)

    def build(self):
        train, y, test, _ = data_src.get()
        ntrain = len(train)
        test['y'] = -1
        df_all = pd.concat([train, test])
        cleanup(df_all)

        df_all.drop('id', axis=1)

        train = df_all[:ntrain].reindex().drop('y', axis=1)
        test = df_all[ntrain:].reindex().drop('y', axis=1)
        return train, y, test, None

d = dataset()
def get():
    return d.get_data()

if '__main__' == __name__:
    d.main()