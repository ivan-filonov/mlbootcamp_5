#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 18:15:02 2017

@author: user
"""
import gzip
import pickle

import data_source_base
class dataset(data_source_base.DataSource):
    def __init__(self):
        name = 'data_raw'
        save_path = None
        super().__init__(name, save_path)

    def build(self):
        with gzip.open('../data/in/all.pickle.gz', 'rb') as f:
            train, y, test = pickle.load(f)
            test['alco'] = test['alco'].replace('None', 0).astype('int32')
            test['smoke'] = test['smoke'].replace('None', 0.0001).astype('float32')
            test['active'] = test['active'].replace('None', 1).astype('int32')
        return train, y, test, None

d = dataset()
def get():
    return d.get_data()

if '__main__' == __name__:
    d.main()
