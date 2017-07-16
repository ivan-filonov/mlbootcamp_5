#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import gzip
import os
import pickle
import sys

import pandas as pd

def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def base_data_name():
    return os.path.basename(sys.argv[0])

def save_local(name, state):
    os.makedirs('../run/local', exist_ok=True)
    path = '../run/local/' + base_data_name() + '_' + name + '.pickle.gz'
    with gzip.open(path, 'wb') as f:
        pickle.dump(state, f)

def load_local(name):
    path = '../run/local/' + base_data_name() + '_' + name + '.pickle.gz'
    try:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def save_global(name, state):
    os.makedirs('../run/global', exist_ok=True)
    path = '../run/global/' + name + '.pickle.gz'
    with gzip.open(path, 'wb') as f:
        pickle.dump(state, f)

def load_global(name):
    path = '../run/global/' + name + '.pickle.gz'
    try:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def drop_local(*args):
    for name in args:
        path = '../run/local/' + base_data_name() + '_' + name + '.pickle.gz'
        if os.path.exists(path):
            os.remove(path)

def save_predicts(z):
    os.makedirs('../predicts', exist_ok=True)
    pred_path = '../predicts/' + base_data_name() + '.csv'
    z[['y']].to_csv(pred_path, header=None, index=False)
    print(z.head(20))
    print('saved', pred_path)

def model_path(layer):
    path = '../data/out_' + layer
    return path, path + '/model_' + base_data_name() + '.csv.gz'

def save_model(v, z, layer='l1'):
    path, name = model_path(layer)
    os.makedirs(path, exist_ok=True)

    v = v.copy()
    z = z.copy()
    v['train'] = 1
    z['train'] = 0

    q = pd.concat([v, z], axis=0)
    q.to_csv(name, index=False, compression='gzip')
    print('saved', name)

def model_cv(mean, std):
    pass

def model_public(value):
    pass

def save_l1(v, z):
    save_model(v, z, 'l1')

def save_l2(v, z):
    save_model(v, z, 'l2')

if '__main__' == __name__:
    pass