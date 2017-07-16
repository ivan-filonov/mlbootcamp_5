#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import gzip
import os
import pickle

class Base(object):
    def __init__(self, name):
        self.name_ = name

    def now(self):
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def temp_name(self, name):
        os.makedirs('../run/tmp', exist_ok=True)
        path = '../run/tmp/' + self.name_ + '_' + name
        return path

    def drop_temp(self, name):
        if os.path.exists(self.temp_name(name)):
            os.remove(self.temp_name(name))

    def _local_name(self, name):
        path = '../run/local/' + self.name_ + '_' + name + '.pickle.gz'
        return path

    def save(self, name, state):
        os.makedirs('../run/local', exist_ok=True)
        path = self._local_name(name)
        with gzip.open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, name):
        path = self._local_name(name)
        try:
            with gzip.open(path, 'rb') as f:
                return pickle.load(f)
        except:
            return None

    def drop(self, *args):
        for name in args:
            path = self._local_name(name)
            if os.path.exists(path):
                os.remove(path)

    def save_global(self, name, state):
        os.makedirs('../run/global', exist_ok=True)
        path = '../run/global/' + name + '.pickle.gz'
        with gzip.open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_global(self, name):
        path = '../run/global/' + name + '.pickle.gz'
        try:
            with gzip.open(path, 'rb') as f:
                return pickle.load(f)
        except:
            return None

if '__main__' == __name__:
    pass
