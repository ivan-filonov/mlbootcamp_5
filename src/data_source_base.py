#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip
import os
import pickle

import numpy as np

from common_base import Base

class DataSource(Base):
    def __init__(self, name, save_path = None):
        super().__init__(name)
        self.save_path_ = save_path

    def build(self):
        print('implement this!')
        raise Exception()

    def main(self):
        self.ensure_data()

    def get_data(self):
        self.ensure_data()
        return self.train_, self.y_, self.test_, None

    def ensure_data(self):
        if hasattr(self, 'train_') and hasattr(self, 'y_') and hasattr(self, 'test_'):
            return self.train_, self.y_, self.test_
        if not self.load_data():
            self.train_, self.y_, self.test_, _ = self.build()
            self.save_data()

    def load_data(self):
        if self.save_path_ == None:
            return False
        try:
            with gzip.open(self.save_path_, 'rb') as f:
                saved = pickle.load(f)
            self.train_, self.y_, self.test_ = saved
            return True
        except:
            pass
        return False

    def save_data(self):
        if self.save_path_ == None:
            return
        os.makedirs('../data/features', exist_ok=True)
        with gzip.open(self.save_path_, 'wb') as f:
            pickle.dump((self.train_, self.y_, self.test_), f)
            print('saved', self.save_path_)

class FeatureSource(Base):
    def __init__(self, name, save_path = None):
        super().__init__(name)
        self.save_path_ = save_path

    def build(self):
        print('implement this!')
        raise Exception()

    def main(self):
        self.ensure_features()

    def get_features(self):
        self.ensure_features()
        return self.train_, self.test_, None

    def ensure_features(self):
        np.random.seed(1)
        if hasattr(self, 'train_') and hasattr(self, 'test_'):
            return self.train_, self.test_, None
        if not self.load_data():
            self.train_, self.test_, _ = self.build()
            self.save_data()

    def load_data(self):
        if self.save_path_ == None:
            return False
        try:
            with gzip.open(self.save_path_, 'rb') as f:
                saved = pickle.load(f)
            self.train_, self.test_ = saved
        except:
            return False
        return True

    def save_data(self):
        if self.save_path_ == None:
            return
        os.makedirs('../data/features', exist_ok=True)
        with gzip.open(self.save_path_, 'wb') as f:
            pickle.dump((self.train_, self.test_), f)
            print('saved', self.save_path_)