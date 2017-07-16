#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import gzip
import os
import pickle

import numpy as np
import pandas as pd

class State(object):
    def run_model(self, model, data_module, feature_modules=[], debug_mode=False):
        saved = self.load('model')
        if debug_mode:
            saved = None
        if saved == None:
            train, y, test, _ = data_module.get()
            ftrain, ftest = [train], [test]
            for fm in feature_modules:
                tr, ts, _ = fm.get()
                ftrain.append(tr)
                ftest.append(ts)
            train = pd.concat(ftrain, axis=1)
            test = pd.concat(ftest, axis=1)
            print(train.shape, test.shape)

            z = pd.DataFrame()
            z['id'] = test.id
            z['y'] = 0

            v = pd.DataFrame()
            v['id'] = train.id
            v['y'] = y
            cv, _ = model(self, train, y, test, v, z)
            self.save('model', (v, z, cv, None))
        else:
            v, z, cv, _ = saved
        return v, z, cv, _

    def run_predict(self, pred, debug_mode=False, public_score=None):
        v, z, cv, _ = pred()
        if not debug_mode:
            self.save_model(v, z, cv)
        if public_score == None:
            # если есть public score - перезаписывать отправленное уже не стоит
            self.save_predicts(z)
        else:
            import os
            if os.path.exists('../model_scores.csv'):
                mdf = pd.read_csv('../model_scores.csv')
            else:
                mdf = pd.DataFrame(columns=['timestamp', 'model', 'cv', 'cv std', 'public score'])
            idx = mdf.model == self.base_name_
            if np.sum(idx) == 0:
                mdf.loc[len(mdf), 'model'] = self.base_name_
                idx = mdf.model == self.base_name_
            if (mdf.ix[idx, 'public score'] != public_score).bool():
                mdf.ix[idx, 'public score'] = public_score
                mdf.ix[idx, 'timestamp'] = self.now()
                mdf.ix[idx, 'cv'] =  np.mean(cv)
                mdf.ix[idx, 'cv std'] = np.std(cv)
            mdf.to_csv('../model_scores.csv', index=None)

    def now(self):
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def __init__(self, base_name):
        self.base_name_ = base_name

    def temp_name(self, name):
        os.makedirs('../run/tmp', exist_ok=True)
        path = '../run/tmp/' + self.base_name_ + '_' + name
        return path

    def drop_temp(self, name):
        if os.path.exists(self.temp_name(name)):
            os.remove(self.temp_name(name))

    def _local_name(self, name):
        path = '../run/local/' + self.base_name_ + '_' + name + '.pickle.gz'
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

    def save_predicts(self, z):
        os.makedirs('../predicts', exist_ok=True)
        pred_path = '../predicts/' + self.base_name_ + '.csv'
        z[['y']].to_csv(pred_path, header=None, index=False)
        print('predicts:')
        print(z.head(12))
        print('saved', pred_path)

    def save_model(self, v, z, cv):
        os.makedirs('../data/out', exist_ok=True)
        out_path = '../data/out/%s_cv%g_std%g.csv.gz'%(self.base_name_, np.mean(cv), np.std(cv))

        v = v.copy()
        z = z.copy()
        v['train'] = 1
        z['train'] = 0

        q = pd.concat([v, z], axis=0)
        q.to_csv(out_path, index=False, compression='gzip')

        print('model:')
        print(q.head(12))
        print('saved', out_path)

if '__main__' == __name__:
    pass
