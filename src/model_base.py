#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd

from functools import partial

from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection

from common_base import Base

class Model(Base):
    def __init__(self, name, data_source=None, feature_sources=[], debug=False, public_score=None):
        super().__init__(name)
        self.debug_ = debug
        self.data_source_ = data_source
        self.public_score_ = public_score
        self.feature_sources_ = feature_sources

    def model(self):
        '''
        returns cv, <something>
        cv - number or array (something recognized by numpy)
        '''
        print('implement this!')
        raise Exception()

    def predict(self):
        saved = None if self.debug_ else self.load('model')
        if saved == None:
            self.prepare_data()

            z = pd.DataFrame()
            z['id'] = self.test_.id
            z['y'] = 0

            v = pd.DataFrame()
            v['id'] = self.train_.id
            v['y'] = self.y_

            self.v_, self.z_ = v, z
            self.cv_, _ = self.model()
            self.save('model', (self.v_, self.z_, self.cv_, None))
        else:
            self.v_, self.z_, self.cv_, _ = saved
        return self.v_, self.z_, self.cv_, None

    def main(self):
        self.predict()
        if not self.debug_:
            self.save_predicts()
        if self.public_score_ == None:
            self.save_features_csv()
        else:
            self.update_public_score()

    def update_public_score(self):
        if os.path.exists('../model_scores.csv'):
            mdf = pd.read_csv('../model_scores.csv')
        else:
            mdf = pd.DataFrame(columns=['timestamp', 'model', 'cv', 'cv std', 'public score'])
        idx = mdf.model == self.name_
        if np.sum(idx) == 0:
            mdf.loc[len(mdf), 'model'] = self.name_
            idx = mdf.model == self.name_
        if (mdf.ix[idx, 'public score'] != self.public_score_).bool():
            mdf.ix[idx, 'public score'] = self.public_score_
            mdf.ix[idx, 'timestamp'] = self.now()
            mdf.ix[idx, 'cv'] =  np.mean(self.cv_)
            mdf.ix[idx, 'cv std'] = np.std(self.cv_)
        print('public score updated for', self.name_)
        mdf.to_csv('../model_scores.csv', index=None)

    def prepare_data(self):
        train, y, test, _ = self.data_source_.get()
        ftrain, ftest = [train], [test]
        for fm in self.feature_sources_:
            f1, f2, _ = fm.get()
            ftrain.append(f1)
            ftest.append(f2)
        self.train_ = pd.concat(ftrain, axis=1)
        self.test_ = pd.concat(ftest, axis=1)
        self.y_ = y

    def save_predicts(self):
        os.makedirs('../predicts', exist_ok=True)
        pred_path = '../predicts/' + self.name_ + '.csv'
        self.z_[['y']].to_csv(pred_path, header=None, index=False)
        print('predicts:')
        print(self.z_.head(12))
        print('saved', pred_path)

    def save_features_csv(self):
        os.makedirs('../data/out', exist_ok=True)
        v, z, cv = self.v_, self.z_, self.cv_

        out_path = '../data/out/%s_cv%g_std%g.csv.gz'%(self.name_, np.mean(cv), np.std(cv))

        v = v.copy()
        z = z.copy()
        v['train'] = 1
        z['train'] = 0

        q = pd.concat([v, z], axis=0)
        q.to_csv(out_path, index=False, compression='gzip')

        print('model:')
        print(q.head(12))
        print('saved', out_path)

class L2Model(Model):
    def __init__(self, name, l1_models, debug=False, public_score=None,
                 preselect_features=False, hyperopt_rounds=100, n_fold=10):
        super().__init__(name, None, [], debug, public_score)
        self.l1_models_ = l1_models
        self.preselect_features_ = preselect_features
        self.hyperopt_rounds_ = hyperopt_rounds
        self.n_fold_ = n_fold

    def predict(self):
        saved = None if self.debug_ else self.load('model')
        if saved == None:
            self.prepare_data()

            z = pd.DataFrame()
            z['id'] = self.test_id_
            z['y'] = 0

            v = pd.DataFrame()
            v['id'] = self.train_id_
            v['y'] = self.y_

            self.v_, self.z_ = v, z
            self.cv_, _ = self.model()
            self.save('model', (self.v_, self.z_, self.cv_, None))
        else:
            self.v_, self.z_, self.cv_, _ = saved
        return self.v_, self.z_, self.cv_, None

    def prepare_data(self):
        vs, zs, cvs = [], [], []
        import importlib
        for module in [importlib.import_module(name) for name in self.l1_models_]:
            v, z, cv, _ = module.predict()
            vs.append(v)
            zs.append(z)
            cvs.append(cv)

        self.train_id_ = vs[-1].id
        self.test_id_ = zs[-1].id

        self.y_ = vs[-1].y
        for s in vs + zs:
            s.drop(['id', 'y'], axis=1, inplace=True)

        self.train_ = pd.concat(vs, axis=1)
        self.test_ = pd.concat(zs, axis=1)

    def greedy_select_features(self):
        saved = None if self.debug_ else self.load('chosen_features')
        if saved == None:
            print('initial shapes:', self.train_.shape, self.test_.shape)
            num_columns = self.train_.shape[1]
            col_names = [str(c) for c in range(num_columns)]
            self.train_.columns = col_names
            self.test_.columns = col_names

            g_best_score = 1e9
            g_best_features = None

            y = self.y_.ravel()
            current = set()
            scorer = metrics.make_scorer(metrics.log_loss)
            for _ in enumerate(col_names):
                avail = set(col_names).difference(current)
                best_score = 1e9
                best_features = None
                for f in avail:
                    newf = list(current | {f})
                    cv = model_selection.cross_val_score(linear_model.BayesianRidge(),
                                                         self.train_[newf], y,
                                                         cv=self.n_fold_, n_jobs=-2,
                                                         scoring = scorer)
                    score = np.mean(cv)
                    if best_score > score:
                        best_score = score
                        best_features = newf
                current = set(best_features)
                if g_best_score > best_score:
                    g_best_score = best_score
                    g_best_features = best_features
                    print('new best:', g_best_score, g_best_features, self.now())
                if len(best_features) - len(g_best_features) > 15:
                    break
            self.save('chosen_features', (g_best_features, None))
        else:
            g_best_features, _ = saved

        print('feature selection complete.', self.now())
        self.train_ = self.train_[g_best_features]
        self.test_ = self.test_[g_best_features]

    def model(self):
        #cname = sys._getframe().f_code.co_name
        if self.preselect_features_:
            self.greedy_select_features()

        train = self.train_.values
        test = self.test_.values
        y = self.y_

        cv = model_selection.cross_val_score(linear_model.BayesianRidge(),
                                             train, y,
                                             cv = self.n_fold_,
                                             scoring = metrics.make_scorer(metrics.log_loss))
        self.baseline_score_ = np.mean(cv)
        self.baseline_stacker_ = linear_model.BayesianRidge()

        for fit_intercept in [False, True]:
            for normalize in [False, True]:
                lr = linear_model.LinearRegression(fit_intercept=fit_intercept,
                                                   normalize=normalize)
                cv = model_selection.cross_val_score(lr, train, y, cv = self.n_fold_,
                                             scoring = metrics.make_scorer(metrics.log_loss))
                score = np.mean(cv)
                if score < self.baseline_score_:
                    self.baseline_score_ = score
                    self.baseline_stacker_ = lr
        print('baseline:', self.baseline_score_, self.baseline_stacker_)

        np.random.seed(1)
        from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
        space_stack = hp.choice('stacking by',
            [
                dict( type = 'Ridge',
                     random_state = 1, #hp.choice('random_state', range(1, 100000)),
                     alpha = hp.loguniform('alpha', -7, 5),
                     fit_intercept = hp.choice('fit_intercept1', [True, False]),
                     normalize = hp.choice('normalize1', [True, False])
                     ),
            ])

        def get_lr(params):
            t = params['type']
            del params['type']

            if t == 'LinearRegression':
                lr = linear_model.LinearRegression(**params)
            elif t == 'Ridge':
                lr = linear_model.Ridge(**params)
            else:
                raise Exception()

            return lr

        def step(params):
            print(params, end = ' ')
            cv = model_selection.cross_val_score(get_lr(params),
                                                 train, y,
                                                 cv=self.n_fold_,
                                                 scoring = metrics.make_scorer(metrics.log_loss))
            score = np.mean(cv)
            print(score, self.now())
            return dict(loss=score, status=STATUS_OK)

        trs = self.load('trials')
        if trs == None:
            tr = Trials()
        else:
            tr, _ = trs
        if len(tr.trials) > 0:
            print('reusing %d trials, best was:'%(len(tr.trials)), space_eval(space_stack, tr.argmin))
        mt = max(self.hyperopt_rounds_, len(tr.trials) + 1)
        while len(tr.trials) < mt:
            print(len(tr.trials), end=' ')
            best = fmin(step, space_stack, algo=partial(tpe.suggest, n_startup_jobs=1), max_evals=len(tr.trials) + 1, trials = tr)
            self.save('trials', (tr, space_stack))
        params = space_eval(space_stack, best)

        print('best params:', params)
        lr = get_lr(params)
        cv = model_selection.cross_val_score(lr,
                                             train, y,
                                             cv=self.n_fold_,
                                             scoring=metrics.make_scorer(metrics.log_loss))
        if np.mean(cv) > self.baseline_score_:
            lr = self.baseline_stacker_
            cv = model_selection.cross_val_score(lr, train, y, cv=self.n_fold_,
                                                 scoring=metrics.make_scorer(metrics.log_loss))

        lr.fit(train, y)

        v, z = self.v_, self.z_
        z['p'] = np.clip(lr.predict(test), 1e-5, 1-1e-5)
        z['y'] = z['p']
        v['p'] = model_selection.cross_val_predict(lr,
                                             train, y,
                                             cv=10)
        print('cv:', np.mean(cv), np.std(cv))
        return cv, None

if '__main__' == __name__:
    pass
