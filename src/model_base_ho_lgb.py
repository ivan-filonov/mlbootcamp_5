#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import lightgbm as lgb

from sklearn import metrics
from sklearn import model_selection

from functools import partial

import model_base

# TODO: categorical_column
class LgbModelBase(model_base.Model):
    def __init__(self,
                 name,
                 data_source,
                 features=[],
                 debug=False,
                 public_score=None,
                 max_ho_trials = 50,
                 num_splits = 9,
                 num_seeds = 3,
                 base_seed = 4242):
        super().__init__(name, data_source,features, debug, public_score)
        self.num_splits_ = num_splits
        self.num_seeds_ = num_seeds
        self.base_seed_ = base_seed
        self.max_ho_trials_ = max_ho_trials

    def model(self):
        #cname = sys._getframe().f_code.co_name
        cname = 'lgb'
        train, y, test = self.train_, self.y_, self.test_
        train.drop('id', axis=1, inplace=True)
        test.drop('id', axis=1, inplace=True)
        from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
        dtrain = lgb.Dataset(train, label=y)
        def fix_params(params):
            for p in ['min_data_in_leaf', 'num_leaves', 'max_bin' ]:
                params[p] = int(params[p])
            params['num_leaves'] = max(params['num_leaves'], 2)
        def step_xgb(params):
            fix_params(params)
            cv = lgb.cv(params, dtrain,
                        num_boost_round=10000,
                        early_stopping_rounds=50,
                        nfold=6,
                        seed=params['seed'])
            rounds = np.argmin(cv['binary_logloss-mean'])
            score = np.min(cv['binary_logloss-mean'])
            print(cname, score, rounds, params, self.now())
            return dict(loss=score, status=STATUS_OK)
        space_lgb = dict(
                bagging_fraction = hp.quniform('bagging_fraction', 0.5, 1, 0.001),
                colsample_bytree = hp.quniform('colsample_bytree', 0.6, 1, 0.05),
                feature_fraction = hp.quniform('feature_fraction', 0.5, 1, 0.001),
                lambda_l1 = hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
                lambda_l2 = hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
                learning_rate = hp.loguniform('learning_rate', -7, 0),
                max_bin = hp.qloguniform('max_bin', 0, 20, 1),
                max_depth = hp.choice('max_depth', range(2, 9)),
                min_child_weight = hp.quniform('min_child_weight', 1, 6, 1),
                min_data_in_leaf = hp.qloguniform('min_data_in_leaf', 0, 6, 1),
                min_sum_hessian_in_leaf = hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
                num_leaves = hp.qloguniform('num_leaves', 2, 7, 1),
                reg_alpha = hp.quniform('reg_alpha', 0, 1, 0.001),
                subsample = hp.quniform('subsample', 0.6, 1, 0.05),

                bagging_freq = 1,
                objective = 'binary',
                metric = 'binary_logloss',
                seed = 1,
                #silent = 1,
            )
        trs = self.load('lightgbm_trials')
        if trs == None or self.debug_:
            tr = Trials()
        else:
            tr, _ = trs
        if len(tr.trials) > 0:
            print('reusing %d trials, best was:'%(len(tr.trials)), space_eval(space_lgb, tr.argmin))
            best = tr.argmin
        while len(tr.trials) < self.max_ho_trials_:
            print(len(tr.trials), end=' ')
            #best = fmin(step_xgb, space_lgb, algo=tpe.suggest, max_evals=len(tr.trials) + 1, trials = tr)
            best = fmin(step_xgb, space_lgb, algo=partial(tpe.suggest, n_startup_jobs=1), max_evals=len(tr.trials) + 1, trials = tr)
            self.save('lightgbm_trials', (tr, space_lgb))
        lgb_params = space_eval(space_lgb, best)
        fix_params(lgb_params)
        print(lgb_params)

        N_splits = self.num_splits_
        N_seeds = self.num_seeds_

        v, z = self.v_, self.z_
        skf = model_selection.StratifiedKFold(n_splits=N_splits, shuffle=True)
        cv = []
        for s in range(N_seeds):
            scores = []
            cname2 = cname + str(s)
            v[cname2], z[cname2] = 0, 0
            lgb_params['seed'] = s + self.base_seed_
            for n, (itrain, ival) in enumerate(skf.split(train, y)):
                dtrain = lgb.Dataset(train.ix[itrain], y[itrain])
                dvalid = lgb.Dataset(train.ix[ival], y[ival])
                clf = lgb.train(lgb_params, dtrain,
                                num_boost_round=10000,
                                valid_sets=[dtrain, dvalid],
                                valid_names=['train', 'valid'],
                                early_stopping_rounds=100, verbose_eval=False)

                p = clf.predict(train.ix[ival])
                v.loc[ival, cname2] += p
                score = metrics.log_loss(y[ival], p)
                z[cname2]  += clf.predict(test)
                print(cname, 'seed %d step %d of %d: '%(lgb_params['seed'], n+1, skf.n_splits), score, self.now())
                scores.append(score)
            z[cname2] /= N_splits
            cv.append(np.mean(scores))
            print('seed %d loss: '%(lgb_params['seed']), scores, np.mean(scores), np.std(scores))
            z['y'] = z[cname2]

        print('cv:', cv, np.mean(cv), np.std(cv))
        return cv, None

if '__main__' == __name__:
    pass
