#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from functools import partial

import model_base

# TODO: categorical_column
class EtModelBase(model_base.Model):
    def __init__(self,
                 name,
                 data_source,
                 features=[],
                 debug=False,
                 public_score=None,
                 max_ho_trials = 30,
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
        cname = 'et'
        train, y, test = self.train_, self.y_, self.test_
        train.drop('id', axis=1, inplace=True)
        test.drop('id', axis=1, inplace=True)
        from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
        def step_et(params):
            clf = ensemble.ExtraTreesRegressor(**params)
            cv = model_selection.cross_val_score(clf,
                                                 train, y,
                                                 scoring=metrics.make_scorer(metrics.log_loss),
                                                 cv = 5,
                                                 n_jobs = -2)
            score = np.mean(cv)
            print(cname, score, params, self.now())
            return dict(loss=score, status=STATUS_OK)
        space_et = dict(
            n_estimators = hp.choice('n_estimators', range(50, 1500)),
            min_samples_split = hp.choice('min_samples_split', range(2, 10)),
            min_samples_leaf = hp.choice('min_samples_leaf', range(1, 10)),
            max_features = hp.choice('max_features', range(4, min(20, train.shape[1]))),
            random_state = 1
            )
        trs = self.load('et_trials')
        if trs == None or self.debug_:
            tr = Trials()
        else:
            tr, _ = trs
        if len(tr.trials) > 0:
            print('reusing %d trials, best was:'%(len(tr.trials)), space_eval(space_et, tr.argmin))
            best = tr.argmin
        while len(tr.trials) < 30:
            print(len(tr.trials), end=' ')
            best = fmin(step_et, space_et, algo=partial(tpe.suggest, n_startup_jobs=1), max_evals=len(tr.trials) + 1, trials = tr)
            self.save('et_trials', (tr, space_et))
        et_params = space_eval(space_et, best)
        print(et_params)

        N_splits = self.num_splits_
        N_seeds = self.num_seeds_

        v, z = self.v_, self.z_
        skf = model_selection.StratifiedKFold(n_splits=N_splits, shuffle=True)
        cv = []
        for s in range(N_seeds):
            scores = []
            cname2 = cname + str(s)
            v[cname2], z[cname2] = 0, 0
            et_params['random_state'] = s + 4242
            for n, (itrain, ival) in enumerate(skf.split(train, y)):
                clf = ensemble.ExtraTreesRegressor(**et_params)
                clf.fit(train.ix[itrain], y[itrain])
                p = clf.predict(train.ix[ival])
                v.loc[ival, cname2] += p
                score = metrics.log_loss(y[ival], p)
                z[cname2]  += clf.predict(test)
                print(cname, 'seed %d step %d of %d: '%(et_params['random_state'], n+1, skf.n_splits), score, self.now())
                scores.append(score)
            z[cname2] /= N_splits
            cv.append(np.mean(scores))
            print('seed %d loss: '%(et_params['random_state']), scores, np.mean(scores), np.std(scores))
            z['y'] = z[cname2]

        print('cv:', cv, np.mean(cv), np.std(cv))
        return cv, None

if '__main__' == __name__:
    pass
