#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

import state
public_score = None
debug_mode = False
state = state.State('l1_4_rf_1')
import data_clean_2 as data
import features_text_1 as fea_1
import features_misc_1 as fea_2

def run(state, train, y, test, v, z):
    #cname = sys._getframe().f_code.co_name
    cname = 'p'
    train.drop('id', axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    def step_rf(params):
        clf = ensemble.RandomForestRegressor(**params)
        cv = model_selection.cross_val_score(clf,
                                             train, y,
                                             scoring=metrics.make_scorer(metrics.log_loss),
                                             cv = 10,
                                             n_jobs = -2)
        score = np.mean(cv)
        print(cname, score, params)
        return dict(loss=score, status=STATUS_OK)
    space_rf = dict(
        n_estimators = hp.choice('n_estimators', range(50, 1500)),
        #criterion = hp.choice('criterion', ["gini", "entropy"]),
        min_samples_split = hp.choice('min_samples_split', range(2, 10)),
        min_samples_leaf = hp.choice('min_samples_leaf', range(1, 10)),
        max_features = hp.choice('max_features', range(1, 16)),
        random_state = 1
        )
    trs = state.load('rf_trials')
    if trs == None or debug_mode:
        tr = Trials()
    else:
        tr, _ = trs
    if len(tr.trials) > 0:
        print('reusing %d trials, best was:'%(len(tr.trials)), space_eval(space_rf, tr.argmin))
        best = tr.argmin
    while len(tr.trials) < 15:
        best = fmin(step_rf, space_rf, algo=tpe.suggest, max_evals=len(tr.trials) + 1, trials = tr)
        state.save('et_trials', (tr, space_rf))
    rf_params = space_eval(space_rf, best)
    print(rf_params)

    N_splits = 9
    N_seeds = 3

    skf = model_selection.StratifiedKFold(n_splits=N_splits, shuffle=True)
    cv = []
    for s in range(N_seeds):
        scores = []
        cname2 = cname + str(s)
        v[cname2], z[cname2] = 0, 0
        rf_params['random_state'] = s + 4242
        for n, (itrain, ival) in enumerate(skf.split(train, y)):
            clf = ensemble.RandomForestRegressor(**rf_params)
            clf.fit(train.ix[itrain], y[itrain])
            p = clf.predict(train.ix[ival])
            v.loc[ival, cname2] += p
            score = metrics.log_loss(y[ival], p)
            z[cname2]  += clf.predict(test)
            print(cname, 'seed %d step %d of %d: '%(rf_params['random_state'], n+1, skf.n_splits), score, state.now())
            scores.append(score)
        z[cname2] /= N_splits
        cv.append(np.mean(scores))
        print('seed %d loss: '%(rf_params['random_state']), scores, np.mean(scores), np.std(scores))
        z['y'] = z[cname2]

    print('cv:', cv, np.mean(cv), np.std(cv))
    return cv, None

def predict():
    return state.run_model(run, data, [fea_1, fea_2], debug_mode)

if '__main__' == __name__:
    print('starting', state.now())
    state.run_predict(predict, debug_mode, public_score)
    print('done.', state.now())
