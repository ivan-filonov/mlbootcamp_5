#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import metrics
from sklearn import model_selection

import state
public_score = 0.5436294
state = state.State('l1_1_ho_xgb_2')
import data_clean_1 as data

def run(train, y, test, v, z):
    #cname = sys._getframe().f_code.co_name
    cname = 'p'
    train.drop('id', axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    dtrain = xgb.DMatrix(train, y)
    def step_xgb(params):
        cv = xgb.cv(params=params,
                    dtrain=dtrain,
                    num_boost_round=10000,
                    early_stopping_rounds=50,
                    nfold=10,
                    seed=params['seed'])
        score = cv.ix[len(cv)-1, 0]
        print(cname, score, len(cv), params)
        return dict(loss=score, status=STATUS_OK)
    space_xgb = dict(
            max_depth = hp.choice('max_depth', range(2, 8)),
            subsample = hp.quniform('subsample', 0.6, 1, 0.05),
            colsample_bytree = hp.quniform('colsample_bytree', 0.6, 1, 0.05),
            learning_rate = hp.quniform('learning_rate', 0.005, 0.03, 0.005),
            min_child_weight = hp.quniform('min_child_weight', 1, 6, 1),
            gamma = hp.quniform('gamma', 0.5, 10, 0.05),

            objective = 'binary:logistic',
            eval_metric = 'logloss',
            seed = 1,
            silent = 1
        )
    trs = state.load('xgb_trials')
    if trs == None:
        tr = Trials()
    else:
        tr, _ = trs
    if len(tr.trials) > 0:
        print('reusing %d trials, best was:'%(len(tr.trials)), space_eval(space_xgb, tr.argmin))
        best = tr.argmin
    while len(tr.trials) < 15:
        best = fmin(step_xgb, space_xgb, algo=tpe.suggest, max_evals=len(tr.trials) + 1, trials = tr)
        state.save('xgb_trials', (tr, space_xgb))
    xgb_params = space_eval(space_xgb, best)
    print(xgb_params)

    N_splits = 9
    N_seeds = 1

    skf = model_selection.StratifiedKFold(n_splits=N_splits, shuffle=True)
    dtest = xgb.DMatrix(test)
    for s in range(N_seeds):
        scores = []
        cname2 = cname + str(s)
        v[cname2], z[cname2] = 0, 0
        xgb_params['seed'] = s + 4242
        for n, (itrain, ival) in enumerate(skf.split(train, y)):
            dtrain = xgb.DMatrix(train.ix[itrain], y[itrain])
            dvalid = xgb.DMatrix(train.ix[ival], y[ival])
            watch = [(dtrain, 'train'), (dvalid, 'valid')]
            clf = xgb.train(xgb_params, dtrain, 10000, watch, early_stopping_rounds=100, verbose_eval=False)

            p = clf.predict(dvalid)
            v.loc[ival, cname2] += p
            score = metrics.log_loss(y[ival], p)
            z[cname2]  += clf.predict(dtest)
            print(cname, 'seed %d step %d of %d: '%(xgb_params['seed'], n+1, skf.n_splits), score, state.now())
            scores.append(score)
        z[cname2] /= N_splits

    cv = scores
    z['y'] = z[cname2]
    print('validation loss: ', cv, np.mean(cv), np.std(cv))

    return cv, None

def predict():
    saved = state.load('model')
    #saved = None
    if saved == None:
        train, y, test, _ = data.get()
        z = pd.DataFrame()
        z['id'] = test.id
        z['y'] = 0

        v = pd.DataFrame()
        v['id'] = train.id
        v['y'] = y
        cv, _ = run(train, y, test, v, z)
        state.save('model', (v, z, cv, None))
    else:
        v, z, cv, _ = saved
    return v, z, cv, _

if '__main__' == __name__:
    print('starting', state.now())
    v, z, cv, _ = predict()
    state.save_model(v, z, cv)
    if public_score == None:
        # если есть public score - перезаписывать отправленное уже не стоит
        state.save_predicts(z)
    else:
        import os
        if os.path.exists('../model_scores.csv'):
            mdf = pd.read_csv('../model_scores.csv')
        else:
            mdf = pd.DataFrame(columns=['timestamp', 'model', 'cv', 'cv std', 'public score'])
        idx = mdf.model == state.base_name_
        if np.sum(idx) == 0:
            mdf.loc[len(mdf), 'model'] = state.base_name_
            idx = mdf.model == state.base_name_
        if (mdf.ix[idx, 'public score'] != public_score).bool():
            mdf.ix[idx, 'public score'] = public_score
            mdf.ix[idx, 'timestamp'] = state.now()
            mdf.ix[idx, 'cv'] =  np.mean(cv)
            mdf.ix[idx, 'cv std'] = np.std(cv)
        mdf.to_csv('../model_scores.csv', index=None)
    print('done.', state.now())
