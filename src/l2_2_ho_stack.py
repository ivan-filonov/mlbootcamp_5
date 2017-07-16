#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection

import state
public_score = 0.5435731
state = state.State('l2_1_ho_stack')

def run(train, y, test, v, z):
    np.random.seed(1)
    #cname = sys._getframe().f_code.co_name
    train = train.values
    test = test.values

    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    space_stack = hp.choice('stacking by', [
            dict( type = 'BayesianRidge' ),
            dict( type = 'Lars' ),
            dict( type = 'LinearRegression' ),
            dict( type = 'Ridge' ),
            dict( type = 'SGDRegressor', random_state = 1 ),
            dict( type = 'XGBRegressor',
                 max_depth = hp.choice('max_depth', range(2, 8)),
                 subsample = hp.quniform('subsample', 0.6, 1, 0.05),
                 colsample_bytree = hp.quniform('colsample_bytree', 0.6, 1, 0.05),
                 learning_rate = hp.quniform('learning_rate', 0.005, 0.03, 0.005),
                 min_child_weight = hp.quniform('min_child_weight', 1, 6, 1),
                 gamma = hp.quniform('gamma', 0, 10, 0.05),
                 reg_alpha = hp.quniform('alpha', 0, 1, 0.0001),
                 ),
                                            ])

    def get_lr(params):
        t = params['type']
        del params['type']

        if t == 'BayesianRidge':
            lr = linear_model.BayesianRidge(**params)
        elif t == 'Lars':
            lr = linear_model.Lars(**params)
        elif t == 'LinearRegression':
            lr = linear_model.LinearRegression(**params)
        elif t == 'Ridge':
            lr = linear_model.Ridge(**params)
        elif t == 'SGDRegressor':
            lr = linear_model.SGDRegressor(**params)
        elif t == 'XGBRegressor':
            lr = xgb.XGBRegressor(**params)

        return lr

    def step(params):
        print(params, end = ' ')
        cv = model_selection.cross_val_score(get_lr(params),
                                             train, y,
                                             cv=10,
                                             scoring=metrics.make_scorer(metrics.log_loss))
        score = np.mean(cv)
        print(score)
        return dict(loss=score, status=STATUS_OK)

    trs = state.load('trials')
    if trs == None:
        tr = Trials()
    else:
        tr, _ = trs
    if len(tr.trials) > 0:
        best = tr.argmin
        print('reusing %d trials, best was:'%(len(tr.trials)), space_eval(space_stack, best))
    mt = max(50, len(tr.trials) + 1)
    while len(tr.trials) < min(50, mt):
        best = fmin(step, space_stack, algo=tpe.suggest, max_evals=len(tr.trials) + 1, trials = tr)
        state.save('trials', (tr, space_stack))
    params = space_eval(space_stack, best)

    print('best params:', params)
    lr = get_lr(params)
    cv = model_selection.cross_val_score(lr,
                                         train, y,
                                         cv=10,
                                         scoring=metrics.make_scorer(metrics.log_loss))
    lr.fit(train, y)
    z['p'] = np.clip(lr.predict(test), 1e-5, 1-1e-5)
    z['y'] = z['p']
    v['p'] = model_selection.cross_val_predict(lr,
                                         train, y,
                                         cv=10)
    print('cv:', np.mean(cv), np.std(cv))
    return cv, None

def predict():
    saved = state.load('model')
    #saved = None
    if saved == None:
        import l1_1_ho_xgb_1
        import l1_1_keras_1
        import l1_1_ho_xgb_2
        import l1_1_keras_2

        vs, zs, cvs = [], [], []
        for module in [l1_1_ho_xgb_1, l1_1_ho_xgb_2,
                       l1_1_keras_1, l1_1_keras_2]:
            v, z, cv, _ = module.predict()
            vs.append(v)
            zs.append(z)
            cvs.append(cv)

        z = pd.DataFrame()
        z['id'] = zs[-1].id
        z['y'] = 0

        v = pd.DataFrame()
        v['id'] = vs[-1].id
        v['y'] = vs[-1].y

        for s in vs + zs:
            s.drop(['id', 'y'], axis=1, inplace=True)

        train = pd.concat(vs, axis=1)
        test = pd.concat(zs, axis=1)
        y = v.y

        cv, _ = run(train, y, test, v, z)
        #state.save('model', (v, z, cv, None))
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
