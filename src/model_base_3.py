import numpy as np
import pandas as pd

from sklearn import base
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection

from functools import partial

from joblib import Parallel, delayed

from model_base import Model

def split_validate_job(bst, X, y, seed):
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.3, random_state=seed)
    bst.fit(xtrain, ytrain)
    p = bst.predict(xtest)
    return metrics.log_loss(ytest, p)

class L2Model2(Model):
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
        for s1, s2, n in zip(vs, zs, self.l1_models_):
            s1.drop(['id', 'y'], axis=1, inplace=True)
            s2.drop(['id', 'y'], axis=1, inplace=True)
            s1.columns = [n + '_' + c for c in s1.columns]
            s2.columns = [n + '_' + c for c in s2.columns]


        self.train_ = pd.concat(vs, axis=1)
        self.test_ = pd.concat(zs, axis=1)

    def ccv(self, bst, X, y, scorer):
        cv1 = model_selection.cross_val_score(bst, X, y, cv=self.n_fold_, n_jobs=-2, scoring = scorer)
        tasks = [delayed(split_validate_job)(base.clone(bst), X, y, seed) for seed in range(self.n_fold_)]
        cv2 = Parallel(n_jobs=-2, backend="threading")(tasks)
        score = (np.sum(cv1) + np.sum(cv2)) / (len(cv1) + len(cv2))
        std = np.std(list(cv1) + list(cv2))
        return score, std

    def greedy_select_features(self):
        print('initial shapes:', self.train_.shape, self.test_.shape)
        saved = None if self.debug_ else self.load('chosen_features')

        if saved == None:
            g_best_score = 1e9
            g_best_features = []
            current = set()
            finished = False
        else:
            g_best_features, g_best_score, finished = saved
            current = set(g_best_features)
            print('SFS REUSE:', g_best_score, len(current), sorted(g_best_features), self.now())


        if not finished:
            col_names = self.train_.columns
            y = self.y_.ravel()
            scorer = metrics.make_scorer(metrics.log_loss)
            loop_count = len(col_names) - len(g_best_features)
            for _ in range(loop_count):
                avail = set(col_names).difference(current)
                best_score = 1e9
                best_features = None
                for f in avail:
                    newf = list(current | {f})
                    score, _ = self.ccv(linear_model.BayesianRidge(), self.train_[newf], y, scorer)
                    if best_score > score:
                        best_score = score
                        best_features = newf
                current = set(best_features)
                if g_best_score > best_score:
                    g_best_score = best_score
                    g_best_features = best_features
                    print('new best:', g_best_score, sorted(g_best_features), self.now())
                else:
                    print('no luck', len(current), self.now())
                if len(best_features) - len(g_best_features) >= 5:
                    break
                self.save('chosen_features', (g_best_features, g_best_score, False))
            # now
            self.save('chosen_features', (g_best_features, g_best_score, True))

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

        self.baseline_score_, _ = self.ccv(linear_model.BayesianRidge(), train, y, metrics.make_scorer(metrics.log_loss))
        self.baseline_stacker_ = linear_model.BayesianRidge()

        for fit_intercept in [False, True]:
            for normalize in [False, True]:
                lr = linear_model.LinearRegression(fit_intercept=fit_intercept,
                                                   normalize=normalize)
                score, _ = self.ccv(lr, train, y, metrics.make_scorer(metrics.log_loss))
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
            score, _ = self.ccv(get_lr(params), train, y, metrics.make_scorer(metrics.log_loss))
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
