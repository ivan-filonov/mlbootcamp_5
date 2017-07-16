#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from model_base_3 import L2Model2

class model(L2Model2):
    def __init__(self):
        name = 'l2_g_ho_stack'
        debug = False
        public_score = None
        preselect_features = False #TODO: врубить для финального сабмита!!! Или не врубать - может быть лютый оверфит!!!
        hyperopt_rounds = 500
        n_fold = 30

        '''
                'l1_1_ho_xgb_1',
                'l1_1_ho_xgb_2',
                'l1_3_ho_xgb_1',
                'l1_3_ho_xgb_2',
                'l1_3_ho_xgb_3',
                'l1_1_keras_2',
                'l1_3_keras_1',
                'l1_3_keras_2',
        '''
        l1_models = [

                'l1_4_et_1',
                'l1_4_et_2',
                'l1_4_keras_1',
                'l1_4_keras_2',
                'l1_4_keras_3',
                'l1_4_keras_4',
                'l1_4_lgb_1',
                'l1_4_rf_1',
                'l1_4_rf_2',
                'l1_4_rf_3',
                'l1_4_xgb_1',
                'l1_4_xgb_2',
                'l1_4_xgb_3',

                'l1_5_et_1',
                'l1_5_keras_1',
                'l1_5_keras_2',
                'l1_5_lgb_1',
                'l1_5_lgb_2',
                'l1_5_lgb_3',
                'l1_5_lgb_4',
                'l1_5_lgb_5',
                'l1_5_lgb_6',
                'l1_5_lgb_7',
                'l1_5_lgb_8',
                'l1_5_rf_1',
                'l1_5_xgb_1',
                'l1_5_xgb_2',
                'l1_5_xgb_3',
            ]
        super().__init__(name, l1_models, debug, public_score,
             preselect_features, hyperopt_rounds, n_fold)

def predict():
    return model().predict()

if '__main__' == __name__:
    m = model()
    print('starting', m.now())
    m.main()
    print('done.', m.now())
