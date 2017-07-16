#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import model_base

class model(model_base.L2Model):
    def __init__(self):
        name = 'l2_5_ho_stack'
        debug = False
        public_score = 0.5427385
        l1_models = [
                'l1_1_ho_xgb_1',
                'l1_1_ho_xgb_2',
                'l1_3_ho_xgb_1',
                'l1_3_ho_xgb_2',
                'l1_3_ho_xgb_3',
                'l1_1_keras_2',
                'l1_3_keras_1',
                'l1_3_keras_2',

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
        super().__init__(name, l1_models, debug, public_score)

def predict():
    return model().predict()

if '__main__' == __name__:
    m = model()
    print('starting', m.now())
    m.main()
    print('done.', m.now())
