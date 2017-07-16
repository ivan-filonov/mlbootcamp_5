#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from model_base_3 import L2Model2

class model(L2Model2):
    def __init__(self):
        name = 'l3_ho_stack'
        debug = False
        public_score = 0.5435568
        preselect_features = True#TODO: врубить для финального сабмита!!! Или не врубать - может быть лютый оверфит!!!
        hyperopt_rounds = 500
        n_fold = 30

        l1_models = [
                'l2_2_ho_stack',
                'l2_3_ho_stack',
                'l2_4_ho_stack',
                'l2_5_ho_stack',
                'l2_5_ho_stack_2',
                'l2_6_ho_stack',
                'l2_6_ho_stack_2',
                'l2_7_ho_stack',
                'l2_7_ho_stack_2',
                'l2_8_ho_stack',
                'l2_9_ho_stack',
                'l2_a_ho_stack',
                'l2_b_ho_stack',
                'l2_c_ho_stack',
                'l2_d_ho_stack',
                'l2_e_ho_stack',
                'l2_f_ho_stack',
                'l2_g_ho_stack',
                'l2_h_ho_stack',
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
