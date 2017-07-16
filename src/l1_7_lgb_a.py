#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import model_base_ho_lgb

import data_clean_1 as data
# то же что l1_5_lgb_4 но другой сид
import features_text_2 as fea_2
import features_misc_1 as fea_1

class model(model_base_ho_lgb.LgbModelBase):
    def __init__(self):
        name = 'l1_7_lgb_a'
        debug = False
        public_score = None
        super().__init__(name, data, [fea_1, fea_2], debug, public_score,
             max_ho_trials = 100, num_splits = 25, num_seeds = 5)

def predict():
    return model().predict()

if '__main__' == __name__:
    m = model()
    print('starting', m.now())
    m.main()
    print('done.', m.now())
