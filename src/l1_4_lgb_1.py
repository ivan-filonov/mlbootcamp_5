#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import model_base_ho_lgb

import data_clean_3_nans as data
#import features_text_2 as fea_1
#import features_misc_1 as fea_2

class model(model_base_ho_lgb.LgbModelBase):
    def __init__(self):
        name = 'l1_4_lgb_1'
        debug = False
        public_score = 0.5432502
        super().__init__(name, data, [], debug, public_score)

def predict():
    return model().predict()

if '__main__' == __name__:
    m = model()
    print('starting', m.now())
    m.main()
    print('done.', m.now())