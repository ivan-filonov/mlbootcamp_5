#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import model_base_ho_xgb

import data_clean_3_nans as data
import features_text_2 as fea_1
import features_misc_1 as fea_2

class model(model_base_ho_xgb.XgbModelBase):
    def __init__(self):
        name = 'l1_5_xgb_1'
        debug = False
        public_score = None
        super().__init__(name, data, [fea_1, fea_2], debug, public_score)

def predict():
    return model().predict()

if '__main__' == __name__:
    m = model()
    print('starting', m.now())
    m.main()
    print('done.', m.now())