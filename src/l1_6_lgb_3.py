#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import model_base_ho_lgb

import data_clean_3_nans as data
import features_text_2 as fea_1
#import features_misc_1 as fea_2
import features_wa_3 as fea_3

class model(model_base_ho_lgb.LgbModelBase):
    def __init__(self):
        name = 'l1_6_lgb_3'
        debug = False
        public_score = None
        super().__init__(name, data, [fea_1, fea_3], debug, public_score)

def predict():
    return model().predict()

if '__main__' == __name__:
    m = model()
    print('starting', m.now())
    m.main()
    print('done.', m.now())
