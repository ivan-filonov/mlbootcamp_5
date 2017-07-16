#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import model_base_ho_xgb

import data_clean_1 as data
#import features_text_2 as fea_1
#import features_misc_2 as fea_2
import features_wa_4 as fea_3

class model(model_base_ho_xgb.XgbModelBase):
    def __init__(self):
        name = 'l1_6_xgb_6'
        debug = False
        public_score = 0.5441663
        super().__init__(name, data, [fea_3], debug, public_score)

def predict():
    return model().predict()

if '__main__' == __name__:
    m = model()
    print('starting', m.now())
    m.main()
    print('done.', m.now())
