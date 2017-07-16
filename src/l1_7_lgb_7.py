#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import model_base_ho_lgb

import data_xgb_smoke as data
#import features_text_2 as fea_1
import features_kmeans_2 as fea_2
#import features_wa_4 as fea_3

class model(model_base_ho_lgb.LgbModelBase):
    def __init__(self):
        name = 'l1_7_lgb_7'
        debug = False
        public_score = None
        super().__init__(name, data, [fea_2], debug, public_score)

def predict():
    return model().predict()

if '__main__' == __name__:
    m = model()
    print('starting', m.now())
    m.main()
    print('done.', m.now())
