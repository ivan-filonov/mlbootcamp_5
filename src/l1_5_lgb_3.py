#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import model_base_ho_lgb

import data_clean_1 as data
import features_text_1 as fea_1
import features_misc_2 as fea_2

class model(model_base_ho_lgb.LgbModelBase):
    def __init__(self):
        name = 'l1_5_lgb_3'
        debug = False
        public_score = 0.5433765
        super().__init__(name, data, [fea_1, fea_2], debug, public_score)

def predict():
    return model().predict()

if '__main__' == __name__:
    m = model()
    print('starting', m.now())
    m.main()
    print('done.', m.now())
