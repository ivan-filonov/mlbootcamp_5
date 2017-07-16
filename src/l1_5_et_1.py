#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import model_base_et_ho

import data_clean_1 as data
#import features_text_2 as fea_1
#import features_misc_1 as fea_2

class model(model_base_et_ho.EtModelBase):
    def __init__(self):
        name = 'l1_5_et_1'
        debug = False
        public_score = None
        super().__init__(name, data, [], debug, public_score)

def predict():
    return model().predict()

if '__main__' == __name__:
    m = model()
    print('starting', m.now())
    m.main()
    print('done.', m.now())