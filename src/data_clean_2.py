#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

def cleanup(df_all):
    idx = (df_all.height == 157).values * (df_all.weight == 10).values
    df_all.ix[idx, 'weight'] = 40

    idx = (df_all.height == 169).values * (df_all.weight == 10).values
    df_all.ix[idx, 'weight'] = 70

    idx = (df_all.height == 165).values * (df_all.weight == 10).values
    df_all.ix[idx, ['weight', 'ap_lo']] = 100, 110

    idx = (df_all.height == 178).values * (df_all.weight == 11).values
    df_all.ix[idx, 'weight'] = 110

    idx = (df_all.height == 183).values * (df_all.weight == 13).values
    df_all.ix[idx, 'weight'] = 130

    idx = (df_all.height == 169).values * (df_all.weight == 16.3).values
    df_all.ix[idx, 'weight'] = 63

    idx = (df_all.height == 170).values * (df_all.weight == 20).values
    df_all.ix[idx, 'weight'] = 80

    idx = (df_all.height == 162).values * (df_all.weight == 21).values
    df_all.ix[idx, 'weight'] = 71

    idx = (df_all.height == 177).values * (df_all.weight == 22).values
    df_all.ix[idx, 'weight'] = 80

    idx = (df_all.height == 157).values * (df_all.weight == 23).values
    df_all.ix[idx, 'weight'] = 43

    idx = (df_all.height == 171).values * (df_all.weight == 29).values
    df_all.ix[idx, 'weight'] = 49

    idx = (df_all.height == 58).values * (df_all.weight == 183).values
    df_all.ix[idx, ['height', 'weight', 'ap_lo']] = 158, 83, 100

    idx = (df_all.weight == 178).values * (df_all.height == 80).values
    df_all.ix[idx, 'height'] = 180

    idx = (df_all.height == 87).values * (df_all.weight == 173).values
    df_all.ix[idx, ['height', 'weight']] = 187, 113

    idx = (df_all.weight == 170).values * (df_all.height == 97).values
    df_all.ix[idx, 'height'] = 197

    idx = (df_all.height < 100).values * (df_all.weight == 168).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.weight == 168).values
    df_all.ix[idx, 'weight'] -= 100

    idx = (df_all.height == 104).values * (df_all.weight == 159).values
    df_all.ix[idx, 'height'] = 164

    idx = (df_all.height == 75).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 68).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 65).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 60).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 57).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 59).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 76).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 66).values
    df_all.ix[idx, ['height', 'ap_hi']] = 166, 120

    idx = (df_all.height == 81).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 55).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 71).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 74).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 70).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 52).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 50).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 56).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 72).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 62).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 64).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 67).values
    df_all.ix[idx, 'height'] += 100

    idx = (df_all.height == 91).values
    df_all.ix[idx, ['height', 'weight']] += 155, 91

    idx = (df_all.height == 96).values
    df_all.ix[idx, 'height'] = 156

    idx = (df_all.height == 98).values
    df_all.ix[idx, ['height', 'ap_hi']] = 158, 120

    idx = (df_all.height == 99).values
    df_all.ix[idx, 'height'] = 159

    idx = (df_all.height == 100).values
    df_all.ix[idx, 'height'] = 160

    idx = (df_all.height == 112).values
    df_all.ix[idx, 'height'] = 172

    idx = (df_all.ap_lo == 170).values * (df_all.ap_hi == 20).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 200, 170

    idx = (df_all.ap_lo == 0).values * (df_all.ap_hi == 108).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 100, 80

    idx = (df_all.ap_lo == 0).values * (df_all.ap_hi == 117).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 110, 70

    idx = (df_all.ap_lo == 0).values * (df_all.ap_hi == 118).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 110, 80

    idx = (df_all.ap_lo == 0).values * (df_all.ap_hi == 138).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 130, 80

    idx = (df_all.ap_lo == 0).values * (df_all.ap_hi == 148).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 140, 80

    idx = (df_all.ap_lo == 0).values * (df_all.ap_hi == 149).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 140, 90

    idx = (df_all.ap_hi == 20).values * (df_all.ap_lo == 80).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 120, 80

    idx = (df_all.ap_hi == 20).values * (df_all.ap_lo == 90).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 120, 90

    idx = (df_all.ap_hi == 172).values * (df_all.ap_lo == 190).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 172, 90

    idx = (df_all.ap_lo == 30).values
    df_all.ix[idx, 'ap_lo'] = 90

    idx = (df_all.ap_hi == 7).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.ap_lo == 12).values
    df_all.ix[idx, 'ap_lo'] = 120

    idx = (df_all.ap_hi == 100).values * (df_all.ap_lo == 0).values
    df_all.ix[idx, 'ap_lo'] = 60

    idx = (df_all.ap_hi == 120).values * (df_all.ap_lo == 0).values
    df_all.ix[idx, 'ap_lo'] = 80

    idx = (df_all.ap_hi == 130).values * (df_all.ap_lo == 0).values
    df_all.ix[idx, 'ap_lo'] = 90 - 10 * df_all.ix[idx, 'active']

    idx = (df_all.ap_lo == 0).values
    df_all.ix[idx, 'ap_lo'] = 90

    idx = (df_all.ap_lo == 10).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo == 15).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 150, 70

    idx = (df_all.ap_lo == 19).values
    df_all.ix[idx, 'ap_lo'] = 90

    idx = (df_all.ap_lo == 20).values * (df_all.ap_hi > 99).values * (df_all.ap_hi < 131).values
    df_all.ix[idx, 'ap_lo'] = 70

    idx = (df_all.ap_lo == 20).values * (df_all.ap_hi == 180).values
    df_all.ix[idx, 'ap_lo'] = 120

    idx = (df_all.ap_hi == 19).values
    df_all.ix[idx, 'ap_hi'] = 190

    idx = (df_all.ap_hi == 11).values * (df_all.ap_lo == 70).values
    df_all.ix[idx, 'ap_hi'] = 110

    idx = (df_all.ap_hi == 11).values * (df_all.ap_lo == 71).values
    df_all.ix[idx, 'ap_hi'] = 110

    idx = (df_all.ap_hi == 11).values * (df_all.ap_lo == 60).values
    df_all.ix[idx, 'ap_hi'] = 110

    idx = (df_all.ap_hi == 11).values * (df_all.ap_lo == 80).values
    df_all.ix[idx, 'ap_hi'] = 110

    idx = (df_all.ap_hi == 11).values * (df_all.ap_lo == 120).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 110, 70

    idx = (df_all.ap_hi == 10).values * (df_all.ap_lo == 160).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 100, 60

    idx = (df_all.ap_hi == 10).values
    df_all.ix[idx, 'ap_hi'] = 100

    idx = (df_all.ap_hi == 12).values * (df_all.ap_lo < 100).values
    df_all.ix[idx, 'ap_hi'] = 120

    idx = (df_all.ap_hi == 12).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 140, 120

    idx = (df_all.ap_hi == 13).values
    df_all.ix[idx, 'ap_hi'] = 130

    idx = (df_all.ap_hi == 14).values
    df_all.ix[idx, 'ap_hi'] = 140

    idx = (df_all.ap_hi == 15).values
    df_all.ix[idx, 'ap_hi'] = 150

    idx = (df_all.ap_hi == 16).values * (df_all.ap_lo == 10).values
    df_all.ix[idx, ['ap_hi', 'ap_lo']] = 160, 100

    idx = (df_all.ap_hi == 16).values
    df_all.ix[idx, 'ap_hi'] = 160

    idx = (df_all.ap_lo == 10).values
    df_all.ix[idx, 'ap_lo'] = 100

    idx = (df_all.ap_lo < 10).values
    df_all.ix[idx, 'ap_lo'] *= 10


import data_clean_extreme_ap as data_src
import data_source_base
class dataset(data_source_base.DataSource):
    def __init__(self):
        name = 'data_clean_2'
        save_path = None
        super().__init__(name, save_path)

    def build(self):
        train, y, test, _ = data_src.get()
        ntrain = len(train)
        test['y'] = -1
        df_all = pd.concat([train, test])
        cleanup(df_all)

        df_all.drop('id', axis=1)

        train = df_all[:ntrain].reindex().drop('y', axis=1)
        test = df_all[ntrain:].reindex().drop('y', axis=1)
        return train, y, test, None

d = dataset()
def get():
    return d.get_data()

if '__main__' == __name__:
    d.main()