#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import os
import hjson
import joblib
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from common import mkdir

import AMD_Tools3 as AMD

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '08 Oct 2017'


def fetch(element, timedomain, lalodomain):
    try:
        Msh, tim, lat, lon = AMD.GetMetData(element, timedomain, lalodomain)
    except OSError:
        print('Warning: {} in {} is not available. Filled with nan'.format(element, year))
        ndays = AMD.timrange(*timedomain)
        Msh = np.empty([len(ndays), 100, 100])   # approximate sizes of lat and lon
        Msh[:, :] = None
        lat = None
        lon = None
        tim = None

    dic = {
        'lat': lat,
        'lon': lon,
        'tim': tim,
        'mesh': Msh,
        'element': element,
    }
    return dic


def split_by_meshcode(lat, lon, tim, y, x, year, pref, df):
    ''' '''
    mesh = {d['element']: d['mesh'][:, y, x] for d in df}
    mesh['DATE'] = tim
    mesh = pd.DataFrame(mesh)
    mesh.set_index('DATE', inplace=True)
    dic = {
        'lat': lat[y],
        'lon': lon[x],
        'mesh': mesh,
        'year': year,
        'pref': pref,
    }

    #rename columns
    mesh = dic['mesh']
    mesh['DOY'] = mesh.index.dayofyear
    mesh['YEAR'] = mesh.index.year
    mesh.rename(columns={
        'TMP_mea': 'T2M',
        'TMP_max': 'T2MX',
        'TMP_min': 'T2MN',
        'GSR': 'swv_dwn',
        'APCP': 'RAIN',
        'RH': 'RH2M'}, inplace=True)

    #save
    outdir = os.path.join('dataset', dic['pref'], str(dic['year']))
    outcsv = os.path.join(outdir, f'{dic["lat"]:.3f}x{dic["lon"]:.3f}.csv')
    mkdir(outdir)
    with open(outcsv, 'w') as f:
        f.write('#config - lat:{}\n'.format(dic['lat']))
        f.write('#config - lon:{}\n'.format(dic['lon']))
        mesh.to_csv(f, float_format='%.3f')
    # print('saved as', outcsv)


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    #load dataset settings
    with open('dataset.hjson', 'r') as f:
        dataset = hjson.load(f)

    #explore locations
    for pref in dataset.keys():

        #filter by prefecture
        if pref != 'yamanashi':
            continue

        #init
        lalodomain = [dataset[pref]['lat0'], dataset[pref]['lat1'],
                      dataset[pref]['lon0'], dataset[pref]['lon1']]
        elements = ('TMP_mea', 'TMP_max', 'TMP_min', 'RH', 'GSR', 'APCP')

        #explore year
        for year in range(1980, 2017):

            #access to server
            timedomain = [f'{year}-05-01', f'{year}-12-31']
            df = joblib.Parallel(n_jobs=6, verbose=1)(joblib.delayed(fetch)(element, timedomain, lalodomain) for element in elements)

            #split by meshcode
            lat = df[0]['lat']
            lon = df[0]['lon']
            tim = df[0]['tim']
            joblib.Parallel(n_jobs=-1, verbose=10)(
                joblib.delayed(split_by_meshcode)(
                    lat, lon, tim, y, x, year, pref, df) for y in range(len(lat)) for x in range(len(lon)))
