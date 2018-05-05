#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Download 1km-mesh data from NARO  """

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


def fetch(element, timedomain, lalodomain, interval):
    ''' Download 1km-mesh data from NARO '''
    try:
        Msh, tim, lat, lon = AMD.GetMetData(element, timedomain, lalodomain)
        print(f'Msh.shape (original, {element}): {Msh.shape}')
        lat = lat[::interval]
        lon = lon[::interval]
        Msh = Msh[:, ::interval, ::interval]
        print(f'Msh.shape (sliced, {element}): {Msh.shape}')
    except OSError:
        print(f'Warning: {element} in {year} is not available. Returned None.')
        return None

    df = pd.DataFrame({
        element: Msh.flatten(),
        'lat': list(np.repeat(lat, len(lon))) * len(tim),
        'lon': list(lon) * len(lat) * len(tim),
        'date': np.repeat(tim, len(lat) * len(lon)),
    })
    print(f'{element} datafram done')

    #set index
    df.set_index(['date', 'lat', 'lon'], inplace=True)
    print(f'{element} set_index done')

    # df[(df.A>0) & (df.B>0)]
    return df


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', '-i', default=10, type=int,
                        help='interval of latitude and longitude')
    args = parser.parse_args()

    #load dataset settings
    with open('dataset.hjson', 'r') as f:
        dataset = hjson.load(f)

    #explore locations
    for pref in dataset.keys():

        #skip if data exists
        # if os.path.exists(os.path.join('meshdata', pref)):
        #     print('skipped', pref)
        #     continue

        #init
        lalodomain = [dataset[pref]['lat0'], dataset[pref]['lat1'],
                      dataset[pref]['lon0'], dataset[pref]['lon1']]
        elements = ('TMP_mea', 'TMP_max', 'TMP_min', 'RH', 'GSR', 'APCP')

        #explore year
        for year in range(1980, 2017):

            #access to server
            timedomain = [f'{year}-05-01', f'{year}-10-31']
            dfs = joblib.Parallel(n_jobs=6, verbose=1)(
                joblib.delayed(fetch)(element, timedomain, lalodomain, args.interval) for element in elements)

            #assert
            # for i in range(1, len(dfs)):
            #     if dfs[i] is None:
            #         continue
            #     for col in ('date', 'lat', 'lon', 'meshcode'):
            #         assert(np.all(dfs[0][col] == dfs[i][col]))

            #concat
            #TODO(kyon): concat by index is slow?
            print('concat...')
            df = pd.concat(dfs)
            df.reset_index(inplace=True)
            df['YEAR'] = df.date.dt.year
            df['DOY'] = df.date.dt.dayofyear
            print('done')

            #rename columns
            df.rename(
                columns={
                    'TMP_mea': 'T2M',
                    'TMP_max': 'T2MX',
                    'TMP_min': 'T2MN',
                    'GSR': 'swv_dwn',
                    'APCP': 'RAIN',
                    'RH': 'RH2M'}, inplace=True)

            #split by meshcode and save as csv
            df_grouped = df.groupby(['lat', 'lon'])
            for (lat, lon), row_idx in df_grouped.groups.items():
                record = df.iloc[row_idx].copy()
                meshcode = AMD.lalo2mesh(lat, lon)
                record['meshcode'] = meshcode
                outdir = os.path.join('meshdata', pref, str(year))
                outcsv = os.path.join(outdir, f'{meshcode}.csv')
                mkdir(outdir)
                with open(outcsv, 'w') as f:
                    f.write('#config - lat:{}\n'.format(lat))
                    f.write('#config - lon:{}\n'.format(lon))
                    record.to_csv(f, float_format='%.3f')
                    #TODO(kyon): tqdm
