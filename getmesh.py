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
from tqdm import tqdm, trange

from utils import mkdir

import AMD_Tools3 as AMD

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '08 Oct 2017'


def fetch(element, timedomain, lalodomain, interval):
    ''' Download 1km-mesh data from NARO '''
    try:
        Msh_org, tim, lat, lon = AMD.GetMetData(element, timedomain, lalodomain)
        lat = lat[::interval]
        lon = lon[::interval]
        Msh = Msh_org[:, ::interval, ::interval]
        print(f'Msh.shape ({element}): {Msh_org.shape} -> {Msh.shape}')
        print(f'null ratio ({element}):', round(pd.isnull(Msh).mean(), 2))
    except OSError:
        print(f'Warning: {element} in {year} is not available. Returned None.')
        return None
    except TypeError:
        print(f'Warning: {element} in {year} is not available. Returned None.')
        return None

    df = pd.DataFrame({
        element: Msh.flatten(),
        'lat': list(np.repeat(lat, len(lon))) * len(tim),
        'lon': list(lon) * len(lat) * len(tim),
        'date': np.repeat(tim, len(lat) * len(lon)),
    })
    # print(f'{element} dataframe done')

    #set index
    df.set_index(['date', 'lat', 'lon'], inplace=True)
    # print(f'{element} set_index done')

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
    for pref in tqdm(dataset.keys(), desc='Area'):

        #debug
        # if pref != 'akita':
        #     continue

        #skip if data exists
        # if os.path.exists(os.path.join('meshdata', pref)):
        #     print('skipped', pref)
        #     continue

        #init
        lalodomain = [dataset[pref]['lat0'], dataset[pref]['lat1'],
                      dataset[pref]['lon0'], dataset[pref]['lon1']]
        elements = ('TMP_mea', 'TMP_max', 'TMP_min', 'RH', 'GSR', 'APCP')

        #explore year
        for year in trange(1980, 2017, desc='Year'):

            #access to server
            timedomain = [f'{year}-05-01', f'{year}-10-31']
            dfs = joblib.Parallel(n_jobs=2, verbose=1)(
                joblib.delayed(fetch)(element, timedomain, lalodomain, args.interval) for element in elements)

            #assert
            # for i in range(1, len(dfs)):
            #     if dfs[i] is None:
            #         continue
            #     for col in ('date', 'lat', 'lon', 'meshcode'):
            #         assert(np.all(dfs[0][col] == dfs[i][col]))

            #concat
            df = pd.concat(dfs, axis=1)
            df.reset_index(inplace=True)
            df['YEAR'] = df.date.dt.year
            df['DOY'] = df.date.dt.dayofyear

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
            for (lat, lon), row_idx in tqdm(df_grouped.groups.items(), desc='meshcode'):
                record = df.iloc[row_idx].copy()

                #skip if on the sea (temperature will be null)
                if np.all(pd.isnull(record.T2M)):
                    # print(f'({lat},{lon}) is on the sea, meshdata has not been saved.')
                    continue

                #add RH2M if not exists
                if 'RH2M' not in record.columns:
                    record['RH2M'] = None

                meshcode = AMD.lalo2mesh(lat, lon)
                record['meshcode'] = meshcode
                outdir = os.path.join('meshdata', pref, str(year))
                outcsv = os.path.join(outdir, f'{meshcode}.csv')
                mkdir(outdir)
                with open(outcsv, 'w') as f:
                    f.write('#config - lat:{}\n'.format(lat))
                    f.write('#config - lon:{}\n'.format(lon))
                    record.to_csv(f, float_format='%.3f', index=False)
