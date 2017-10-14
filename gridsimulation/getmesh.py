#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import hjson
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

import AMD_Tools3 as AMD

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '08 Oct 2017'


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
        meshes = {}   # key=name

        #explore year
        for year in range(1980, 2017):

            #access to server
            elements = ('TMP_mea', 'TMP_max', 'TMP_min', 'RH', 'GSR', 'APCP')
            timedomain = ['{}-01-01'.format(year), '{}-12-31'.format(year)]
            for element in elements:
                try:
                    Msh, tim, lat, lon = AMD.GetMetData(element, timedomain, lalodomain)

                except OSError:
                    print('Warning: {} in {} is not available. Filled with nan'.format(element, year))
                    Msh = np.empty([len(tim), len(lat), len(lon)])   # assuming other element in the same year has been downloaded just before
                    Msh[:, :] = np.nan

                #split by meshcode
                for y in range(len(lat)):
                    for x in range(len(lon)):
                        mesh = pd.DataFrame({'DATE': tim, element: Msh[:, y, x]})
                        mesh.set_index('DATE', inplace=True)
                        name = '{0}_{1:0>3}_{2:0>3}'.format(pref, y, x)
                        if name not in meshes.keys():
                            meshes[name] = {
                                'lat': lat[y],
                                'lon': lon[x],
                                'mesh': defaultdict(lambda: []),
                            }
                        meshes[name]['mesh'][element].append(mesh)

        #save as csv
        for name, dic in meshes.items():

            #concat
            _mesh = [pd.concat(dic['mesh'][element]) for element in elements]
            mesh = pd.concat(_mesh, axis=1)

            #rename columns
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
            outcsv = 'dataset/{}.csv'.format(name)
            with open(outcsv, 'w') as f:
                f.write('#config - lat:{}\n'.format(dic['lat']))
                f.write('#config - lon:{}\n'.format(dic['lon']))
                mesh.to_csv(f)
            print('saved as', outcsv)
