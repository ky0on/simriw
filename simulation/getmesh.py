#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import os
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
    for key in dataset.keys():

        #init
        lalodomain = [dataset[key]['lat'], dataset[key]['lat'], dataset[key]['lon'], dataset[key]['lon']]
        meshes = defaultdict(lambda: [])

        #explore year
        for year in range(1980, 2016):

            #access to server
            timedomain = ['{}-01-01'.format(year), '{}-12-31'.format(year)]
            for element in ('TMP_mea', 'TMP_max', 'TMP_min', 'RH', 'GSR', 'APCP'):
                try:
                    Msh, tim, lat, lon = AMD.GetMetData(element, timedomain, lalodomain)
                    Msh = Msh[:, 0, 0]     # dimension (from 3 to 1)
                except OSError:
                    print('Warning: {} in {} is not available'.format(element, year))
                    Msh = np.nan
                mesh = pd.DataFrame({'DATE': tim, element: Msh})
                mesh.set_index('DATE', inplace=True)
                meshes[element].append(mesh)

        #concat
        meshes = [pd.concat(meshes[element]) for element in meshes.keys()]
        mesh = pd.concat(meshes, axis=1)

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
        outcsv = os.path.join('dataset', key + '.csv')
        with open(outcsv, 'w') as f:
            f.write('#config - lat:{}\n'.format(dataset[key]['lat']))
            f.write('#config - lon:{}\n'.format(dataset[key]['lon']))
            mesh.to_csv(f)
        print('saved as', outcsv)
