#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import re
import os
import hjson
import argparse
import pandas as pd

import AMD_Tools3 as AMD

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '08 Oct 2017'


def has_key(keys, dict):
    for key in keys:
        if key not in dict.keys():
            return False
    return True


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', type=str, default='./history.hjson')
    args = parser.parse_args()

    #extract
    with open(args.path, 'r') as f:
        history_all = hjson.load(f)

    #save
    for key in history_all.keys():

        #filter
        if has_key(['planting_date', 'harvesting_date', 'yield', 'gps'], history_all[key]):
            # print(key)
            history = {}
            history['planting_date'] = history_all[key]['planting_date'][:10]
            history['harvesting_date'] = history_all[key]['harvesting_date'][:10]
            history['yield'] = history_all[key]['yield']
            history['density'] = history_all[key]['planting_density']
            history['lat'] = float(history_all[key]['gps'].split(',')[0])
            history['lon'] = float(history_all[key]['gps'].split(',')[1])
        else:
            continue

        #get mesh
        timedomain = [history['planting_date'], history['harvesting_date']]
        lalodomain = [history['lat'], history['lat'], history['lon'], history['lon']]
        meshes = []
        for element in ('TMP_mea', 'TMP_max', 'TMP_min', 'RH', 'GSR', 'APCP'):
            Msh, tim, lat, lon = AMD.GetMetData(element, timedomain, lalodomain)
            Msh = Msh[:, 0, 0]     # dimension (from 3 to 1)
            mesh = pd.DataFrame({'date': tim, element: Msh})
            mesh.set_index('date', inplace=True)
            meshes.append(mesh)
        mesh = pd.concat(meshes, axis=1)
        mesh['DOY'] = mesh.index.dayofyear
        mesh['YEAR'] = mesh.index.year
        mesh.rename(columns={
            'TMP_mea': 'T2M',
            'TMP_max': 'T2MX',
            'TMP_min': 'T2MN',
            'GSR': 'swv_dwn',
            'APCP': 'RAIN',
            'RH': 'RH2M'}, inplace=True)
        history['mesh'] = mesh.to_json()

        #save
        csv = '#config - lat:{}\n'.format(history['lat'])
        csv += '#config - lon:{}\n'.format(history['lon'])
        csv += '#config - planting_date:{}\n'.format(history['planting_date'])
        csv += '#config - harvesting_date:{}\n'.format(history['harvesting_date'])
        csv += '#config - yield:{}\n'.format(history['yield'])
        mesh = pd.read_json(history['mesh'])
        mesh.index.name = 'DATE'
        csv += re.sub(' +', ',', mesh.reset_index().to_string(index=False))
        outcsv = os.path.join('dataset', key + '.csv')
        with open(outcsv, 'w') as f:
            f.write(csv)
