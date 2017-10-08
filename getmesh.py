#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import hjson
import argparse

import AMD_Tools3 as AMD
# import os
# import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

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

    #init
    # plt.style.use('ggplot')
    history = {}

    #extract
    with open(args.path, 'r') as f:
        history_all = hjson.load(f)
    for key in history_all.keys():
        if has_key(['planting_date', 'harvesting_date', 'yield', 'gps'], history_all[key]):
            # print(key)
            history[key] = {}
            history[key]['planting_date'] = history_all[key]['planting_date']
            history[key]['harvesting_date'] = history_all[key]['harvesting_date']
            history[key]['yield'] = history_all[key]['yield']
            history[key]['lat'] = float(history_all[key]['gps'].split(',')[0])
            history[key]['lon'] = float(history_all[key]['gps'].split(',')[1])

    #get mesh
    for key in history.keys():
        element = 'TMP_mea'
        timedomain = [history[key]['planting_date'], history[key]['harvesting_date']]
        lalodomain = [history[key]['lat'], history[key]['lat'], history[key]['lon'], history[key]['lon']]
        Msh, tim, lat, lon = AMD.GetMetData(element, timedomain, lalodomain)
        Msh = Msh[:, 0, 0]     # dimension (from 3 to 1)
        print(Msh)
        # mesh = pd.DataFrame({'meshtemp': Msh})
        # mesh.set_index('date', inplace=True)

        #concat and save
        # df = pd.concat([ek.airtemp, mesh], axis=1)
        # df.to_csv(os.path.join(args.out, key + '.csv'))
