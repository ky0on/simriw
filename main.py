#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import os
import re
import hjson
import argparse
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '08 Oct 2017'

if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('hoge')
    parser.add_argument('path', nargs='?', type=str, default='./history_with_yield.hjson')
    # parser.add_argument('path', nargs='+', type=str, help='one or more arguments')
    # parser.add_argument('path', nargs='*', type=str, help='one or more args. uses default if empty.')
    # parser.add_argument('--model', '-m', default='conv3layer', type=str, help='model type')
    # parser.add_argument('--foo', '-f', action='store_true')
    args = parser.parse_args()

    #init
    # plt.style.use('ggplot')
    tmpout = './tmp'

    #load dataset
    with open(args.path, 'r') as f:
        history = hjson.load(f)

    #main
    for key in history.keys():

        #json -> csv
        csv = '#config - lat:{}\n'.format(history[key]['lat'])
        csv += '#config - lon:{}\n'.format(history[key]['lon'])
        # csv += '#config - alt:{}\n'.format(history[key]['alt'])
        mesh = pd.read_json(history[key]['mesh'])
        mesh.index.name = 'DATE'
        csv += re.sub(' +', ',', mesh.reset_index().to_string(index=False))
        outcsv = os.path.join(tmpout, key + '.csv')
        with open(outcsv, 'w') as f:
            f.write(csv)

        #run simulation
