#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import os
import sys
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append((os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir)))
import simriw

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '08 Oct 2017'


def load_config(csvpath):
    ''' Load configs in csvfile '''
    config = {}
    f = open(csvpath)
    lines = f.readlines()
    f.close()
    for line in lines:
        if not line[0] == '#':
            break
        else:
            if line[:10] == '#config - ':
                _config = line.replace('#config - ', '').replace('\n', '')
                key, value = _config.split(':')
                try:
                    config[key] = float(value)
                except ValueError:
                    config[key] = str(value)

    return config


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', type=str, default='./dataset')
    parser.add_argument('--out', '-o', default='output', type=str)
    args = parser.parse_args()

    #init
    plt.style.use('ggplot')
    result = {}

    #main
    for csvpath in glob.glob(os.path.join(args.path, '*.csv')):

        #load params
        print('\n#', csvpath)
        name = os.path.basename(csvpath).replace('.csv', '')
        config = load_config(csvpath)

        #run simulation
        simulated = simriw.main(
            'Nipponbare', csvpath, True, config['planting_date'], 350,
            '../cultivars.hjson'
        )

        #fin
        simulated['d'][['DW', 'GY', 'PY']].plot()
        plt.savefig(os.path.join(args.out, 'simulated_{}.pdf').format(name))
        simulated['d'].to_csv(os.path.join(args.out, 'simulated_{}.csv').format(name))
        # print('\nsimulated["d"].tail():\n', simulated['d'].tail())

        #comparison
        print(config['harvesting_date'], str(simulated['d'].date.iloc[-1]))
        print(config['yield'], str(simulated['d'].GY.iloc[-1]))
        result[name] = {
            'yield': config['yield'],
            'GY': simulated['d'].GY.iloc[-1],
            'density': config['density'],
        }

    #fin
    result = pd.DataFrame(result).T
    result.plot.scatter(x='yield', y='GY', c='density')
    plt.savefig(os.path.join(args.out, 'yield_vs_GY.pdf'))
