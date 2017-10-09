#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import argparse
import os
import sys
import glob
import hjson
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append((os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir)))
import simriw

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '09 Oct 2017'

if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plotraw', action='store_true', help='Plot raw data')
    args = parser.parse_args()

    #init
    plt.style.use('ggplot')
    outdir = 'output'

    #load dataset settings
    with open('dataset.hjson', 'r') as f:
        dataset = hjson.load(f)

    #explore datasets
    for key in dataset.keys():

        #plot raw data
        if args.plotraw:
            csv = pd.read_csv(dataset[key]['csv'], comment='#', index_col='DATE', parse_dates=['DATE'])
            csv.plot(subplots=True)
            plt.savefig(os.path.join(outdir, 'raw_{}.jpg'.format(key)))

        #explore year
        for year in range(1980, 2016):

            #init
            start_date = str(year) + '-' + dataset[key]['planting']
            print(start_date)

            #run simulation
            simulated = simriw.main(
                'Nipponbare', dataset[key]['csv'], True, start_date, 350,
                '../cultivars.hjson'
            )

            #fin
            # simulated['d'][['DW', 'GY', 'PY']].plot()
            # plt.savefig(os.path.join(outdir, 'sim_{}_{}.pdf').format(key, year))
            # simulated['d'].to_csv(os.path.join(outdir, 'sim_{}.csv').format(key))
            # print('\nsimulated["d"].tail():\n', simulated['d'].tail())
