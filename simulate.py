#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Run SIMRIW simulation  """

from __future__ import print_function
import argparse
import os
import sys
import glob
import hjson
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import mkdir
from pysimriw import simriw

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '09 Oct 2017'


def simulate(csvpath):

    #check weather data
    # print(f'Loading {csvpath}')
    csv = pd.read_csv(csvpath, comment='#', index_col='date', parse_dates=['date'])
    meshcode = os.path.splitext(os.path.basename(csvpath))[0]
    if np.any(pd.isnull(csv.T2M)):
        print(f'np.nan in T2M. Skipped {csvpath}.')
        return None

    #plot raw data
    if args.plotraw:
        csv.plot(subplots=True)
        plt.savefig(os.path.join(outdir, 'raw_{}.jpg'.format(pref)))

    #assert
    years = np.unique(csv.index.year)
    assert(len(years) == 1)   # asuming cultivation finishes in a year
    year = years[0]

    #init
    start_date = str(year) + '-' + dataset[pref]['planting']

    #run simulation
    simulated = simriw.main(
        'Koshihikari', csvpath, True, start_date, 350,
        './cultivars.hjson', silent=True
    )
    simulated['d']['year'] = year
    simulated['d']['meshcode'] = meshcode

    for col in ['DL', 'DVI', 'DVR', 'DW', 'GY', 'LAI', 'PPM', 'PY', 'RAD', 'TAV', 'TMX']:
        simulated['d'][col] = pd.to_numeric(simulated['d'][col])

    return simulated['d']


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plotraw', action='store_true', help='Plot raw data')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    #init
    plt.style.use('ggplot')
    outdir = 'simdata'

    #load dataset settings
    with open('dataset.hjson', 'r') as f:
        dataset = hjson.load(f)

    #explore datasets
    for pref in dataset.keys():

        #debug (pref limitation)
        if args.debug and pref != 'kyushu':
            continue

        #explore year (to reduce memory consumption)
        for year in range(1980, 2017):

            #debug (year limitation)
            if args.debug and year > 1981:
                continue

            #init
            csvpaths = glob.glob(os.path.join('meshdata', pref, str(year), '*.csv'))
            csvpaths.sort()
            if len(csvpaths) == 0:
                continue

            #run simulation
            print(f'{pref} ({year})')
            result = joblib.Parallel(n_jobs=6, verbose=1)(
                joblib.delayed(simulate)(csvpath) for csvpath in csvpaths)
            print('Loaded meshdata')

            #convert result to dataframe
            result = pd.concat(result)
            result = result.round(3)

            #save
            outdir2 = os.path.join(outdir, pref, str(year))
            mkdir(outdir2)
            for meshcode in result.meshcode.unique():
                outcsv = os.path.join(outdir2, f'{meshcode}.csv')
                result[result.meshcode == meshcode].to_csv(outcsv, index=False)
                print(f'saved as {outcsv}')

    #plot GY in subplots
    # for loc in simulated_all['loc'].unique():
    #     df = simulated_all.loc[simulated_all['loc'] == loc, :]
    #     df.set_index('date', inplace=True)
    #     grouped = df.groupby('year')
    #     fig, axs = plt.subplots(figsize=(30, 20),
    #                             nrows=int(np.ceil(grouped.ngroups / 4)),
    #                             ncols=4)
    #
    #     targets = zip(grouped.groups.keys(), axs.flatten())
    #     for i, (year, ax) in enumerate(targets):
    #         grouped.get_group(year).GY.plot(ax=ax)
    #         ax.set_title(str(year))
    #         ax.set_ylim(0, 800)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(outdir, 'sim_' + loc + '.png'))
