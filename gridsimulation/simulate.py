#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

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

sys.path.append((os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir)))
import simriw

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '09 Oct 2017'


def simulate(csvpath):

    #check weather data
    # print(f'Loading {csvpath}')
    csv = pd.read_csv(csvpath, comment='#', index_col='DATE', parse_dates=['DATE'])
    name = os.path.splitext(csvpath)[0]
    if np.any(pd.isnull(csv.T2M)):
        print(f'np.nan in T2M. Skipped {csvpath}')
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
        '../cultivars.hjson', silent=True
    )
    simulated['d']['year'] = year
    simulated['d']['name'] = name

    return simulated['d']


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plotraw', action='store_true', help='Plot raw data')
    args = parser.parse_args()

    #init
    plt.style.use('ggplot')
    outdir = 'simulated'

    #load dataset settings
    with open('dataset.hjson', 'r') as f:
        dataset = hjson.load(f)

    #explore datasets
    for pref in dataset.keys():

        #explore year (to reduce memory consumption)
        for year in range(1980, 2017):

            #init
            csvpaths = glob.glob(os.path.join('meshdata', pref, str(year), '*.csv'))
            csvpaths.sort()
            if len(csvpaths) == 0:
                continue

            result = joblib.Parallel(n_jobs=-1, verbose=1)(joblib.delayed(simulate)(csvpath) for csvpath in csvpaths)

            #convert result to dataframe
            result = pd.concat(result)

            #save
            outcsv = os.path.join(outdir, f'{pref}_{year}.csv')
            result.to_csv(outcsv, index=False)
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
