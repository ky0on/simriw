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
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append((os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir)))
import simriw

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '09 Oct 2017'


def simulate(csvpath):

    simulated_all = []

    #check weather data
    csv = pd.read_csv(csvpath, comment='#', index_col='DATE', parse_dates=['DATE'])
    name = csvpath.split('.')[0]
    if np.any(pd.isnull(csv.T2M)):
        print(f'np.nan in T2M. Skipped {csvpath}')
        return None

    #plot raw data
    if args.plotraw:
        csv.plot(subplots=True)
        plt.savefig(os.path.join(outdir, 'raw_{}.jpg'.format(pref)))

    #explore year
    # for year in range(1980, 2017):
    for year in range(1980, 1982):

        #init
        start_date = str(year) + '-' + dataset[pref]['planting']

        #run simulation
        simulated = simriw.main(
            'Koshihikari', csvpath, True, start_date, 350,
            '../cultivars.hjson', silent=True
        )
        simulated['d']['year'] = year
        simulated['d']['name'] = name
        simulated_all.append(simulated['d'])

        #fin
        # simulated['d'].to_csv(os.path.join(outdir, 'sim_{}_{}.csv').format(pref, year))

    return simulated_all


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plotraw', action='store_true', help='Plot raw data')
    args = parser.parse_args()

    #init
    plt.style.use('ggplot')
    outdir = 'output'
    result = []

    #load dataset settings
    with open('dataset.hjson', 'r') as f:
        dataset = hjson.load(f)

    #explore datasets
    for pref in dataset.keys():

        #filter by prefecture
        if pref != 'yamanashi':
            continue

        #init
        csvpaths = glob.glob(os.path.join('dataset', '{}*.csv'.format(pref)))
        csvpaths.sort()

        r = joblib.Parallel(n_jobs=-1, verbose=1)(joblib.delayed(simulate)(csvpath) for csvpath in csvpaths)
        result.append(pd.concat([pd.concat(_r) for _r in r if _r is not None]))

    #convert result to dataframe
    result = pd.concat(result)
    result.to_csv(os.path.join(outdir, 'sim.csv'), index=False)

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
