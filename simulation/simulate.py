#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import argparse
import os
import sys
import hjson
import numpy as np
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
    simulated_all = []

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

            #run simulation
            simulated = simriw.main(
                'Koshihikari', dataset[key]['csv'], True, start_date, 350,
                '../cultivars.hjson'
            )
            simulated['d']['year'] = year
            simulated['d']['loc'] = key
            simulated_all.append(simulated['d'])

            #fin
            # simulated['d'].to_csv(os.path.join(outdir, 'sim_{}_{}.csv').format(key, year))

    #convert result to dataframe
    simulated_all = pd.concat(simulated_all)
    simulated_all.to_csv(os.path.join(outdir, 'sim.csv'), index=False)

    #plot GY in subplots
    for loc in simulated_all['loc'].unique():
        df = simulated_all.loc[simulated_all['loc'] == loc, :]
        df.set_index('date', inplace=True)
        grouped = df.groupby('year')
        fig, axs = plt.subplots(figsize=(30, 20),
                                nrows=int(np.ceil(grouped.ngroups / 4)),
                                ncols=4)

        targets = zip(grouped.groups.keys(), axs.flatten())
        for i, (year, ax) in enumerate(targets):
            grouped.get_group(year).GY.plot(ax=ax)
            ax.set_title(str(year))
            ax.set_ylim(0, 800)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'sim_' + loc + '.png'))
