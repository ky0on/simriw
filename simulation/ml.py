#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import argparse
# import os
# import glob
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '09 Oct 2017'

if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    #init
    plt.style.use('ggplot')
    df = pd.read_csv('output/sim.csv')
    shortest = df.groupby(['loc', 'year']).apply(len).min()
    x_types = ['DL', 'TMP', 'RAD']
    x, y = [], []

    #x_names
    x_names = []
    for t in x_types:
        for i in range(shortest):
            x_names.append(f'{t}_day{i:0>3}')

    #extract dataset
    xs, ys = [], []
    for loc in df['loc'].unique():
        for year in df.loc[df['loc'] == loc, :].year.unique():
            adf = df.loc[(df['loc'] == loc) & (df['year'] == year), :] 
            adf = adf.iloc[:shortest, :]
            x = adf[x_types]
            x = x.values.T.flatten()
            y = adf.GY.iloc[-1]
            xs.append(x)
            ys.append(y)

    #run machine learning
    xs = np.array(xs)
    ys = np.array(ys)
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(xs, ys)
    p = clf.predict(xs)
    result = pd.DataFrame({'predicted': p, 'observed': ys})
    result.plot.scatter(x='observed', y='predicted')
    plt.savefig('output/result.pdf')
