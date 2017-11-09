#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import argparse
import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn import tree
from slacker import Slacker
import matplotlib.pyplot as plt

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '15 Oct 2017'


def load_dataset(csvpath):
    # print('Loading', csvpath)
    df = pd.read_csv(csvpath)
    return df


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    #init
    plt.style.use('ggplot')
    slack = Slacker(os.environ['SLACK_API_KYONAWS'])
    csvpaths = glob.glob(os.path.join('simdata', '*.csv'))
    if args.debug:
        csvpaths = csvpaths[:2]
    dfs = joblib.Parallel(n_jobs=-1, verbose=1)(
        joblib.delayed(load_dataset)(csvpath) for csvpath in csvpaths)
    df = pd.concat(dfs)
    shortest = df.groupby(['meshcode', 'year']).apply(len).min()
    x_types = ['DL', 'TMP', 'RAD']
    x, y = [], []

    #x_names
    x_names = []
    for t in x_types:
        for i in range(shortest):
            x_names.append(f'{t}_day{i:0>3}')

    #extract dataset
    xs, ys = [], []
    for meshcode in df['meshcode'].unique():
        for year in df.loc[df['meshcode'] == meshcode, :].year.unique():
            # print(year, meshcode)
            adf = df.loc[(df['meshcode'] == meshcode) & (df['year'] == year), :]
            adf = adf.iloc[:shortest, :]
            x = adf[x_types]
            x = x.values.T.flatten()
            y = adf.GY.iloc[-1]
            xs.append(x)
            ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    print('xs.shape:', xs.shape)
    print('ys.shape:', ys.shape)

    #run machine learning
    from sklearn.model_selection import ShuffleSplit
    ss = ShuffleSplit(n_splits=5, test_size=0.50, random_state=308)
    for fold, (train, test) in enumerate(ss.split(ys)):
        clf = tree.DecisionTreeRegressor(max_depth=10)
        clf = clf.fit(xs[train], ys[train])
        p = clf.predict(xs)
        is_test = np.zeros(ys.shape, dtype=bool)
        is_test[test] = True
        result = pd.DataFrame({'predicted': p, 'observed': ys, 'is_test': is_test})
        result.plot.scatter(x='observed', y='predicted', c='is_test')
        plt.savefig(f'output/simresult{fold}.pdf')
        # slack.files.upload('output/simresult.pdf', initial_comment='simresult.pdf', channels='#general')

        #plot tree
        import pydotplus
        from sklearn.externals.six import StringIO
        dot_data = StringIO()
        tree.export_graphviz(clf, out_file=dot_data,
                             feature_names=x_names,
                             filled=True, rounded=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf(f'output/tree{fold}.pdf')
        # slack.files.upload('output/tree.pdf', initial_comment='tree.pdf', channels='#general')
