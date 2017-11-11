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

    #load simdata
    csvpaths = glob.glob(os.path.join('simdata', '*.csv'))[:3]
    if args.debug:
        csvpaths = csvpaths[:2]
    simdata_all = joblib.Parallel(n_jobs=-1, verbose=1)(
        joblib.delayed(load_dataset)(csvpath) for csvpath in csvpaths)
    simdata = pd.concat(simdata_all)
    simdata.dropna(how='any', inplace=True)   # drop nan records

    #run machine learning
    from sklearn.model_selection import ShuffleSplit
    ss = ShuffleSplit(n_splits=5, test_size=0.30, random_state=308)
    for fold, (train, test) in enumerate(ss.split(simdata)):

        #init
        xd = ['DL', 'TMP', 'RAD', 'DVI']
        yd = ['DVR']

        #train
        clf = tree.DecisionTreeRegressor(max_depth=10)
        clf = clf.fit(simdata.iloc[train][xd],
                      simdata.iloc[train][yd])

        #test
        pr = clf.predict(simdata[xd])
        is_test = np.zeros(simdata.shape[0], dtype=int)
        is_test[test] = 1
        result = pd.DataFrame({'predicted': pr, 'observed': simdata[yd].values[:, 0], 'is_test': is_test})

        #plot prediction
        result.plot.scatter(x='observed', y='predicted', c='is_test')
        plt.savefig(f'output/simresult{fold}.png')
        # slack.files.upload('output/simresult.png', initial_comment='simresult.png', channels='#general')

        #plot tree
        import pydotplus
        from sklearn.externals.six import StringIO
        dot_data = StringIO()
        tree.export_graphviz(clf,
                             out_file=dot_data,
                             feature_names=xd,
                             filled=True,
                             rounded=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(f'output/tree{fold}.png')
        # slack.files.upload('output/tree.png', initial_comment='tree.png', channels='#general')
