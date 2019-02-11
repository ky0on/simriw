#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import json
import sys
sys.path.append('keras-vis')
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
# from sklearn.externals import joblib

from utils import save_and_slack_file

from keras.models import load_model
from vis.utils import utils
from vis.visualization import visualize_saliency
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', type=str, default='./output/0624-144155/21.h5')
    parser.add_argument('--sample', '-n', default=100, type=int, help='number of sampled images')
    parser.add_argument('--ATHHT', default=1.0, type=float, help='Eliminate data where the smallest ATHHT is smaller than')
    parser.add_argument('--ATHLT', default=1.0, type=float, help='Eliminate data where the smallest ATHLT is smaller than')
    args = parser.parse_args()

    #init
    np.random.seed(308)
    srcdir = os.path.dirname(args.path)
    outdir = os.path.dirname(args.path)
    with open(os.path.join(srcdir, 'args.json'), 'r') as fp:
        ml_args = json.load(fp)
    inputs = ml_args['input']

    #load model
    model = load_model(args.path)
    # model.summary()

    #load data
    x_train = np.load(os.path.join(srcdir, 'x_train.npy'))
    y_train = np.load(os.path.join(srcdir, 'y_train.npy'))
    r_train = np.load(os.path.join(srcdir, 'r_train.npy'))
    print('x_train.shape:', x_train.shape)

    #filter
    DVI = r_train[:, :, 0, 0]     # fixed column
    # LAI = r[:, 1, 0]            # fixed column
    ATHHT = r_train[:, :, 2, 0]   # fixed column
    ATHLT = r_train[:, :, 3, 0]   # fixed column
    ATHHT[ATHHT == 0.0] = np.nan
    ATHLT[ATHLT == 0.0] = np.nan
    idxs = np.where((DVI.max(axis=1) >= 2.0) & (np.nanmin(ATHHT, axis=1) <= args.ATHHT) & (np.nanmin(ATHLT, axis=1) <= args.ATHLT))[0]
    print(f'{len(idxs)} of {DVI.shape[0]} were extracted.')
    if len(idxs) < args.sample:
        raise Exception('No enough data.')

    #shuffle
    idxs = np.random.permutation(idxs)[:args.sample]

    #load scaler
    # x_scaler = joblib.load(os.path.join(srcdir, 'x_scaler.dump'))
    # y_scaler = joblib.load(os.path.join(srcdir, 'y_scaler.dump'))

    #init
    # modifiers = {'positive': None, 'negate': 'negate', 'small_values': 'small_values'}
    # modifiers = {'positive': None, 'negate': 'negate'}
    modifiers = {'positive': None}
    layer_idx = utils.find_layer_idx(model, 'dense_2')

    #explore modifires
    for modifier_title, modifier in modifiers.items():
        counts = {inp: np.zeros((11, 21), dtype=int) for inp in inputs}

        for idx in tqdm(idxs, desc=modifier_title):

            #random sampling
            x = x_train[idx]
            r = r_train[idx]
            # y = y_train[idx][0]
            dvi = r[:, 0, 0]

            #calculate saliency
            #TODO(kyon): why become slow after several iterations?
            grads = visualize_saliency(model, layer_idx, filter_indices=0, seed_input=x, grad_modifier=modifier)
            # print('grads.shape:', grads.shape)

            #save as dataframe
            for col in range(grads.shape[1]):
                inp = inputs[col]     # DL, TMX etc.
                for g, d in zip(grads[:, col], dvi):
                    if d > 0.0:   # ignore filled rows by ml.py
                        ig = int(g * 10)   # ig: [0, 1] (saliency -> row index)
                        id = int(d * 10)   # id: [0, 2] (dvi -> column index)
                        counts[inp][ig, id] += 1     # count up cell with specific saliency (row) and DVI (column)

        #heatmap
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, inp in enumerate(inputs):

            #normalize
            count = pd.DataFrame(counts[inp]).astype(float)
            count.index = count.index / 10
            count.columns = count.columns / 10
            scaler = MinMaxScaler()
            count_normalized = pd.DataFrame(scaler.fit_transform(count))
            count_normalized.index /= 10
            count_normalized.columns /= 10
            count_normalized.index.name = 'Saliency'
            count_normalized.columns.name = 'DVI'

            #title
            if inp == 'TAV':
                title = '$T_{mean}$'
            elif inp == 'TMX':
                title = '$T_{max}$'
            elif inp == 'RAD':
                title = '$S_s$'
            elif inp == 'PPM':
                title = '$P$'
            elif inp == 'DL':
                title = '$L$'
            else:
                title = inp

            #plot
            ax = axes.flatten()[i]
            sns.heatmap(count_normalized, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('$DVI$')
            ax.set_ylabel('Saliency')
            ax.invert_yaxis()

        #save
        out = os.path.join(
            outdir,
            f'saliency_{modifier_title}_ATHHT={args.ATHHT}_ATHLT={args.ATHLT}.tiff')
        # fig.suptitle(modifier_title)
        fig.suptitle('')
        fig.tight_layout()
        save_and_slack_file(fig, out, dpi=300, msg=str(args))
