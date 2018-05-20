#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# from sklearn.externals import joblib
from collections import defaultdict

from utils import save_and_slack_file

from keras.models import load_model
from vis.utils import utils
from vis.visualization import visualize_saliency
from sklearn.preprocessing import MinMaxScaler   # , StandardScaler


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', type=str, default='./output/05-07-00-08-30')
    parser.add_argument('--sample', '-n', default=5, type=int, help='number of sampled images')
    parser.add_argument('--plot_input_dvi', action='store_true')
    args = parser.parse_args()

    #init
    outdir = args.path
    inputs = np.loadtxt(os.path.join(args.path, 'inputs.csv'), dtype=str)

    #load model
    model = load_model(os.path.join(args.path, 'model.h5'))
    model.summary()

    #load data
    x_train = np.load(os.path.join(args.path, 'x_train.npy'))
    y_train = np.load(os.path.join(args.path, 'y_train.npy'))
    r_train = np.load(os.path.join(args.path, 'r_train.npy'))
    print('x_train.shape:', x_train.shape)

    #load scaler
    # x_scaler = joblib.load(os.path.join(args.path, 'x_scaler.dump'))
    # y_scaler = joblib.load(os.path.join(args.path, 'y_scaler.dump'))

    #init
    modifiers = {'positive': None, 'negate': 'negate', 'small_values': 'small_values'}
    # modifiers = {'positive': None}
    layer_idx = utils.find_layer_idx(model, 'dense_2')

    #explore modifires
    for modifier_title, modifier in modifiers.items():
        counts = {inp: np.zeros((11, 21), dtype=int) for inp in inputs}

        for cnt in tqdm(range(args.sample), desc=modifier_title):

            #random sampling
            i = np.random.randint(0, len(x_train))
            x = x_train[i]
            r = r_train[i]
            # y = y_train[i][0]
            dvi = r[:, 0, 0]    # fixed! (DVI is in 0th column)

            #debug (plot inputs and dvi)
            if args.plot_input_dvi:
                fig2, axes2 = plt.subplots(6, 1)
                for c in range(x.shape[1]):
                    axes2[c].plot(x[:, c, 0])
                axes2[5].plot(dvi)
                fig2.tight_layout()
                fig2.savefig(f'/tmp/{i:0>5}.png')

            #calculate saliency
            try:
                #TODO(kyon): why become slow after several iterations?
                grads = visualize_saliency(model, layer_idx, filter_indices=0, seed_input=x, grad_modifier=modifier)
            except:
                print('failed in vis')
                continue

            #save as dataframe
            for col in range(grads.shape[1]):
                inp = inputs[col]
                for g, d in zip(grads[:, col], dvi):
                    ig = int(g * 10)
                    id = int(d * 10)
                    counts[inp][ig, id] += 1

        #heatmap
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, inp in enumerate(inputs):

            #normalize
            count = pd.DataFrame(counts[inp])
            count.index = count.index / 10
            count.columns = count.columns / 10
            scaler = MinMaxScaler()
            count_normalized = scaler.fit_transform(count)

            #plot
            ax = axes.flatten()[i]
            sns.heatmap(count_normalized, ax=ax)
            ax.set_title(inp)
            # ax.set_xlim(0, 2.0)
            # ax.set_ylim(0, 1.0)

        #save
        outpng = os.path.join(outdir, f'saliency_{modifier_title}.png')
        fig.suptitle(modifier_title)
        fig.tight_layout()
        save_and_slack_file(fig, outpng)
