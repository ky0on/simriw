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
from collections import defaultdict

from utils import save_and_slack_file

from keras.models import load_model
from vis.utils import utils
from vis.visualization import visualize_saliency


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', type=str, default='./output/03-24-21-21-23')
    parser.add_argument('--sample', '-n', default=10, type=int, help='number of sampled images')
    args = parser.parse_args()

    #init
    outdir = args.path
    inputs = ['DL', 'TAV', 'TMX', 'RAD', 'PPM']   # TODO(kyon): load from log or something

    #load model
    model = load_model(os.path.join(args.path, 'model.h5'))
    model.summary()

    #load data
    x_train = np.load(os.path.join(args.path, 'x_train.npy'))
    y_train = np.load(os.path.join(args.path, 'y_train.npy'))
    r_train = np.load(os.path.join(args.path, 'r_train.npy'))

    #init
    modifiers = {'positive': None, 'negate': 'negate', 'small_values': 'small_values'}
    # modifiers = {'positive': None}
    layer_idx = utils.find_layer_idx(model, 'dense_2')

    #explore modifires
    for modifier_title, modifier in modifiers.items():
        saliency = []

        for cnt in tqdm(range(args.sample)):

            #random sampling
            i = np.random.randint(0, len(x_train))
            x = x_train[i]
            r = r_train[i]
            y = y_train[i][0]
            dvi = r[:, 0, 0]    # fixed! (DVI is in 0th column)

            #debug (plot inputs and dvi)
            fig2, axes2 = plt.subplots(6, 1)
            for c in range(x.shape[1]):
                axes2[c].plot(x[:, c, 0])
            axes2[5].plot(dvi)
            fig2.tight_layout()
            fig2.savefig(f'/tmp/{i:0>5}.png')

            #calculate saliency
            try:
                grads = visualize_saliency(model, layer_idx, filter_indices=0, seed_input=x, grad_modifier=modifier)
            except:
                print('failed in vis')
                continue

            #save as dataframe
            #TODO(kyon): plot _saliency with dvi
            _saliency = pd.DataFrame(grads, columns=inputs)
            _saliency['dvi'] = dvi
            _saliency['image_num'] = i
            _saliency['count'] = 1
            # print(_saliency.head())
            # print(_saliency.tail())
            saliency.append(_saliency)

        #heatmap
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        df = pd.concat(saliency).round(1)
        for i, col in enumerate(inputs):
            ax = axes.flatten()[i]
            counts = df.groupby(['dvi', col]).count()['count']
            counts = counts.unstack().T
            counts.index = np.round(counts.index, 1)
            counts.columns = np.round(counts.columns, 1)
            counts = counts.fillna(0)
            # counts = counts.astype(int)
            counts_normalized = (counts-counts.min())/(counts.max()-counts.min())
            sns.heatmap(counts_normalized, ax=ax)
            ax.set_title(col)
            # ax.set_xlim(0, 2.0)
            # ax.set_ylim(0, 1.0)

        #save
        outpng = os.path.join(outdir, f'saliency_{modifier_title}.png')
        fig.suptitle(modifier_title)
        fig.tight_layout()
        save_and_slack_file(fig, outpng)
