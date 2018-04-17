#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt

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
    inputs = ['DL', 'TAV', 'TMX', 'RAD', 'PPM']

    #load model
    model = load_model(os.path.join(args.path, 'model.h5'))
    model.summary()

    #load data
    x_train = np.load(os.path.join(args.path, 'x_train.npy'))
    y_train = np.load(os.path.join(args.path, 'y_train.npy'))
    r_train = np.load(os.path.join(args.path, 'r_train.npy'))

    #init
    modifiers = {'positive': None, 'negate': 'negate', 'small_values': 'small_values'}
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
            # print(_saliency.head())
            # print(_saliency.tail())
            saliency.append(_saliency)

        #plot
        #TODO(kyon): scatter -> heat map
        df = pd.concat(saliency)
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, col in enumerate(inputs):
            ax = axes.flatten()[i]
            df.plot.scatter(x='dvi', y=col, ax=ax)
            ax.set_title(col)

        #save
        outpng = os.path.join(outdir, f'saliency_{modifier_title}.png')
        fig.suptitle(modifier_title)
        fig.tight_layout()
        save_and_slack_file(fig, outpng)
