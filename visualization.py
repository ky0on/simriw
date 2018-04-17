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
from collections import defaultdict
from utils import save_and_slack_file

from keras import activations
from keras.models import load_model

from vis.utils import utils
from vis.visualization import visualize_activation, visualize_saliency
# from vis.input_modifiers import Jitter


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', type=str, default='./output/03-24-21-21-23')
    parser.add_argument('--sample', '-n', default=10, type=int, help='number of sampled images')
    args = parser.parse_args()

    #init
    outdir = args.path

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
            for col in range(x.shape[1]):
                saliency.append(pd.DataFrame(
                    {'col': col, 'image_num': i, 'dvi': dvi,
                     'saliency': grads[:, col]}))

        #plot
        df = pd.concat(saliency)
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, col in enumerate(df.col.unique()):
            ax = axes.flatten()[i]
            df[df.col == col].plot.scatter(x='dvi', y='saliency', ax=ax)
            ax.set_title(f'col={col}')

        #save
        outpng = os.path.join(outdir, f'saliency_{modifier_title}.png')
        fig.suptitle(modifier_title)
        fig.tight_layout()
        save_and_slack_file(fig, outpng)
