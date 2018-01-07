#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import numpy as np
from tqdm import tqdm
# import pandas as pd
import matplotlib.pyplot as plt

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '31 Dec 2017'

if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', type=str,
                        default='./output/180107180327/model.h5',
                        help='Path to keras model')
    args = parser.parse_args()

    #init
    # plt.style.use('ggplot')
    basedir = os.path.dirname(args.path)
    outdir = os.path.dirname(args.path)

    #
    # Dense Layer Visualizations
    #

    from vis.utils import utils
    from keras import activations
    from keras.models import load_model
    from vis.visualization import visualize_activation
    from vis.input_modifiers import Jitter

    model = load_model(args.path)

    #Visualizing a specific output category
    layer_idx = utils.find_layer_idx(model, 'dense_2')   # last layer
    img1 = visualize_activation(model, layer_idx, filter_indices=0)
    img2 = visualize_activation(model, layer_idx, filter_indices=0, max_iter=500, verbose=True)
    img3 = visualize_activation(model, layer_idx, filter_indices=0, max_iter=500, input_modifiers=[Jitter(16)])
    #TODO(kyon): plot history
    stitched = utils.stitch_images([img1, img2, img3], cols=3)
    fig, ax = plt.subplots(1, 1)
    ax.axis('off')
    cax = ax.imshow(stitched[:, :, 0])
    fig.colorbar(cax)
    fig.savefig(os.path.join(outdir, 'dense_layer_vis.pdf'))

    #
    # Visualizing Conv filters
    # TODO: Try Jitter (already tried but could not understand the result...)
    #

    from vis.visualization import get_num_filters

    #Visualize all filters in this layer
    layer_idx = utils.find_layer_idx(model, 'conv2d_1')
    filters = np.arange(get_num_filters(model.layers[layer_idx]))

    #generate input image for each filter
    vis_images = []
    for idx in tqdm(filters):
        img = visualize_activation(model, layer_idx, filter_indices=idx)
        # img = utils.draw_text(img[:, :, 0], f'Filter {idx}', font='ipaexg', color=0)
        # img = np.expand_dims(img, axis=2)
        vis_images.append(img)

    #plot
    stitched = utils.stitch_images(vis_images, cols=32)
    fig, ax = plt.subplots(1, 1)
    ax.axis('off')
    cax = ax.imshow(stitched[:, :, 0])
    fig.colorbar(cax)
    fig.savefig(os.path.join(outdir, 'conv_layer_vis.pdf'))
