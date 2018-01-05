#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '31 Dec 2017'

if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', type=str, default='./output/171231162244/model.h5',
                        help='Path to keras model')
    args = parser.parse_args()

    #init
    # plt.style.use('ggplot')
    outdir = os.path.dirname(args.path)

    #
    # Dense Layer Visualizations
    #

    from vis.utils import utils
    from keras import activations
    from keras.models import load_model

    model = load_model(args.path)
    layer_idx = utils.find_layer_idx(model, 'dense_2')   # last layer

    #Swap softmax with linear
    # model.layers[layer_idx].activation = activations.linear
    # model = utils.apply_modifications(model)

    #Visualizing a specific output category
    from vis.visualization import visualize_activation
    from vis.input_modifiers import Jitter
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
    #

    from vis.visualization import get_num_filters

    layer_name = 'conv2d_1'
    layer_idx = utils.find_layer_idx(model, layer_name)

    #Visualize all filters in this layer
    filters = np.arange(get_num_filters(model.layers[layer_idx]))

    #generate input image for each filter
    vis_images = []
    for idx in filters:
        print(f'Filter {idx}')
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

    #generate input image for each filter.
    vis_images = []
    for idx in filters:
        #jitter 5% relative to the image size.
        img = visualize_activation(model, layer_idx, filter_indices=idx,
                                   tv_weight=0.,
                                   input_modifiers=[Jitter(0.05)])
        vis_images.append(img)

    #plot
    stitched = utils.stitch_images(vis_images, cols=32)
    fig, ax = plt.subplots(1, 1)
    ax.axis('off')
    cax = ax.imshow(stitched[:, :, 0])
    fig.colorbar(cax)
    fig.savefig(os.path.join(outdir, 'conv_layer_vis_Jitter.pdf'))

    #Generate input image for each filter.
    new_vis_images = []
    for i, idx in enumerate(filters):
        #We will seed with optimized image this time.
        img = visualize_activation(model, layer_idx, filter_indices=idx,
                                   seed_input=vis_images[i],
                                   input_modifiers=[Jitter(0.05)])

        new_vis_images.append(img)

    #plot
    stitched = utils.stitch_images(vis_images, cols=32)
    fig, ax = plt.subplots(1, 1)
    ax.axis('off')
    cax = ax.imshow(stitched[:, :, 0])
    fig.colorbar(cax)
    fig.savefig(os.path.join(outdir, 'conv_layer_vis_Jitter_seedinput.pdf'))
