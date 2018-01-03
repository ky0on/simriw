#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
# import glob
# import numpy as np
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
    stitched = utils.stitch_images([img1, img2, img3], cols=3)
    plt.imshow(stitched[:, :, 0])
    plt.colorbar()
    plt.savefig(os.path.join(outdir, 'dense_layer_vis.pdf'))

    #
    # Visualizing Conv filters
    #

    # # In a CNN, each Conv layer has several learned *template matching* filters that maximize their output when a similar
    # # template pattern is found in the input image. First Conv layer is easy to interpret; simply visualize the weights as an image. To see what the Conv layer is doing, a simple option is to apply the filter over raw input pixels.
    # # Subsequent Conv filters operate over the outputs of previous Conv filters (which indicate the presence or absence
    # # of some templates), making them hard to interpret.
    # #
    # # One way of interpreting them is to generate an input image that maximizes the filter output. This allows us to generate an input that activates the filter.
    # #
    # # Lets start by visualizing the second conv layer of vggnet (named as 'block1_conv2'). Here is the VGG16 model for reference.
    #
    # # In[13]:
    #
    #
    # model.summary()
    #
    #
    # # In[7]:
    #
    #
    # from vis.visualization import get_num_filters
    #
    # # The name of the layer we want to visualize
    # # You can see this in the model definition.
    # layer_name = 'block1_conv2'
    # layer_idx = utils.find_layer_idx(model, layer_name)
    #
    # # Visualize all filters in this layer.
    # filters = np.arange(get_num_filters(model.layers[layer_idx]))
    #
    # # Generate input image for each filter.
    # vis_images = []
    # for idx in filters:
    #     img = visualize_activation(model, layer_idx, filter_indices=idx)
    #
    #     # Utility to overlay text on image.
    #     img = utils.draw_text(img, 'Filter {}'.format(idx), font='ipaexg')
    #     print(f'Filter {idx}')
    #     vis_images.append(img)
    #
    # # Generate stitched image palette with 8 cols.
    # stitched = utils.stitch_images(vis_images, cols=8)
    # plt.axis('off')
    # plt.imshow(stitched)
    # plt.title(layer_name)
    # plt.show()
    #
    #
    # # They mostly seem to match for specific color and directional patterns. Lets try a bunch of other layers.
    # # We will randomly visualize 10 filters within various layers.
    #
    # # In[8]:
    #
    #
    # selected_indices = []
    # for layer_name in ['block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']:
    #     layer_idx = utils.find_layer_idx(model, layer_name)
    #
    #     # Visualize all filters in this layer.
    #     filters = np.random.permutation(get_num_filters(model.layers[layer_idx]))[:3]
    #     selected_indices.append(filters)
    #
    #     # Generate input image for each filter.
    #     vis_images = []
    #     for idx in filters:
    #         img = visualize_activation(model, layer_idx, filter_indices=idx)
    #
    #         # Utility to overlay text on image.
    #         img = utils.draw_text(img, 'Filter {}'.format(idx), font='ipaexg')
    #         print(f'{layer_name} Filter {idx}')
    #         vis_images.append(img)
    #
    #     # Generate stitched image palette with 5 cols so we get 2 rows.
    #     stitched = utils.stitch_images(vis_images, cols=5)
    #     plt.figure()
    #     plt.axis('off')
    #     plt.imshow(stitched)
    #     plt.show()
    #
    #
    # # We can see how filters evolved to look for simple -> complex abstract patterns.
    # #
    # # We also notice that some of the filters in `block5_conv3` (the last one) failed to converge.  This is usually because regularization losses (total variation and LP norm) are overtaking activation maximization loss (set verbose=True to observe). There are a couple of options to make this work better,
    # #
    # # - Different regularization weights.
    # # - Increase number of iterations.
    # # - Add `Jitter` input_modifier.
    # # - Try with 0 regularization weights, generate a converged image and use that as `seed_input` with regularization enabled.
    #
    # # I will show a subset of these ideas here. Lets start by adidng Jitter and disabling total variation.
    #
    # # In[16]:
    #
    #
    # layer_idx = utils.find_layer_idx(model, 'block5_conv3')
    #
    # # We need to select the same random filters in order to compare the results.
    # filters = selected_indices[-1]
    # selected_indices.append(filters)
    #
    # # Generate input image for each filter.
    # vis_images = []
    # for idx in filters:
    #     # We will jitter 5% relative to the image size.
    #     img = visualize_activation(model, layer_idx, filter_indices=idx,
    #                             tv_weight=0.,
    #                             input_modifiers=[Jitter(0.05)])
    #
    #     # Utility to overlay text on image.
    #     img = utils.draw_text(img, 'Filter {}'.format(idx))
    #     vis_images.append(img)
    #
    # # Generate stitched image palette with 5 cols so we get 2 rows.
    # stitched = utils.stitch_images(vis_images, cols=5)
    # plt.figure()
    # plt.axis('off')
    # plt.imshow(stitched)
    # plt.show()
    #
    #
    # # We can see how previously unconverged filters show something this
    # # time. Lets take a specific output from here and use it as a
    # # `seed_input` with total_variation enabled this time.
    #
    # # In[17]:
    #
    #
    # # Generate input image for each filter.
    # new_vis_images = []
    # for i, idx in enumerate(filters):
    #     # We will seed with optimized image this time.
    #     img = visualize_activation(model, layer_idx, filter_indices=idx,
    #                             seed_input=vis_images[i],
    #                             input_modifiers=[Jitter(0.05)])
    #
    #     # Utility to overlay text on image.
    #     img = utils.draw_text(img, 'Filter {}'.format(idx))
    #     new_vis_images.append(img)
    #
    # # Generate stitched image palette with 5 cols so we get 2 rows.
    # stitched = utils.stitch_images(new_vis_images, cols=5)
    # plt.figure()
    # plt.axis('off')
    # plt.imshow(stitched)
    # plt.show()
    #
    #
    # # And that, folks, is how we roll :)
    # # This trick works pretty well to get those stubborn filters to converge.
    #
    # # # ## Other fun stuff
    # #
    # # # The API to `visualize_activation` accepts `filter_indices`. This is generally meant for *multi label* classifiers, but nothing prevents us from having some fun.
    # # #
    # # # By setting `filter_indices`, to multiple output categories, we can generate an input that the network thinks is both those categories. Maybe we can generate a cool looking crab fish. I will leave this as an exersice to the reader. You mgith have to experiment with regularization weights a lot.
    # # #
    # # # Ideally, we can use a GAN trained on imagenet and use the discriminator loss as a regularizer. This is easily done using `visualize_activations_with_losses` API. If you ever do this, please consider submitting a PR :)
    # #
    # # # ## Visualizations without swapping softmax
    # #
    # # # As alluded at the beginning of the tutorial, we want to compare and see what happens if we didnt swap out softmax for linear activation.
    # # #
    # # # Lets try the `ouzel` visualization again.
    # #
    # # # In[21]:
    # #
    # #
    # # layer_idx = utils.find_layer_idx(model, 'predictions')
    # #
    # # # Swap linear back with softmax
    # # model.layers[layer_idx].activation = activations.softmax
    # # model = utils.apply_modifications(model)
    # #
    # # img = visualize_activation(model, layer_idx, filter_indices=20, input_modifiers=[Jitter(16)])
    # # plt.rcParams['figure.figsize'] = (18, 6)
    # # plt.imshow(img)
    # #
    # #
    # # # It does not work! The reason is that maximizing an output node can be done by minimizing other outputs. Softmax is weird that way. It is the only activation that depends on other node output(s) in the layer.
