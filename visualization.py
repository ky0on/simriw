
# coding: utf-8

# In[ ]:


from __future__ import print_function
import os
import numpy as np
from tqdm import tqdm
# import pandas as pd
import matplotlib.pyplot as plt

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '31 Dec 2017'

#easydict
import easydict
args = easydict.EasyDict({
        'path': './output/180107180327/model.h5',
})

#init
# plt.style.use('ggplot')
basedir = os.path.dirname(args.path)
outdir = os.path.dirname(args.path)


# In[ ]:


from vis.utils import utils
from keras import activations
from keras.models import load_model
from vis.visualization import visualize_activation
from vis.input_modifiers import Jitter

model = load_model(args.path)


# In[ ]:


#
# Denase Layer Visualization
#

result = {}

layer_idx = utils.find_layer_idx(model, 'dense_2')   # last layer
result['normal'] = visualize_activation(model, layer_idx, filter_indices=0)
result['iter_50'] = visualize_activation(model, layer_idx, filter_indices=0, max_iter=50, verbose=False)
result['iter_500'] = visualize_activation(model, layer_idx, filter_indices=0, max_iter=500, verbose=False)
# result['iter_1000'] = visualize_activation(model, layer_idx, filter_indices=0, max_iter=1000, verbose=False)
result['Jitter'] = visualize_activation(model, layer_idx, filter_indices=0, max_iter=50, input_modifiers=[Jitter(16)])
#TODO(kyon): plot history

fig, axes = plt.subplots(1, len(result.keys()))
for i, (title, img) in enumerate(result.items()):
    ax = axes[i]
    # ax.axis('off')
    ax.set_title(title)
    cax = ax.imshow(img[..., 0], interpolation='nearest', aspect='auto',
                   clim=(np.array(list(result.values())).min(), np.array(list(result.values())).max()))
fig.colorbar(cax)
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'dense_layer_vis.pdf'))


# In[ ]:


#
# Conv filter visualization
#

# TODO: Try Jitter (already tried but could not understand the result...)

from vis.visualization import get_num_filters

#Visualize all filters in this layer
layer_idx = utils.find_layer_idx(model, 'conv2d_1')
filters = np.arange(get_num_filters(model.layers[layer_idx]))

#generate input image for each filter
vis_images = []
for idx in tqdm(filters):
    img = visualize_activation(model, layer_idx, filter_indices=idx)
    vis_images.append(img)

#plot
fig, axes = plt.subplots(3, 12)
for i, (img, ax) in enumerate(zip(vis_images, axes.flatten())):
    ax.axis('off')
    ax.set_title(f'l{i}')
    cax = ax.imshow(img[..., 0], interpolation='nearest',
                    aspect='auto', clim=(np.array(vis_images).min(), np.array(vis_images).max()))
fig.colorbar(cax)
# fig.tight_layout()
fig.savefig(os.path.join(outdir, 'conv_layer_vis.pdf'))


# In[ ]:


#
# Attension Map
#

import matplotlib.cm as cm
from vis.visualization import visualize_saliency

x_train = np.load(os.path.join(basedir, 'x_train.npy'))
y_train = np.load(os.path.join(basedir, 'y_train.npy'))
fig, axes = plt.subplots(10, 4, figsize=(10, 20))
cnt = 0

for y in tqdm(np.arange(0, 1, .1)):

    x = x_train[(y_train.flatten() >= y-.03) & (y_train.flatten() < y+.03)][0]
    y = y_train[(y_train.flatten() >= y-.03) & (y_train.flatten() < y+.03)][0][0]
    ax = axes[cnt, 0]
    cax = ax.imshow(x[..., 0], cmap='jet', clim=(0, 1), interpolation='nearest', aspect='auto')
    ax.axis('off')
    ax.set_title(f'input (y={y:.2f})')
    fig.colorbar(cax, orientation='horizontal', ax=ax)

    for j, modifier in enumerate([None, 'guided', 'relu']):

        layer_idx = utils.find_layer_idx(model, 'dense_2')
        grads = visualize_saliency(model, layer_idx, filter_indices=0,
                                   seed_input=x, backprop_modifier=modifier)
        #TODO: grads is 3-channels image.
        ax = axes[cnt, j+1]
        ax.axis('off')
        cax = ax.imshow(grads, cmap='jet', clim=(0, 255), interpolation='nearest', aspect='auto')
        fig.colorbar(cax, orientation='horizontal', ax=ax)
        
        if cnt == 0:
            ax.set_title('vanilla' if modifier is None else modifier)
    
    cnt += 1

fig.tight_layout()
fig.savefig(os.path.join(outdir, 'saliency_vis.pdf'))

