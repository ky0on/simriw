#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" GY prediction using CNN  """

from __future__ import print_function
import argparse
import json
import glob
import numpy as np
import tensorflow as tf
import random as rn
import pandas as pd
from tqdm import tqdm
# from slacker import Slacker
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

#seed https://keras.io/getting-started/faq/
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

#cnn
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
# from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

from utils import save_and_slack_file, xyline
from utils import Logger

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '15 Oct 2017'


def load_dataset(csvpath, input):
    log('Loading', csvpath)
    df = pd.read_csv(csvpath)
    x = df[input]
    y = df.GY.iloc[-1]
    r = df[['DVI', 'LAI', 'ATHHT', 'ATHLT']]    # add to last if any other index is required (order is fixed in visualization.ipynb)
    return {'x': x, 'y': y, 'r': r}


def fill_na_rows(df, target_nrows):
    """ Add nan rows to dataframe """

    if df.shape[0] == target_nrows:
        return df

    zeros = [[0] * df.shape[1]] * (target_nrows - df.shape[0])
    zeros_df = pd.DataFrame(zeros, columns=df.columns)
    df = df.append(zeros_df)

    return df


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='the number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=32, help='mini-batch size')
    parser.add_argument('--noslack', action='store_false')
    parser.add_argument('--GYth', default=0, type=int, help='Eliminate data where y is smaller than')
    parser.add_argument('--DVIth', default=2.0, type=float, help='Eliminate data where final DVI is smaller than')
    parser.add_argument('--input', '-i', nargs='*', default=['DL', 'TAV', 'TMX', 'RAD', 'PPM'], type=str, help='Input variables (DL|TAV|TMX|RAD|PPM)')
    parser.add_argument('--model', '-m', default='3x3', type=str, help='Model structure (3x3|1x1)')
    parser.add_argument('--optimizer', '-o', default='rmsprop', type=str, help='Optimizer (rmsprop|sgd|adam|adagrad)')
    parser.add_argument('--noise', '-n', default=0, type=float, help='noise level')
    args = parser.parse_args()
    # args.debug = True

    #init
    plt.style.use('ggplot')
    outdir = os.path.join('output', pd.Timestamp.now().strftime('%m%d-%H%M%S'))
    if args.debug:
        outdir = os.path.join('/tmp', outdir)  # move to /tmp if debug
        args.epochs = 5                        # set epochs=5 if debug
    channel = '#xxx_simriw' if not args.debug else '#xxx_debug'
    os.mkdir(outdir)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    #save args
    with open(os.path.join(outdir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=2)

    #logger
    logpath = os.path.join(outdir, 'log')
    log = Logger(logpath)
    log(str(args))

    #load simdata
    log('inputs:', args.input)
    csvpaths = glob.glob(os.path.join('simdata', '*', '*', '*.csv'))
    csvpaths.sort()
    if args.debug:
        csvpaths = rn.sample(csvpaths, 50)
    simdata = joblib.Parallel(n_jobs=-1, verbose=1)(
        joblib.delayed(load_dataset)(csvpath, args.input) for csvpath in csvpaths)

    #extract dataset
    longest = np.max([d['x'].shape[0] for d in simdata])
    xs, ys, rs = [], [], []  # r: references (DVI, LAT etc.)
    for d in tqdm(simdata):
        xs.append(np.array(fill_na_rows(d['x'], longest)))
        rs.append(np.array(fill_na_rows(d['r'], longest)))
        ys.append(d['y'])
    xs = np.array(xs).astype(np.float32)
    ys = np.array(ys).astype(np.float32)
    rs = np.array(rs).astype(np.float32)
    log('xs.shape:', xs.shape)
    log('ys.shape:', ys.shape)
    log('rs.shape:', rs.shape)

    #histogram of x and y
    ax = axes[0, 0]
    ax.hist(ys)
    ax.set_xlabel('GY')
    ax.set_title('Before thresholding')

    #NaN -> 0
    xs = np.nan_to_num(xs)   # TODO: NaN -> 0. OK?

    #remove smaller GY
    xs = xs[ys >= args.GYth]
    rs = rs[ys >= args.GYth]
    ys = ys[ys >= args.GYth]
    ax = axes[0, 1]
    ax.hist(ys)
    ax.set_xlabel('GY')
    ax.set_title('After GY thresholding')
    log('xs.shape (After GYth):', xs.shape)

    #remove smaller DVI
    xs = xs[rs[:, :, 0].max(axis=1) >= args.DVIth]
    ys = ys[rs[:, :, 0].max(axis=1) >= args.DVIth]
    rs = rs[rs[:, :, 0].max(axis=1) >= args.DVIth]
    ax = axes[0, 2]
    ax.hist(ys)
    ax.set_xlabel('GY')
    ax.set_title('After DVI thresholding')
    log('xs.shape (After DVIth):', xs.shape)

    #normalization
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    xs_scaled = x_scaler.fit_transform(xs.reshape(-1, xs.shape[2]))   # normalize each column
    ys_scaled = y_scaler.fit_transform(ys.reshape(-1, 1))
    xs_scaled = xs_scaled.reshape(xs.shape)
    ys_scaled = ys_scaled.reshape(ys.shape)

    #split train/valid
    # x_train = xs.reshape(xs.shape[0], xs.shape[1], xs.shape[2], 1)
    # y_train = ys.reshape(ys.shape[0], 1)
    x_train = xs_scaled.reshape(xs_scaled.shape[0], xs_scaled.shape[1], xs_scaled.shape[2], 1)
    y_train = ys_scaled.reshape(ys.shape[0], 1)
    r_train = rs.reshape(rs.shape[0], rs.shape[1], rs.shape[2], 1)
    x_train, x_valid = train_test_split(x_train, test_size=.25, random_state=0)
    y_train, y_valid = train_test_split(y_train, test_size=.25, random_state=0)
    r_train, r_valid = train_test_split(r_train, test_size=.25, random_state=0)
    log('x_train shape:', x_train.shape)
    log('y_train shape:', y_train.shape)
    log(x_train.shape[0], 'train samples')
    log(x_valid.shape[0], 'valid samples')

    #add noise
    x_train += np.random.uniform(low=-args.noise, high=args.noise, size=x_train.shape)

    #model
    if args.model == '3x3':
        #3x3 conv2d
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                         input_shape=x_train.shape[1:], name='conv2d_1', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', name='conv2d_2', padding='same'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', name='dense_1'))
        # model.add(Dropout(0.5))
        model.add(Dense(1, name='dense_2'))
        model.summary(print_fn=log)
    elif args.model == '1x1':
        #1x1 conv2d
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(1, 1), activation='relu',
                         input_shape=x_train.shape[1:], name='conv2d_1'))
        model.add(Conv2D(32, (1, 4), activation='relu', name='conv2d_2'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        # model.add(Dense(128, activation='relu', name='dense_1'))
        # model.add(Dropout(0.5))
        model.add(Dense(1, name='dense_1'))
        model.summary(print_fn=log)
    else:
        raise Exception(f'Unknown model type: {args.model}')

    #optimizer
    if args.optimizer == 'rmsprop':
        optimizer = keras.optimizers.RMSprop()
    elif args.optimizer == 'sgd':
        optimizer = keras.optimizers.SGD()
    elif args.optimizer == 'adam':
        optimizer = keras.optimizers.Adam()
    elif args.optimizer == 'adagrad':
        optimizer = keras.optimizers.Adagrad()
    else:
        raise Exception(f'Unknown optimizer: {args.optimizer}')

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=optimizer,
                  metrics=[keras.metrics.mae])

    #callback
    csv_logger = CSVLogger(os.path.join(outdir, 'history.csv'))
    check_pointer = ModelCheckpoint(filepath=os.path.join(outdir, 'best.h5'), save_best_only=False)
    early_stopping = EarlyStopping(monitor='val_mean_absolute_error', patience=10, verbose=0)

    #learn
    history = model.fit(x_train, y_train,
                        batch_size=args.batchsize,
                        epochs=args.epochs,
                        verbose=1,
                        validation_data=(x_valid, y_valid),
                        callbacks=[early_stopping, check_pointer, csv_logger],
                        )

    #history
    for i, t in enumerate(('loss', 'mean_absolute_error')):
        axes[1, i].set_title(t)
        axes[1, i].plot(history.history[t], label=f'train')
        axes[1, i].plot(history.history[f'val_{t}'], label=f'valid')
        axes[1, i].legend(loc='upper right')

    #score
    score = model.evaluate(x_train, y_train, verbose=0)
    log('Train loss (mse):', score[0])
    log('Train acc  (mae):', score[1])
    score = model.evaluate(x_valid, y_valid, verbose=0)
    log('Valid loss (mse):', score[0])
    log('Valid acc  (mae):', score[1])

    #plot prediction
    for i, (x, y, title) in enumerate(((x_train, y_train, 'train'), (x_valid, y_valid, 'valid'))):
        pred = model.predict(x)
        for j, invert in enumerate((False, True)):
            result = pd.DataFrame({
                'actual': y_scaler.inverse_transform(y).flatten() if invert else y.flatten(),
                'predict': y_scaler.inverse_transform(pred).flatten() if invert else pred.flatten(),
            })
            ax = axes[2, 0+i*2+j]
            result.plot.scatter(x='actual', y='predict', ax=ax, alpha=.01, s=1)
            ax.set_title(title)
            xyline(ax)
            result.to_csv(f'{outdir}/result_{title}.csv')

    #slack
    fig.tight_layout()
    save_and_slack_file(fig, f'{outdir}/ml.png',
                        msg=f'{str(args)}\nTest mse={round(score[0], 3)}\nTest mae={round(score[1], 3)}',
                        post=args.noslack, channel=channel)
    # slack_file(logpath, post=args.noslack)

    #save
    np.savetxt(f'{outdir}/inputs.csv', args.input, delimiter=',', fmt='%s')
    np.save(f'{outdir}/x_train.npy', x_train)
    np.save(f'{outdir}/y_train.npy', y_train)
    np.save(f'{outdir}/r_train.npy', r_train)
    joblib.dump(x_scaler, f'{outdir}/x_scaler.dump')
    joblib.dump(y_scaler, f'{outdir}/y_scaler.dump')
