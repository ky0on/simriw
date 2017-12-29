#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" GY prediction using CNN  """

from __future__ import print_function
import argparse
import os
import glob
import joblib
import numpy as np
import pandas as pd
from slacker import Slacker
import matplotlib.pyplot as plt

from utils import save_and_slack_file, slack_file, log, xyline

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '15 Oct 2017'


def load_dataset(csvpath):
    log('Loading', csvpath)
    df = pd.read_csv(csvpath)
    return df


def fill_na_rows(df, target_nrows):
    """ Add nan rows to dataframe """

    nrows = target_nrows - df.shape[0]
    assert nrows >= 0, 'Maybe something wrong in longest calculation...'
    assert len(df.shape) == 2, 'Supports 2d dataframe only'

    ar = np.zeros((nrows, df.shape[1]))
    ar[:, :] = np.nan
    ar = pd.DataFrame(ar, columns=df.columns)
    df = df.append(ar, ignore_index=True)

    return df


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--std', action='store_true')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='the number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=32, help='mini-batch size')
    parser.add_argument('--noslack', action='store_false')
    parser.add_argument('--threshold', '-t', default=100, type=int, help='Eliminate data where y is smaller than this')
    args = parser.parse_args()
    # args.debug = True
    log(str(args))

    #init
    plt.style.use('ggplot')
    slack = Slacker(os.environ['SLACK_API_KYONAWS'])

    #load simdata
    csvpaths = glob.glob(os.path.join('simdata', '*.csv'))
    if args.debug:
        csvpaths = csvpaths[:2]
    simdata_all = joblib.Parallel(n_jobs=-1, verbose=1)(
        joblib.delayed(load_dataset)(csvpath) for csvpath in csvpaths)
    simdata = pd.concat(simdata_all)
    longest = simdata.groupby(['meshcode', 'year']).apply(len).max()
    # simdata.dropna(how='any', inplace=True)   # drop nan records

    #extract dataset
    x_types = ['DL', 'TMP', 'RAD']   # TODO: Tmax, co2
    xs, ys = [], []
    for meshcode in simdata['meshcode'].unique():
        for year in simdata.loc[simdata['meshcode'] == meshcode, :].year.unique():
            # log(year, meshcode)
            a_simdata = simdata.loc[(simdata['meshcode'] == meshcode) & (simdata['year'] == year), :]
            x = a_simdata[x_types]
            x = fill_na_rows(x, target_nrows=longest)
            y = a_simdata.GY.iloc[-1]
            xs.append(np.array(x))
            ys.append(y)
    xs = np.array(xs).astype(np.float32)
    ys = np.array(ys).astype(np.float32)
    log('xs.shape:', xs.shape)
    log('ys.shape:', ys.shape)

    #histogram of x and y
    fig, ax = plt.subplots(1, 1)
    ax.hist(ys)
    ax.set_xlabel('GY')
    save_and_slack_file(fig, 'output/hist_y_before.png', post=args.noslack)

    #cnn
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D
    # from keras.layers import Conv2D, MaxPooling2D
    from sklearn.preprocessing import MinMaxScaler

    #NaN -> 0
    xs = np.nan_to_num(xs)   # TODO: NaN -> 0. OK?

    #remove smaller GY
    xs = xs[ys > args.threshold]
    ys = ys[ys > args.threshold]

    #normalization
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    xs_scaled = x_scaler.fit_transform(xs.reshape(-1, xs.shape[2]))
    ys_scaled = y_scaler.fit_transform(ys.reshape(-1, 1))
    xs_scaled = xs_scaled.reshape(xs.shape)
    ys_scaled = ys_scaled.reshape(ys.shape)
    if args.std:
        x_train = xs_scaled.reshape(xs_scaled.shape[0], xs_scaled.shape[1], xs_scaled.shape[2], 1)
        y_train = ys_scaled.reshape(ys.shape[0], 1)
    else:
        x_train = xs.reshape(xs.shape[0], xs.shape[1], xs.shape[2], 1)
        y_train = ys.reshape(ys.shape[0], 1)
    log('x_train shape:', x_train.shape)
    log(x_train.shape[0], 'train samples')
    # log(x_test.shape[0], 'test samples')

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=x_train.shape[1:]))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=[keras.metrics.mae])

    #learn
    history = model.fit(x_train, y_train,
                        batch_size=args.batchsize,
                        epochs=args.epochs,
                        verbose=1,
                        # validation_data=(x_test, y_test),
                        )

    #history
    fig, axes = plt.subplots(1, 2)
    for i, t in enumerate(('loss', 'mean_absolute_error')):
        axes[i].plot(history.history[t], label=f'train ({t})')
        axes[i].legend(loc='upper right')
    save_and_slack_file(fig, 'output/history.png', post=args.noslack)

    #score
    score = model.evaluate(x_train, y_train, verbose=0)
    log('Test loss:', score[0])
    log('Test accuracy:', score[1])

    #plot prediction
    pred = model.predict(x_train)
    result = pd.DataFrame({
        'actual': y_train.flatten(),
        'predict': pred.flatten(),
    })
    fig, ax = plt.subplots(1, 1)
    result.plot.scatter(x='actual', y='predict', ax=ax)
    xyline(ax)
    save_and_slack_file(fig, 'output/predict.png', post=args.noslack)

    #post log
    slack_file('output/log.txt', post=args.noslack)
