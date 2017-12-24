#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import argparse
import os
import glob
import joblib
import numpy as np
import pandas as pd
from slacker import Slacker
import matplotlib.pyplot as plt

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '15 Oct 2017'


def load_dataset(csvpath):
    print('Loading', csvpath)
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
    args = parser.parse_args()
    # args.debug = True

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
    x_types = ['DL', 'TMP', 'RAD']
    xs, ys = [], []
    for meshcode in simdata['meshcode'].unique():
        for year in simdata.loc[simdata['meshcode'] == meshcode, :].year.unique():
            # print(year, meshcode)
            a_simdata = simdata.loc[(simdata['meshcode'] == meshcode) & (simdata['year'] == year), :]
            x = a_simdata[x_types]
            x = fill_na_rows(x, target_nrows=longest)
            y = a_simdata.GY.iloc[-1]
            xs.append(np.array(x))
            ys.append(y)
            #TODO: eliminate lower GY?
    xs = np.array(xs).astype(np.float32)
    ys = np.array(ys).astype(np.float32)
    print('xs.shape:', xs.shape)
    print('ys.shape:', ys.shape)

    #histogram of x and y
    plt.cla()
    plt.hist(ys)
    plt.xlabel('GY')
    plt.savefig('output/hist_y.pdf')

    #cnn
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D
    # from keras.layers import Conv2D, MaxPooling2D

    batch_size = 32
    epochs = 100

    #NaN -> 0
    xs = np.nan_to_num(xs)   # TODO: NaN -> 0. OK?

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

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
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        # validation_data=(x_test, y_test),
                        )

    #history
    for t in ('loss', 'mean_absolute_error'):
        plt.cla()
        plt.plot(history.history[t], label=f'train ({t})')
        plt.legend()
        plt.savefig(f'output/history_{t}.png')

    #score
    score = model.evaluate(x_train, y_train, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #plot prediction
    pred = model.predict(x_train)
    result = pd.DataFrame({
        'actual': y_train.flatten(),
        'predict': pred.flatten(),
    })
    result.plot.scatter(x='actual', y='predict')
    plt.savefig('output/predict.pdf')
