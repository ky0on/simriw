#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import hjson
import argparse
# import os
# import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '08 Oct 2017'


def has_key(keys, dict):
    for key in keys:
        if key not in dict.keys():
            return False
    return True


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', type=str, default='./history.hjson')
    args = parser.parse_args()

    #init
    # plt.style.use('ggplot')
    history = {}

    #
    with open(args.path, 'r') as f:
        history_all = hjson.load(f)
    for key in history_all.keys():
        if has_key(['planting_date', 'harvesting_date', 'yield', 'gps'], history_all[key]):
            print(key)
            history[key] = {}
            history[key]['yield'] = history_all[key]['yield']
            history[key]['planting_date'] = history_all[key]['planting_date']
            history[key]['harvesting_date'] = history_all[key]['harvesting_date']
