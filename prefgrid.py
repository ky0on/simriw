#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import hjson
import subprocess


if __name__ == '__main__':

    #init
    apikey = os.environ['GOOGLE_MAP_API']

    #load
    with open('dataset.hjson', 'r') as f:
        dataset = hjson.load(f)

    #save map
    for pref, v in dataset.items():
        lat0 = v['lat0']
        lat1 = v['lat1']
        lon0 = v['lon0']
        lon1 = v['lon1']
        mapurl = \
            'https://maps.googleapis.com/maps/api/staticmap?' +\
            f'markers={lat0},{lon0}' +\
            f'&markers={lat0},{lon1}' +\
            f'&markers={lat1},{lon0}' +\
            f'&markers={lat1},{lon1}' +\
            f'&path={lat0},{lon0}|{lat0},{lon1}' +\
            f'&path={lat0},{lon0}|{lat1},{lon0}' +\
            f'&path={lat1},{lon0}|{lat1},{lon1}' +\
            f'&path={lat0},{lon1}|{lat1},{lon1}' +\
            f'&size=600x600' +\
            f'&key={apikey}'
        cmd = f'wget "{mapurl}" -O output/map/{pref}.png'
        subprocess.call(cmd, shell=True)
