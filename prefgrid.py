#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import re
import hjson
import requests
import subprocess
import pandas as pd
from html.parser import HTMLParser


def to_dd(degminsec):
    deg, min, sec = re.sub('[^0-9]', ',', degminsec).split(',')[:3]
    deg, min, sec = int(deg), int(min), int(sec)

    if deg > 0:
        dd = (sec/3600) + (min/60) + deg
    else:
        dd = (sec/3600) - (min/60) + deg

    return dd


class Parser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.hrefs = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and 'href' in attrs and 'kendata' in attrs['href']:
            url = attrs['href']
            pref = url.split('/')[-1].split('_')[0]
            self.hrefs[pref] = url

    def handle_data(self, data):
        pass


if __name__ == '__main__':

    #init
    dataset = {}

    #get links for prefectures
    r = requests.get('http://www.gsi.go.jp/KOKUJYOHO/center.htm')
    parser = Parser()
    parser.feed(r.text)
    parser.close()

    #explore prefectures
    for pref, url in parser.hrefs.items():
        print(pref)
        table = pd.read_html(url, header=0, index_col=0, attrs={'border': 2})[0]
        table.columns = ['cho', 'east', 'west', 'south', 'north']
        table.index = ['lon', 'lat']
        table = table.applymap(to_dd)
        nw = {'lat': table.loc['lat', 'north'], 'lon': table.loc['lon', 'west']}
        se = {'lat': table.loc['lat', 'south'], 'lon': table.loc['lon', 'east']}
        lat0 = min(nw['lat'], se['lat'])
        lon0 = min(nw['lon'], se['lon'])
        lat1 = max(nw['lat'], se['lat'])
        lon1 = max(nw['lon'], se['lon'])
        dataset[pref] = {
            'planting': '05-01',
            'harvesting': '10-05',
            'lat0': lat0,
            'lon0': lon0,
            'lat1': lat1,
            'lon1': lon1,
            'csv': 'dataset/{}.csv'.format(pref),
        }

        #save map
        apikey = os.environ['GOOGLE_MAP_API']
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

    #save
    with open('dataset.hjson', 'w') as fp:
        hjson.dump(dataset, fp)
