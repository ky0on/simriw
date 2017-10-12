#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import re
import hjson
import requests
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
        table = pd.read_html(url)[0]
        nw = [to_dd(table.iloc[2, 5]), to_dd(table.iloc[1, 4])]
        se = [to_dd(table.iloc[2, 3]), to_dd(table.iloc[1, 2])]
        dataset[pref] = {
            'planting': '05-01',
            'harvesting': '10-05',
            'lat0': min(nw[0], se[0]),
            'lon0': min(nw[1], se[1]),
            'lat1': max(nw[0], se[0]),
            'lon1': max(nw[1], se[1]),
            'csv': 'dataset/{}.csv'.format(pref),
        }

    #save
    with open('dataset.hjson', 'w') as fp:
        hjson.dump(dataset, fp)
