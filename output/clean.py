#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import subprocess

""" Move folders with single file to /tmp """

for d in glob.glob('./*'):

    if not os.path.isdir(d):
        continue

    files = glob.glob(os.path.join(d, '*'))
    if len(files) < 2:
        print(f'files in {d}: {files}')
        cmd = f'mv {d} /tmp/{d}'
        subprocess.check_output(cmd, shell=True)
