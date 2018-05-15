#!/usr/bin/env python
# coding=utf-8

import os
import re
import shutil


SOURCE_DIR = './gtFine_trainvaltest/gtFine'
subpaths = os.listdir(SOURCE_DIR)
OUT_DIR = './gtFine/'
pattern = r'.+.json$'

# for subpath in ['train', 'test', 'val']
for subpath in subpaths:
    # join the './gtFine' for walk dir
    tmp_subpath = os.path.join(SOURCE_DIR, subpath)
    
    # create the output dir
    if not os.path.exists(os.path.join(OUT_DIR, subpath)):
        os.mkdir(os.path.join(OUT_DIR, subpath))

    temp_dst = os.path.join(OUT_DIR, subpath)

    for dirpath, dirnames, filenames in os.walk(tmp_subpath):
        for filename in filenames:
            if re.match(pattern, filename):
                srcimage = os.path.join(dirpath, filename)
                shutil.copy(srcimage, temp_dst)
                print("Successfully copy {} -> {}".format(srcimage, temp_dst))
