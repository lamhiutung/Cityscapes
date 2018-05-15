#!/usr/bin/env python
# coding=utf-8

import os
import re

path = './test'
files = os.listdir(path)

counter = 0
for file in files:
    try:
        newname = re.findall(r'(.+)_gt.+\.json$', file)[0]
        newname += '.json'
    except Exception as e:
        print(e)
        continue
    print(newname)
    src_file = os.path.join(path, file)
    dst_file = os.path.join(path, newname)
    os.rename(src_file, dst_file)
    print("Successfully rename {} -> {}".format(src_file, dst_file))
    counter += 1

print(counter)