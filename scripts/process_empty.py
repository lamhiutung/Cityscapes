#!/usr/bin/env python
# coding=utf-8

import os
import re

mask_dir = '../data/gtFine/'
source_dir = '../data/leftImg8bit/'

# keys : train/val
subdir_list = os.listdir(mask_dir)
sub_dirs = [mask_dir + subdir for subdir in subdir_list]

payloads = {}
for item in sub_dirs:
    temp_key = re.findall(r"/(\w+$)", item)[0]
    payloads[temp_key] = [os.path.join(item, mask) for mask in os.listdir(item)]

empty = {}
# enumerate train/val
for key in subdir_list:
    empty_path = [item for item in payloads[key] if not os.listdir(item)]
    empty_id = [re.findall(r"/(\w+_\d+_\d+$)", path)[0] for path in empty_path]
    for id in empty_id:
        print(os.path.join(source_dir, key, id + '.png'))
        os.remove(os.path.join(source_dir, key, id + '.png'))
    for path in empty_path:
        os.rmdir(path)
    print(empty_path)
