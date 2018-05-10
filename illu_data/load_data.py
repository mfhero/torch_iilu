#!/usr/bin/env python
##########################################################
# File Name: illu_data/load_data.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-05-10 17:52:31
##########################################################

import os

def load_dataset(src_dir, filelist):
    filelist = os.path.join(src_dir, filelist)
    labels = []
    data = []
    for line_cnt, line in enumerate(open(filelist).readlines()):
        label_filename = line[:-1]
        labels.append(os.path.join(src_dir, label_filename))
        base_filename = label_filename[:-4]
        while base_filename[-1] >= '0' and base_filename[-1] <= '9':
            base_filename = base_filename[:-1]
        tmp_data = []
        for i in xrange(1, 11):
            tmp_data.append(os.path.join(src_dir, base_filename + str(i) + '.png'))
        data.append(tmp_data)
    #labels = tf.constant(label)
    #data = tf.constant(data)

    #dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    #dataset = dataset.shuffle(buffer_size = line_cnt)
    #dataset = dataset.map(_parse_function)

    return data, labels


