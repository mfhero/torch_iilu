#!/usr/bin/env python
##########################################################
# File Name: illu_data/illu_dataset.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-05-10 17:56:28
##########################################################

import torch.utils.data as data
from torchvision.datasets.folder import default_loader
import os
import numpy as np

import load_data
import random

def _crop(img, i, j, h, w):
    return img.crop((j, i, j + w, i + h))

def _get_crop_imgs(imgs, out_size):
    '''
    all images should have same shape, PILImage
    '''
    w, h = imgs[0].size
    tw, th = out_size 
    if w == tw and h == th:
        return imgs
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    
    func = lambda x : _crop(x, i, j, th, tw)
    return map(func, imgs)

class IlluDataset(data.Dataset):
    def __init__(self, root, transform = None, target_transform = None):
        self.root = root
        self.data, self.labels = load_data.load_dataset(root, "filelist.txt")
        self.transform = transform 
        self.target_transform = target_transform
        self.out_size = (224, 224)

    def preprocess(self, datas, label):
        crop_pics = _get_crop_imgs(datas + [label], self.out_size)
        return crop_pics[:-1], crop_pics[-1]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        
        pics = []
        for im_name in self.data[index]:
            pics.append(default_loader(im_name))
        
        target = default_loader(label)

        pics, target = self.preprocess(pics, target)
        pic = np.concatenate([np.array(v) for v in pics], axis = -1)
        target = np.array(target)
        
        if self.transform:
            pic = self.transform(pic)

        if self.target_transform:
            target = self.target_transform(target)
        
        return pic, target
    
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    c = IlluDataset("/data3/lzh/10000x672x672_box_diff/")
    for img, label in c:
        print img.shape
        print label.shape
