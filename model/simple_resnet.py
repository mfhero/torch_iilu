#!/usr/bin/env python
##########################################################
# File Name: model/simple_resnet.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-05-10 14:52:08
##########################################################

import torch
import torch.nn as nn
from torchvision.models.resnet import  ResNet, BasicBlock

class Generator(nn.Module):
    def __init__(self, ngpu, layers, inplanes = 3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.inplanes = 32
        self.conv1 = nn.Conv2d(inplanes, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, layers[2], stride=2)

        self.deconv1 = self._make_downsample(128)
        self.deconv2 = self._make_downsample(64)
        self.deconv3 = self._make_downsample(32)

        self.shortcut1 = self._make_shortcut(64, 128)
        self.shortcut2 = self._make_shortcut(32, 64)

        self.output = nn.Conv2d(self.inplanes, 3,
                kernel_size = 1, stride = 1, bias = True)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #    elif isinstance(m, nn.BatchNorm2d):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)
    
    def _make_downsample(self, planes):
        s = nn.Sequential(
            nn.ConvTranspose2d(self.inplanes, planes, 
                kernel_size = 2,
                stride = 2),
            nn.BatchNorm2d(planes),
            #self.relu,
            )
        self.inplanes = planes 
        return s
    
    def _make_shortcut(self, in_planes, out_planes):
        return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, 1, bias = False),
                nn.BatchNorm2d(out_planes),
            )
       

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        d1 = self.deconv1(x3)
        d1 = d1 + self.shortcut1(x2)
        d1 = self.relu(d1)
        d2 = self.deconv2(d1)
        d2 = d2 + self.shortcut2(x1)
        d2 = self.relu(d2)
        d3 = self.deconv3(d2)
        d3 = self.relu(d3)
        output = self.output(d3)
        output = self.relu(output)
        return output

if __name__ == "__main__":
    print Generator([3,3,3])
