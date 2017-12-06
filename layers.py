#!/usr/bin/env python
# coding=utf-8

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace

class FRRU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(FRRU, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.conv_fcn1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        self.conv_fcn2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        self.bn_fcn1 = nn.BatchNorm2d(self.out_channel)
        self.bn_fcn2 = nn.BatchNorm2d(self.out_channel)
        self.relu_fcn = nn.ReLU(inplace=True)
        self.convz_fcn = nn.Conv2d(self.out_channel, 32, kernel_size=1)

    def forward(self, input):
        z, y = input
        batch_size, residual_channel, residual_height, residual_width = z.size()
        batch_size, pooling_channel, pooling_height, pooling_width = y.size()
        assert (residual_height//pooling_height) == (residual_width//pooling_width), 'pooling sizes for height and width are not equal!'

        pool_size = residual_height//pooling_height
        pool= nn.MaxPool2d(pool_size, pool_size, return_indices=True)
        pooled_z, indices = pool(z)
        assert (pooled_z.size(2)==pooling_height) and (pooled_z.size(3)==pooling_width), "pooled z dimension not consistent with y! can't concatenate!"

        cat_input = torch.cat([pooled_z, y], dim=1)
        cat_channel = cat_input.size(1)

        relu1 = self.relu_fcn(self.bn_fcn1(self.conv_fcn1(cat_input)))
        relu2 = self.relu_fcn(self.bn_fcn2(self.conv_fcn2(relu1)))


        conv1 = self.convz_fcn(relu2)
        assert conv1.size(1) == z.size(1), 'residual dimension not consistent!'
        unpool = nn.MaxUnpool2d(pool_size, pool_size)
        z_out = unpool(conv1, indices) + z
        y_out = relu2

        return (z_out, y_out)

class RU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RU, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        if self.in_channel != self.out_channel:
            input = self.conv1(input)
        conv2 = self.conv2(input)
        relu2 = self.relu(conv2)
        conv3 = self.conv3(relu2)

        return conv3+input
        
        
