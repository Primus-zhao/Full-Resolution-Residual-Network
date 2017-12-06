#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn.functional as F
from ipdb import set_trace
epsilon = 1e-9

def one_hot(x, length):
    batch_size, h, w = x.size()
    x = x.view(-1, 1)
    y = torch.LongTensor(x)

    y_onehot = torch.FloatTensor(batch_size*h*w, length).zero_()
    y_onehot.scatter_(1, y, 1)
    y_onehot = y_onehot.view(batch_size, h, w, 2)

    return y_onehot

def probs2img(probs, threshold):
    batch_size, h, w, c = probs.size()
    assert c==2, 'output probability channel not right!'

    return probs[:, :, :, 1].gt(threshold).type(torch.LongTensor)
