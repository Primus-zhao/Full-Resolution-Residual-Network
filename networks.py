#!/usr/bin/env python
# coding=utf-8

import torch 
import torch.nn as nn
import torch.nn.functional as F
from layers import RU, FRRU
from ipdb import set_trace

class FRRN(nn.Module):
    def __init__(self, img_channel, img_height, img_width, cls_num):
        super(FRRN, self).__init__()
        self.img_channel = img_channel
        self.img_height = img_height
        self.img_width = img_width
        self.cls_num = cls_num
        self.init_conv = nn.Conv2d(img_channel, out_channels=48, kernel_size=5, padding=2)
        self.init_bn = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(48, 32, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.init_rus = nn.ModuleList([RU(48, 48) for i in range(3)])
        self.rus = nn.ModuleList([RU(128, 48), RU(48, 48), RU(48, 48)])
        self.frrus1 = nn.ModuleList([FRRU(80, 96), FRRU(128, 96), FRRU(128, 96)])
        self.frrus2 = nn.ModuleList([FRRU(128, 192), FRRU(224, 192), FRRU(224, 192), FRRU(224, 192)])
        self.frrus3 = nn.ModuleList([FRRU(224, 384), FRRU(416, 384)]) 
        self.frrus4 = nn.ModuleList([FRRU(416, 384), FRRU(416, 384)]) 
        self.frrus5 = nn.ModuleList([FRRU(416, 192), FRRU(224, 192)]) 
        self.frrus6 = nn.ModuleList([FRRU(224, 192), FRRU(224, 192)]) 
        self.frrus7 = nn.ModuleList([FRRU(224, 96), FRRU(128, 96)]) 
        _, self.indices = self.pool(torch.ones(1, img_height, img_width))
        '''
        self.frrus1 = nn.ModuleList([FRRU(96) for i in range(3)])
        self.frrus2 = nn.ModuleList([FRRU(192) for i in range(4)])
        self.frrus3 = nn.ModuleList([FRRU(384) for i in range(2)])
        self.frrus4 = nn.ModuleList([FRRU(384) for i in range(2)])
        self.frrus5 = nn.ModuleList([FRRU(192) for i in range(2)])
        self.frrus6 = nn.ModuleList([FRRU(192) for i in range(2)])
        self.frrus7 = nn.ModuleList([FRRU(96) for i in range(2)])
        '''
        self.out_conv = nn.Conv2d(48, cls_num, kernel_size=1)
        
    def forward(self, img):

        batch_size = img.size(0)
        relu0 = self.relu(self.init_bn(self.init_conv(img.float())))
        ru0 = relu0
        # 3* RU_48 layer
        for i in range(len(self.init_rus)):
            ru0 = self.init_rus[i](ru0) 
        #left branch of layer1
        y1, indices1 = self.pool(ru0)
        #right branch of layer1
        z1 = self.conv1(ru0) 
        for i in range(len(self.frrus1)):
            z1, y1 = self.frrus1[i]((z1, y1))
        
        z2 = z1
        y2, indices2 = self.pool(y1)
        for i in range(len(self.frrus2)):
            z2, y2 = self.frrus2[i]((z2, y2))

        z3 = z2
        y3, indices3 = self.pool(y2)
        for i in range(len(self.frrus3)):
            z3, y3 = self.frrus3[i]((z3, y3))

        z4 = z3
        y4, indices4 = self.pool(y3)
        for i in range(len(self.frrus4)):
            z4, y4 = self.frrus4[i]((z4, y4))

        z5 = z4
        b, c, h, w = y4.size()
        _, indices = self.pool(torch.ones(b, c, 2*h, 2*w).cuda())
        y5 = self.unpool(y4, indices)
        for i in range(len(self.frrus5)):
            z5, y5 = self.frrus5[i]((z5, y5))

        z6 = z5 
        b, c, h, w = y5.size()
        _, indices = self.pool(torch.ones(b, c, 2*h, 2*w).cuda())
        y6 = self.unpool(y5, indices)
        for i in range(len(self.frrus6)):
            z6, y6 = self.frrus6[i]((z6, y6))

        z7 = z6
        b, c, h, w = y6.size()
        _, indices = self.pool(torch.ones(b, c, 2*h, 2*w).cuda())
        y7 = self.unpool(y6, indices)
        for i in range(len(self.frrus7)):
            z7, y7 = self.frrus7[i]((z7, y7))

        b, c, h, w = y7.size()
        _, indices = self.pool(torch.ones(b, c, 2*h, 2*w).cuda())
        y7 = self.unpool(y7, indices)
        
        cat_zy = torch.cat([z7, y7], dim=1) 
        ru1 = cat_zy
        for i in range(len(self.rus)):
            ru1 = self.rus[i](ru1)

        digit = self.out_conv(ru1)
        cls_prob = F.softmax(digit).permute(0, 2, 3, 1)

        return cls_prob


    def loss(self, cls_prob, target, K, batch_average=True):
        '''target is in one_hot format'''
        '''FRRN use bootstrap loss for pixel loss calculation'''
        
        batch_size, image_height, image_width, cls_num = target.size()
        image_size = image_height*image_width
        t = target.contiguous().view(batch_size, image_size, cls_num)
        prob = cls_prob.contiguous().view(batch_size, image_size, cls_num)
        # target_prob -> [batch_size, image_size]
        target_prob, _ = torch.max(prob * t, dim=2)

        sorted_target_prob, _ = target_prob.sort(dim=1)
        # false_target_prob -> [batch_size, tK]
        false_target_prob = sorted_target_prob[:, :K] 
        
        loss = -torch.sum(torch.log(false_target_prob))/K
        if batch_average: loss = loss.mean()

        return loss




