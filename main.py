#!/usr/bin/env python
# coding=utf-8

from __future__ import division
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from networks import FRRN
import torch.nn.functional as F
from utils import one_hot, probs2img
from ipdb import set_trace

data_dir = './data'
train_file = 'training.pt'
test_file = 'test.pt'

img_height = 256
img_width = 256
img_channels = 3
cls_num = 2
learning_rate = 0.01
K = 128*32
batch_size = 8
display_step = 10
threshold = 0.2

network = FRRN(img_channels, img_height, img_width, cls_num).cuda()

train_tensor = torch.load(os.path.join(data_dir, train_file))
train_dataset = torch.utils.data.TensorDataset(train_tensor[0], train_tensor[1])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

test_tensor = torch.load(os.path.join(data_dir, test_file))
test_dataset = torch.utils.data.TensorDataset(test_tensor[0], test_tensor[1])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle=True)

def train(epoch):
    print('epoch {} start!'.format(epoch))
    optimizer = optim.Adam(network.parameters(), lr = learning_rate)
    train_loss = None 
    network.train()
    
    for ind, train_tup in enumerate(train_loader):
        data, label = train_tup
        label = one_hot(label, length = cls_num)
        data = Variable(data).cuda()
        label = Variable(label).cuda()

        optimizer.zero_grad()
        probs = network(data)
        loss = network.loss(probs, label, K)
        loss.backward()
        optimizer.step()
    if ind % display_step == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss:{:.3f}'.format(
            epoch, 
            ind*len(data),
            len(train_loader.dataset),
            100. * ind/len(train_loader),
            loss.data[0]
            ))

def test(epoch):
    network.eval()
    test_loss = 0
    correct = 0
    for ind, test_tup in enumerate(test_loader):
        data, _label = test_tup
        label = one_hot(_label, length = cls_num)
        data = Variable(data).cuda()
        label = Variable(label).cuda()
    
        probs = network(data)
        output_img = probs2img(probs, threshold)
        loss = network.loss(probs, label, K).cpu()
        test_loss = test_loss + loss.data[0]
        correct += output_img.data.eq(_label).type(torch.LongTensor).sum()

    test_loss = test_loss/len(test_loader.dataset)*1.0
    test_accuracy = 1.0*correct/(len(test_loader.dataset)*256*256)

    print('\nTest set {}: Average Loss: {:.4f}, Accuracy: {:.0f}%)'.format(
        epoch,
        test_loss,
        100.*test_accuracy
        ))
        

if __name__=='__main__':
    for epoch in range(1, 10):
        train(epoch)
        test(epoch)
        torch.save(network, './weights/model_'+str(epoch)+'.pt')
        

