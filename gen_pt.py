#!/usr/bin/env python
# coding=utf-8

from __future__ import division
import os
import torch
import codecs
import numpy as np
import cv2
from tqdm import tqdm
from ipdb import set_trace

train_img_path ='images'
train_mask_path='masks'
test_img_path = 'test_images'
test_mask_path = 'test_masks'

def read_images(path):
    '''read images from the path'''
    f_lst = [os.path.join(path, x) for x in os.listdir(path)]
    pbar = tqdm(total=len(f_lst))

    img = cv2.imread(f_lst[0])//130
    img_height, img_width, img_channel = img.shape
    imgs = np.zeros((len(f_lst), img_height, img_width, img_channel))
    for i in range(len(f_lst)):
        pbar.update()        
        imgs[i, :, :, :] = cv2.imread(f_lst[i])//130


    output = np.rollaxis(imgs, 3, 1)
    assert output.shape[1] == 3, 'wrong axis roll!'

    return torch.from_numpy(output).type(torch.LongTensor)

def read_masks(mask_path, img_path):
    '''read masks from the path'''
    img_lst = os.listdir(img_path)
    mask_lst = [os.path.join(mask_path, x.replace('image', 'mask')) for x in img_lst]
    pbar = tqdm(total=len(mask_lst))

    mask = cv2.imread(mask_lst[0], 0)//130
    mask_height, mask_width = mask.shape
    masks = np.zeros((len(mask_lst), mask_height, mask_width))
    for i, mask in enumerate(mask_lst):
        pbar.update()
        masks[i, :, :] = cv2.imread(mask, 0)//130

    return torch.from_numpy(masks).type(torch.LongTensor)
     

print(os.getcwd())
data = './data'
training_file = 'training.pt'
test_file = 'test.pt'

print('Processing...')

training_set = (
    read_images(train_img_path),
    read_masks(train_mask_path, train_img_path)
)
test_set = (
    read_images(test_img_path),
    read_masks(test_mask_path, test_img_path)
)

with open(os.path.join(data, training_file), 'wb') as f:
    torch.save(training_set, f)
with open(os.path.join(data, test_file), 'wb') as f:
    torch.save(test_set, f)

print('Done!')
