#!/usr/bin/env python
# coding=utf-8

from __future__ import division
import cv2
import os
import numpy as np
from tqdm import tqdm
from ipdb import set_trace

masks_dir = 'split_masks'
merged_dir = 'masks'
mask_height = 256
mask_width = 256

i = 0 
#masks_length = len(os.listdir(masks_dir))
masks_length = 2816
pbar = tqdm(total = masks_length)
while i < masks_length:
    pbar.update(1)
    image_name = 'image'+str(i)+'_'
    mask_lst = [os.path.join(masks_dir, x) for x in os.listdir(masks_dir) if x.startswith(image_name)]
    
    masks = np.zeros((mask_height, mask_width), dtype=np.uint8)
    for mask_name in mask_lst:
        mask = cv2.imread(mask_name, 0)//125
        masks = np.logical_or(masks, mask)

    masks_name = 'mask'+str(i)+'.jpg'
    cv2.imwrite(os.path.join(merged_dir, masks_name), masks.astype(np.uint8)*255)
    i += 1
