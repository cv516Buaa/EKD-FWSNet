import copy
import json
import logging
import random
import re
import shutil

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import cv2

from sklearn.metrics import auc

logger = logging.getLogger('global')

def vis_featuremap(x, out_url, imgfile_list):
    if not os.path.isdir(out_url):
        os.makedirs(out_url)
    N, C, H, W = x.shape
    c = C / 8
    for i in range(N):
        if i < 8:
            plt.figure(figsize=(16, 9))
            imgname = imgfile_list[i][0].split('/')[1]
            out_path_tmp = os.path.join(out_url, imgname)
            for j in range(C):
                plt.subplot(8, c, j+1)
                fm_tmp = x[i, j, :, :]
                plt.imshow(fm_tmp.cpu().numpy())
                plt.axis('off')
            plt.savefig(out_path_tmp)

def save_checkpoint(state, filename):
    torch.save(state, filename +'_best.pth.tar')

def vis_loader(loader, training, num_iters=1000):
    if training:
        for i, sample in enumerate(loader):
            sample_1 = sample['input_images']
            sample_1 = sample_1[0].numpy()
            sample_1 = sample_1.transpose(1, 2, 0) * 255
            out_url = './results/vis_loader/trainimg' + str(i) + '.jpg'
            cv2.imwrite(out_url, sample_1)
            sample_2 = sample['output_images']
            sample_2 = sample_2[0].numpy()
            sample_2 = sample_2.transpose(1, 2, 0) * 255
            out_url = './results/vis_loader/traingt' + str(i) + '.jpg'
            cv2.imwrite(out_url, sample_2)
            if i > num_iters:
                break
    else:
        for i, sample in enumerate(loader):
            sample_1 = sample['input_images']
            sample_1 = sample_1[0].numpy()
            sample_1 = sample_1.transpose(1, 2, 0) * 255
            out_url = './results/vis_loader/valimg' + str(i) + '.jpg'
            cv2.imwrite(out_url, sample_1)
            sample_2 = sample['output_images']
            sample_2 = sample_2[0].numpy()
            sample_2 = sample_2.transpose(1, 2, 0) * 255
            out_url = './results/vis_loader/valgt' + str(i) + '.jpg'
            cv2.imwrite(out_url, sample_2)
            if i > num_iters:
                break