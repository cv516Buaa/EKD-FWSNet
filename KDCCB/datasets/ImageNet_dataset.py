import numpy as np
import os
import pickle
import torch
import logging
import cv2
import random
import io

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import RandomSampler

import torchvision.transforms as transforms
import torchvision.datasets as datasets

logger = logging.getLogger('global')

def build_ImageNetloader(cfg, training):
    if training:
        transform_cls = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.RandomSizedCrop(cfg['scales']),
                        transforms.ToTensor(),
                        transforms.Normalize(cfg['pi_mean'], cfg['pi_std'])
                        ])

    else:
        transform_cls = transforms.Compose([
                        transforms.Scale(cfg['scales']),
                        transforms.CenterCrop(cfg['crop_size']),
                        transforms.ToTensor(),
                        transforms.Normalize(cfg['pi_mean'], cfg['pi_std'])
                        ])

    logger.info('build dataset from: {}'.format(cfg['meta_url']))

    if training:
        traindir = os.path.join(cfg['meta_url'], 'train')
        loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transform=transform_cls),
        batch_size=cfg['train']['batch_size'], shuffle=True,
        num_workers=cfg['num_workers'], pin_memory=True)
    else:
        valdir = os.path.join(cfg['meta_url'], 'val')
        loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transform=transform_cls),
        batch_size=cfg['test']['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=True)
    return loader