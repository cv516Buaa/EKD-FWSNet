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

def build_C100loader(cfg, training):
    if training:
        transform_cls = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize(cfg['pi_mean'], cfg['pi_std'])
                        ])

    else:
        transform_cls = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(cfg['pi_mean'], cfg['pi_std'])
                        ])

    logger.info('build dataset from: {}'.format(cfg['meta_url']))

    if training:
        loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=cfg['meta_url'], train=True, transform=transform_cls, download=False),
        batch_size=cfg['train']['batch_size'], shuffle=True,
        num_workers=cfg['num_workers'], pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=cfg['meta_url'], train=False, transform=transform_cls),
        batch_size=cfg['test']['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=True)

    return loader

def build_C10loader(cfg, training):
    if training:
        transform_cls = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.RandomResizedCrop(size=cfg['train']['scales'], scale=cfg['train']['augmentation']['ratio']),
                        transforms.ToTensor(),
                        transforms.Normalize(cfg['pi_mean'], cfg['pi_std'])
                        ])

    else:
        transform_cls = transforms.Compose([transforms.Resize(size=(cfg['test']['scales'], cfg['test']['scales'])),
                        transforms.ToTensor(),
                        transforms.Normalize(cfg['pi_mean'], cfg['pi_std'])
                        ])

    logger.info('build dataset from: {}'.format(cfg['meta_url']))

    if training:
        loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=cfg['meta_url'], train=True, transform=transform_cls, download=False),
        batch_size=cfg['train']['batch_size'], shuffle=True,
        num_workers=cfg['num_workers'], pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=cfg['meta_url'], train=False, transform=transform_cls),
        batch_size=cfg['test']['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=True)

    return loader