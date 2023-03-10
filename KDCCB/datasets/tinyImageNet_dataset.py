import numpy as np
import os
import pickle
import torch
import logging
import cv2
import random
from PIL import Image
import io

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as transforms
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import RandomSampler

logger = logging.getLogger('global')

def build_tinyImageNetloader(cfg, training):
    normalize = transforms.Normalize(mean=cfg['pi_mean'], std=cfg['pi_std'])
    if training:
        cfg_aug = cfg['train']['augmentation']['colorjitter']
        transform_cls = transforms.Compose([
            transforms.RandomResizedCrop(cfg['scales'], cfg['augmentation']['ratio']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(cfg['augmentation']['ro_angle']),
            transforms.ColorJitter(brightness=cfg_aug[0], contrast=cfg_aug[1], saturation=cfg_aug[2], hue=cfg_aug[3]),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform_cls = transforms.Compose([
            transforms.Resize(cfg['scales']),
            transforms.CenterCrop(cfg['crop_size']),
            transforms.ToTensor(),
            normalize
        ])
    logger.info('build dataset from: {}'.format(cfg['meta_url']))

    if training:
        split = 'train'
    else:
        split = 'test'
     
    dataset = tinyImageNetDataset(
        cfg['meta_url'],
        cfg['num_cls'],
        transform_fn=transform_cls,
        split=split)
    
    loader = DataLoader(
        dataset, batch_size=cfg[split]['batch_size'], shuffle=cfg[split]['shuffle'],
        num_workers=cfg['num_workers'], pin_memory=False)

    return loader

class tinyImageNetDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 num_classes,
                 transform_fn,
                 split='train'):
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.transform_fn = transform_fn
        self.split = split
        self.imageset_dir = os.path.join(self.root_dir, 'tiny_imagenet-200_all')

        if self.split == 'train':
            self.split_dir = os.path.join(self.root_dir, 'anno', 'train.txt')
        if self.split == 'test':
            self.split_dir = os.path.join(self.root_dir, 'anno', 'val.txt')
        
        self.image_idx_list = [x.strip() for x in open(self.split_dir).readlines()]
        self.num_sample = self.image_idx_list.__len__()

        self.imglb_list = self.get_img_label_list(self.image_idx_list)     

    # for train and val
    def get_img_label_list(self, imglist):
        imglb_list = []
        for i in range(len(imglist)):
            imgname, cls_name_id = imglist[i].split('  ')
            imglb_list.append((imgname, int(cls_name_id)-1))
        return imglb_list
    
    def get_img_size(img_url):
        img = cv2.imread(img_url)
        H, W, _ = img.shape
        size = H * W
        return size

    def get_imglb(self, idx):
        image_file = os.path.join(self.imageset_dir, self.imglb_list[idx][0])
        assert os.path.exists(image_file)
        img = Image.open(image_file)
        lb = self.imglb_list[idx][1]
        return img, lb

    def __len__(self):
        return len(self.imglb_list)

    def __getitem__(self, idx):
        '''
        Arguments:
            idx(int): index of image, 0 <= idx < len(self)

        Returns:
            training:(img, label) (tuple)
                img: (H, W, 3)
                label: int
            test:(img)
                img: (H, W, 3)
        '''
        img, lb = self.get_imglb(idx)
        img = img.convert('RGB')
        if self.transform_fn:
            img = self.transform_fn(img)
        
        return img, lb