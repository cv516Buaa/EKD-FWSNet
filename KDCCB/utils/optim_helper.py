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

logger = logging.getLogger('global')

def build_cls_instance(module, cfg):
    cls = getattr(module, cfg['type'])
    return cls(**cfg['kwargs'])

def build_optimizer(cfg_optim, model):
    lr = cfg_optim['kwargs']['lr']
    trainable_params = model.parameters()
    cfg_optim['kwargs']['params'] = trainable_params
    optim_type = cfg_optim['type']
    optimizer = build_cls_instance(torch.optim, cfg_optim)
    logger.info('build optimizer done')
    return optimizer

def adjust_learning_rate_poly(lr, max_steps, optimizer, iter, power=0.9):
    base_lr = lr
    max_iter = max_steps
    reduce = ((1-float(iter)/max_iter)**(power))
    lr = base_lr * reduce
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr