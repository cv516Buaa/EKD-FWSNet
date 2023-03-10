import copy
import logging

from .cifar_dataset import build_C100loader, build_C10loader
from .CUB200_dataset import build_CUB200loader
from .tinyImageNet_dataset import build_tinyImageNetloader
from .ImageNet_dataset import build_ImageNetloader

logger = logging.getLogger('global')

def build_dataloader(cfg_dataset):
    train_loader = None
    if cfg_dataset.get('train', None):
        train_loader = build(cfg_dataset, training=True)

    test_loader = None
    if cfg_dataset.get('test', None):
        test_loader = build(cfg_dataset, training=False)

    logger.info('build dataset done')
    return train_loader, test_loader

def build(cfg, training): 
    cfg = copy.deepcopy(cfg)
    if training:
        cfg.update(cfg.get('train', {}))
    else:
        cfg.update(cfg.get('test', {}))

    dataset = cfg['type']
    if dataset == 'CIFAR100':
        data_loader = build_C100loader(cfg, training)
    elif dataset == 'CIFAR10':
        data_loader = build_C10loader(cfg, training)
    elif dataset == 'tiny-ImageNet':
        data_loader = build_tinyImageNetloader(cfg, training)
    elif dataset == 'ImageNet':
        data_loader = build_ImageNetloader(cfg, training)
    elif dataset == 'CUB200':
        data_loader = build_CUB200loader(cfg, training, is_test=None)
    else:
        raise NotImplementedError("Not supported")

    return data_loader