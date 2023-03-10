import argparse
import os
import time
import sys
import yaml
import logging
import cv2
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from collections import Counter
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import torch.nn.functional as F

from KDCCB.datasets.data_builder import build_dataloader
from KDCCB.models.model_builder import ModelBuilder

from KDCCB.utils.log_helper import init_log 
from KDCCB.utils.optim_helper import build_optimizer, adjust_learning_rate_poly
from KDCCB.utils.evaluator import AverageMeter,accuracy
from KDCCB.utils.loss_helper import LabelSmoothingCrossEntropy, FocalLoss
from KDCCB.utils.misc_helper import vis_loader, save_checkpoint

parser = argparse.ArgumentParser(description='classfication code base')
parser.add_argument('--config', dest='config', help='settings of classification in yaml format')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--num_epoch', default=300, type=int, metavar='N', help='num of training epochs')
parser.add_argument('--is_offline', default=0, type=int, metavar='N', help='decide whether if it is offline testing')

args = parser.parse_args()
init_log('global', logging.INFO)
logger = logging.getLogger('global')

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.best_prec = 0
        self.best_prect = 0

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ## 1. config file configuration
        self.cfg = yaml.load(open(self.args.config, 'r'), Loader=yaml.Loader)
        logger.info('cfg loading is done:\n{}'.format(self.cfg))
        ## 2. dataloader
        self.train_loader, self.test_loader = build_dataloader(self.cfg['dataloader'])
        #vis_loader(self.test_loader, training=False)
        ## 3. model
        model = ModelBuilder(self.cfg['net'])
        #model = nn.DataParallel(model)
        model = model.to(self.device)
        self.model = model
        self.branch_num = self.cfg['net']['branch_num']
        #logger.info('model building is done:\n{}'.format(model))
        ## 3.1. load model 
        if self.args.is_offline == 1:
            #for submodule in self.model.children():
            #    self.load_model(submodule)
            self.load_model(self.model)
        ## 4. optimizer
        cfg_opt = self.cfg['train_params']
        self.optimizer = build_optimizer(cfg_opt['optimizer'], self.model)
        self.base_lr = cfg_opt['optimizer']['kwargs']['lr']
        ## 5. LR scheduler
        lr_kwargs = cfg_opt['lr_schedule']['kwargs']
        if cfg_opt['lr_schedule']['type'] == 'MultiStepLR':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **lr_kwargs)
        if cfg_opt['lr_schedule']['type'] == 'Cosine':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **lr_kwargs)
        ## 6. Loss function
        loss_type = self.cfg['dataloader']['train']['loss_type']
        if loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        if loss_type == 'focal':
            self.criterion = FocalLoss().to(self.device)
        if loss_type == 'lbs_cross_entropy':
            self.criterion = LabelSmoothingCrossEntropy(self.cfg['dataloader']['num_cls']).to(self.device)

    def load_model(self, model):
        pth_url = self.cfg['load_params']['load_dir']
        checkpoint_dict = torch.load(pth_url)
        model_stat_dict = checkpoint_dict['state_dict']
        for params in model.state_dict():
            if params  in model_stat_dict:
                model.state_dict()[params].copy_(model_stat_dict[params])

    def train(self, epoch):
        logger.info("epoch: {}, start training".format(epoch))
        self.model.train()
        self.model = self.model.to(self.device)
        losses = AverageMeter()
        top1 = AverageMeter()

        end = time.time()
        # 1. adjust learning rate 
        self.lr_scheduler.step()
        for i, (input, target) in enumerate(self.train_loader):
            ####
            # 2. zero_grad
            # 3. model forward
            # 4. calculate loss
            # 5. backward and update
            self.optimizer.zero_grad()
            inp = input.to(self.device)
            lbl = target.to(self.device)
            output, KL_Loss = self.model(inp)
            loss = self.criterion(output[0], lbl)

            for j in range(1, len(output)):
                loss += (1.0 / (self.branch_num-1)) * self.criterion(output[j], lbl)
            
            loss += KL_Loss
            losses.update(loss.item(), inp.size(0))
            loss.backward()
            self.optimizer.step()
            ####
            if (i + 1) % args.print_freq == 0:
                logger.info('epoch: {}: [{}/{}]' .format(epoch, i, len(self.train_loader)))
                logger.info('Loss: {}' .format(float('%.4f' % losses.val)))

    def test(self, epoch):
        logger.info("epoch: {}, start testing".format(epoch))
        self.model.eval()
        self.model = self.model.to(self.device)
        losses = AverageMeter()
        
        top1 = []
        top_et = AverageMeter()
        for i in range(self.branch_num):
            top1_tmp = AverageMeter()
            top1.append(top1_tmp)

        with torch.no_grad():
            for i, (input, target) in enumerate(self.test_loader):
                ####
                # 1. model forward
                # 2. calculate loss and accuracy
                inp = input.to(self.device)
                lbl = target.to(self.device)
                
                output, KL_Loss = self.model(inp)
                loss = self.criterion(output[0], lbl)
                for j in range(1, len(output)):
                    loss += 0.5 * self.criterion(output[j], lbl)
                loss += KL_Loss
                losses.update(loss.item(), inp.size(0))
                
                prec_bl = accuracy(output[0].data, lbl)
                top1[0].update(prec_bl[0], input.size(0))
                out_et = output[0]
                for j in range(1, self.branch_num):
                    prec_tmp = accuracy(output[j].data, lbl)
                    top1[j].update(prec_tmp[0], input.size(0))
                    out_et += output[j]

                prec_et = accuracy(out_et.data, lbl)
                top_et.update(prec_et[0], input.size(0))
        if top1[0].avg > self.best_prec:
            if not os.path.exists(self.cfg['save_params']['save_dir']):
                os.makedirs(self.cfg['save_params']['save_dir'])
            self.best_prec = top1[0].avg
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.cpu().state_dict(),
                'best_prec': self.best_prec,
                'optimizer' : self.optimizer.state_dict(),
            }, self.cfg['save_params']['save_dir'] + 'ckpt_e')
        logger.info('Loss: {}' .format(float('%.4f' % losses.val)))
        logger.info('prec_et: {}' .format(float('%.4f' % top_et.avg)))
        logger.info('prec: {}' .format(float('%.4f' % top1[0].avg)))
        logger.info('best_prec: {}' .format(float('%.4f' % self.best_prec)))
        logger.info('#######')
        for i in range(1, self.branch_num):
            logger.info('prec_b{}: {}' .format(i, float('%.4f' % top1[i].avg)))
        
    def off_test(self):
        logger.info("start offline testing")
        self.model.eval()
        self.model = self.model.to(self.device)
        losses = AverageMeter()
        top1 = AverageMeter()

        with torch.no_grad():
            for i, (input, target) in enumerate(self.test_loader):
                ####
                # 1. model forward
                # 2. calculate loss and accuracy
                inp = input.to(self.device)
                lbl = target.to(self.device)
                out, _ = self.model(inp)
                loss = self.criterion(out[0], lbl)
                losses.update(loss.item(), inp.size(0))
                prec1 = accuracy(out[0].data, lbl)
                top1.update(prec1[0], input.size(0))
                ####
        logger.info('Loss: {}' .format(float('%.4f' % losses.val)))
        logger.info('prec: {}' .format(float('%.4f' % top1.avg)))

def main():
    global args
    trainer = Trainer(args)
    if args.is_offline == 0:
        for epoch in range(args.num_epoch):
            trainer.train(epoch)
            trainer.test(epoch)
    else:
        trainer.off_test()

if __name__ == '__main__':
    main()
