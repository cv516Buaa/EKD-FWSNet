import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.init as init

def load_state_dict_res(model, model_stat_dict):
    for params in model.state_dict():
        if 'tracked' in params[-7:]:
            continue
        elif 'fc' in params[0: 2]:
            continue
        elif 'aux_layer' in params:
            model.state_dict()[params].copy_(model_stat_dict[params[4:]])
            continue
        else:
            if params in model_stat_dict:
                model.state_dict()[params].copy_(model_stat_dict[params])

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class BasicBlock_se(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock_se, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
    
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_features=out_channel, out_features=round(out_channel / 4))
        self.fc2 = nn.Linear(in_features=round(out_channel / 4), out_features=out_channel)
        self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0),out.size(1),1,1)
        out = out * original_out

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck_se(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_se, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_features=planes * 4, out_features=round(planes / 4))
        self.fc2 = nn.Linear(in_features=round(planes / 4), out_features=planes * 4)
        self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0),out.size(1),1,1)
        out = out * original_out
      
        out += residual
        out = self.relu(out)

        return out

class FWS_ResNet_HE_LC(nn.Module):
    def __init__(self, block, FA_block, num_blocks, KL_w=100, MSE_w=0.25, num_classes=100):
        super(FWS_ResNet_HE_LC, self).__init__()
        self.in_channel = 64

        self.num_classes = num_classes
        ## Sorry for lazy hardcoding
        ## For CIFAR with small-scale images
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ## For CUB and ImageNet with large-scale images
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        ## For tiny-ImageNet, this maxpool can be removed
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        ## IMPORTANT: reset in_channel (* expansion)
        self.in_channel = 64 * block.expansion 
        self.aux_layer2 = self._make_layer(FA_block, 128, num_blocks[1], stride=2)
        self.aux_layer3 = self._make_layer(FA_block, 256, num_blocks[2], stride=2)
        self.aux_layer4 = self._make_layer(FA_block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        if block.expansion == 1:
            self.classifier = nn.Linear(512, self.num_classes, bias=False)
            self.aux_classifier1 = nn.Linear(512, self.num_classes, bias=False)
            self.aux_classifier2 = nn.Linear(512, self.num_classes, bias=False)
            self.aux_classifier3 = nn.Linear(512, self.num_classes, bias=False)
        elif block.expansion == 4:
            self.classifier = nn.Linear(2048, self.num_classes, bias=False)
            self.aux_classifier1 = nn.Linear(2048, self.num_classes, bias=False)
            self.aux_classifier2 = nn.Linear(2048, self.num_classes, bias=False)
            self.aux_classifier3 = nn.Linear(2048, self.num_classes, bias=False)

        self.dropout = nn.Dropout(p=0.1)
        self.softmax  = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.att_loss = nn.MSELoss()
        self.KL_w = KL_w
        self.MSE_w = MSE_w

        self.apply(_weights_init)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def cam(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
    
    def KL_loss(self, teacher, Ss, T=1):
        KL_loss = nn.KLDivLoss()(F.log_softmax(Ss/T, dim=1),
                             F.softmax(teacher/T, dim=1)) * (T * T)
        return KL_loss
    
    def self_attention(self, feature):
        N, C, H, W = feature.shape
        s_att = torch.sum(feature,dim = 1) 
        s = torch.reshape(s_att,(N,H*W))   
        mean = torch.mean(s, 1)
        mean = torch.reshape(mean,(N, 1, 1))
        std = torch.std(s, 1)
        std = torch.reshape(std,(N, 1, 1))  
        s_att = (s_att - mean) / std
        return s_att

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        out = self.layer1(x)
        aux_out = out
        layer4_out = []

        out = self.layer2(out)
        aux_out_2 = out
        out = self.layer3(out)
        aux_out_3 = out
        out = self.layer4(out)
        layer4_out.append(out)

        aux_out = self.dropout(aux_out)    
        aux_out_1 = self.aux_layer2(aux_out)     
        aux_out_1 = self.aux_layer3(aux_out_1)
        aux_out_1 = self.aux_layer4(aux_out_1)
        layer4_out.append(aux_out_1)

        aux_out_2 = self.dropout(aux_out_2)
        aux_out_2 = self.aux_layer3(aux_out_2)
        aux_out_2 = self.aux_layer4(aux_out_2)
        layer4_out.append(aux_out_2)

        aux_out_3 = self.dropout(aux_out_3)
        aux_out_3 = self.aux_layer4(aux_out_3)
        layer4_out.append(aux_out_3)

        output = []
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        aux_out_1 = self.avgpool(aux_out_1)
        aux_out_1 = aux_out_1.view(aux_out_1.size(0), -1)

        aux_out_2 = self.avgpool(aux_out_2)
        aux_out_2 = aux_out_2.view(aux_out_2.size(0), -1)

        aux_out_3 = self.avgpool(aux_out_3)
        aux_out_3 = aux_out_3.view(aux_out_3.size(0), -1)

        output.append(self.classifier(out))
        output.append(self.aux_classifier1(aux_out_1))
        output.append(self.aux_classifier2(aux_out_2))
        output.append(self.aux_classifier3(aux_out_3))

        layer4_student = self.self_attention(layer4_out[0])
        layer4_teacher = (self.self_attention(layer4_out[0])+self.self_attention(layer4_out[1])+self.self_attention(layer4_out[2])+self.self_attention(layer4_out[3]))/4
        att_Loss = self.att_loss(layer4_student, layer4_teacher)

        out_et = (output[0] + output[1] + output[2] + output[3])/4
        KL_Loss = self.KL_loss(out_et, output[0], T=3)
        KD_Loss = self.KL_w * KL_Loss + self.MSE_w * att_Loss
        return output, KD_Loss

## For CIFAR100 and ImageNet
def FWS_resnet18(**kwargs):
    return FWS_ResNet_HE_LC(block=BasicBlock, FA_block=BasicBlock, num_blocks=[2,2,2,2], **kwargs)

def FWS_resnet34(**kwargs):
    return FWS_ResNet_HE_LC(block=BasicBlock, FA_block=BasicBlock, num_blocks=[3,4,6,3], **kwargs)

def FWS_resnet50(**kwargs):
    return FWS_ResNet_HE_LC(block=Bottleneck, FA_block=Bottleneck, num_blocks=[3,4,6,3], **kwargs)

def FWS_resnet101(**kwargs):
    return FWS_ResNet_HE_LC(block=Bottleneck, FA_block=Bottleneck, num_blocks=[3,4,23,3], **kwargs)

## For CUB200 and tiny-ImageNet
def FWS_resnet18_pretrained(**kwargs):
    model = FWS_ResNet_HE_LC(block=BasicBlock, FA_block=BasicBlock, num_blocks=[2,2,2,2], **kwargs)
    checkpoint_dict = torch.load('../../KDCCB/models/backbone/resnet18-5c106cde.pth')
    load_state_dict_res(model, checkpoint_dict)
    return model

def FWS_resnet34_pretrained(**kwargs):
    model = FWS_ResNet_HE_LC(block=BasicBlock, FA_block=BasicBlock, num_blocks=[3,4,6,3], **kwargs)
    checkpoint_dict = torch.load('../../KDCCB/models/backbone/resnet34-333f7ec4.pth')
    load_state_dict_res(model, checkpoint_dict)
    return model

def FWS_resnet50_pretrained(**kwargs):
    model = FWS_ResNet_HE_LC(block=Bottleneck, FA_block=Bottleneck, num_blocks=[3,4,6,3], **kwargs)
    checkpoint_dict = torch.load('../../KDCCB/models/backbone/resnet50-19c8e357.pth')
    load_state_dict_res(model, checkpoint_dict)
    return model

def FWS_resnet101_pretrained(**kwargs):
    model = FWS_ResNet_HE_LC(block=Bottleneck, FA_block=Bottleneck, num_blocks=[3,4,23,3], **kwargs)
    checkpoint_dict = torch.load('../../KDCCB/models/backbone/resnet101-5d3b4d8f.pth')
    load_state_dict_res(model, checkpoint_dict)
    return model