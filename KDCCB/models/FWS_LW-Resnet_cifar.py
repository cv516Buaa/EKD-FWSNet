import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import copy

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock_se(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_se, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 4))
        self.fc2 = nn.Linear(in_features=round(planes / 4), out_features=planes)
        self.sigmoid = nn.Sigmoid()
        self.stride = stride

        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0),out.size(1),1,1)
        out = out * original_out

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        a = self.shortcut(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FWS_ResNet(nn.Module):
    def __init__(self, block, FA_block, num_blocks, KL_w=50, num_classes=10):
        super(FWS_ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        ## IMPORTANT: reset in_channel (* expansion)
        self.in_planes = 16
        self.aux_layer2 = self._make_layer(FA_block, 32, num_blocks[1], stride=2)
        self.aux_layer3 = self._make_layer(FA_block, 64, num_blocks[2], stride=2)

        self.linear = nn.Linear(64, num_classes)
        self.aux_linear_1 = nn.Linear(64, num_classes)
        self.aux_linear_2 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(p=0.2)
        self.att_loss = nn.MSELoss()
        self.KL_w = KL_w

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out) 
        aux_out = out
        #cam_out = out
        layer3_out = []

        out = self.layer2(out)
        aux_out_2 = out
        out = self.layer3(out)
        layer3_out.append(out)

        aux_out = self.dropout(aux_out)
        aux_out_1 = self.aux_layer2(aux_out)
        aux_out_1 = self.aux_layer3(aux_out_1)
        layer3_out.append(aux_out_1)

        aux_out_2 = self.dropout(aux_out_2)
        aux_out_2 = self.aux_layer3(aux_out_2)
        layer3_out.append(aux_out_2)
        
        output = []
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        aux_out_1 = F.avg_pool2d(aux_out_1, aux_out_1.size()[3])
        aux_out_1 = aux_out_1.view(aux_out_1.size(0), -1)

        aux_out_2 = F.avg_pool2d(aux_out_2, aux_out_2.size()[3])
        aux_out_2 = aux_out_2.view(aux_out_2.size(0), -1)
        
        output.append(self.linear(out))
        output.append(self.aux_linear_1(aux_out_1))
        output.append(self.aux_linear_2(aux_out_2))

        layer3_student = self.self_attention(layer3_out[0])
        layer3_teacher = (self.self_attention(layer3_out[0])+self.self_attention(layer3_out[1])+self.self_attention(layer3_out[2]))/3
        att_Loss = self.att_loss(layer3_student, layer3_teacher)

        out_et = (output[0] + output[1] + output[2])/3
        #out_et = copy.copy(out_et)
        KL_Loss = self.KL_loss(out_et, output[0], T=3)
        KD_Loss = self.KL_w * KL_Loss + att_Loss
        return output, KD_Loss

def FWS_resnet20(**kwargs):
    return FWS_ResNet(BasicBlock, BasicBlock, [3, 3, 3], **kwargs)

def FWS_resnet32(**kwargs):
    return FWS_ResNet(BasicBlock, BasicBlock, [5, 5, 5], **kwargs)

def FWS_resnet44(**kwargs):
    return FWS_ResNet(BasicBlock, BasicBlock, [7, 7, 7], **kwargs)

def FWS_resnet56(**kwargs):
    return FWS_ResNet(BasicBlock, BasicBlock, [9, 9, 9], **kwargs)