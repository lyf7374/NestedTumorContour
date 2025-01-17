import numpy as np
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 

dropout =0.1
def normalization(planes, norm='bn', NN=False):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    if NN==True:
        m = nn.BatchNorm1d(planes)
    return m

class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0, norm='bn', first=False, padding = 0):
        super(ConvD, self).__init__()

        self.first = first
        # if self.first==True:
        #     group = inplanes
        # else:
        #     group = 1
        group = 1
        self.maxpool = nn.MaxPool3d(2, 2,padding = padding)

        self.dropout = dropout

        self.relu = nn.LeakyReLU(0.2,inplace=False)
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False,groups=group)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False,groups=group)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False,groups=group)
        self.bn3   = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.bn1(self.conv1(x))
        if self.dropout > 0:
            x = F.dropout3d(x, self.dropout)
        y = self.relu(self.bn2(self.conv2(x)))
        y = self.bn3(self.conv3(x))
        return self.relu(x + y)


class ConvU(nn.Module):
    def __init__(self, planes,dropout=0, norm='bn', first=False, padding = 0):
        super(ConvU, self).__init__()

        self.first = first

        if not self.first:
            self.conv1 = nn.Conv3d(2*planes, planes, 3, 1, 0, bias=False)
            self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes//2, planes//2, 3, 1, 1, bias=False)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv3d(planes, planes//2, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes//2, norm)

        self.upsampling = nn.ConvTranspose3d(planes, planes//2,
                                      kernel_size=2,
                                      stride=2,padding=padding)
        self.dropout = dropout
     
        self.relu = nn.LeakyReLU(0.2,inplace=False)  
    def forward(self, x, prev):
        # final output is the localization layer
        y = self.upsampling(x)
        if self.dropout > 0:
            x = F.dropout3d(x, self.dropout)
        y = self.relu(self.bn2(self.conv2(y)))
        y = torch.cat([prev, y], 1)
        y = self.relu(self.bn3(self.conv3(y)))

        return y