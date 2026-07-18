# -*- coding: utf-8 -*-
"""
@author: P Xia
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Resnet1d import resnet18 as resnet_1d
from models.Resnet2d import resnet18 as resnet_2d

class MultiModal(nn.Module):
    def __init__(self):
        super(MultiModal, self).__init__()
        self.vib_enc = resnet_2d(in_channel=3)
        self.cur_enc = resnet_1d(in_channel=3)
        self.aud_enc = resnet_2d(in_channel=6)
        self.dim = 256
        
        self.vib_trans = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.dim,self.dim),
            nn.ReLU()
            )
        self.cur_trans = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.dim,self.dim),
            nn.ReLU()
            )
        self.aud_trans = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.dim,self.dim),
            nn.ReLU()
            )
    
    def forward(self, vibration, current, audio):
        vib_codes = self.vib_enc(vibration)
        cur_codes = self.cur_enc(current)
        aud_codes = self.aud_enc(audio)
        
        vib_features = self.vib_trans(vib_codes)
        cur_features = self.cur_trans(cur_codes)
        aud_features = self.aud_trans(aud_codes)
        
        return vib_features, cur_features, aud_features
