#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
import os
#from PIL import Image
#from torchvision import transforms
import numpy as np
from datasets.sequence_aug import *

class dataset(Dataset):

    def __init__(self, list_data, test=False, transform=None):
        self.test = test
        # 适配多模态结构
        self.seq_data = list_data[['vibration', 'current', 'audio']].to_dict('records')
        self.labels = list_data['label'].tolist()
        # transform 应为 {'sensor':..., 'audio':...}
        if transform is None:
            self.transforms = {
                'vibration': Compose([Reshape()]),
                'current': Compose([Reshape()]),
                'audio': Compose([Reshape()]),
            }
        else:
            self.transforms = transform

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        seq = self.seq_data[item]
        label = self.labels[item]
        # 分别对 sensor/audio 做变换
        vibration = seq['vibration']
        current = seq['current']
        audio = seq['audio']
        if self.transforms.get('vibration'):
            vibration = self.transforms['vibration'](vibration)
        if self.transforms.get('current'):
            current = self.transforms['current'](current)
        if self.transforms.get('audio'):
            audio = self.transforms['audio'](audio)
        sample = {'vibration': vibration, 'current': current, 'audio': audio}
        return sample, label

