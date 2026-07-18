#!/usr/bin/python
# -*- coding:utf-8 -*-

from models.Resnet1d import resnet18 as resnet_1d
from models.Resnet2d import resnet18 as resnet_2d

from models.MultiModal import MultiModal as MultiModal
from models.FinalClassifier import FinalClassifier as FinalClassifier
from models.DomainFeatureExtractor import DomainFeatureExtractor as DomainFeatureExtractor
