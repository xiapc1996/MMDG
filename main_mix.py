# -*- coding: utf-8 -*-
"""
@author: P Xia
"""
# coding=utf-8
import numpy as np
import pandas as pd
import os
import argparse
import torch
import models
from train_mix import init_seed, init_dataset, train_mix, test_mix
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, default='MultiModal', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='MOTOR_MultiSensor', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='./Motor_processed_MMdata', help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')
    parser.add_argument('--num_samples', type=int, default=500, help='sample number of each category')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='validation set ratio')

    parser.add_argument('--source_condition', type=list, default=[0,2,6], help='source domain data working conditions')
    parser.add_argument('--target_condition', type=list, default=[8], help='target domain data working conditions')
    
    parser.add_argument('-root', '--dataset_root', type=str, help='path to dataset', default='.' + os.sep + 'dataset')
    parser.add_argument('-exp', '--experiment_root', type=str, help='root where to store models, losses and accuracies', default='.' + os.sep + 'output')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--main_dir', type=str, help='root for task', default='.'+os.sep+'output_baseline_test')

    parser.add_argument('-nep', '--epochs', type=int, help='number of epochs to train for', default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate for the model, default=0.001', default=0.001)
    parser.add_argument('-lrS', '--lr_scheduler_step', type=int, help='StepLR learning rate scheduler step, default=20', default=200)
    parser.add_argument('-lrG', '--lr_scheduler_gamma', type=float, help='StepLR learning rate scheduler gamma, default=0.5', default=0.1)
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)

    parser.add_argument('--lambda_mmd', type=float, default=0.5, help='weight for similarity loss')
    parser.add_argument('--lambda_cov', type=float, default=0.5, help='weight for difference loss')
    parser.add_argument('--lambda_domain_mmd', type=float, default=1.0, help='weight for similarity loss')
    parser.add_argument('--lambda_domain_cov', type=float, default=1.0, help='weight for difference loss')
    parser.add_argument('--val_every', type=int, default=5, help="number of val every steps")

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=42)

    return parser.parse_args()
    

def main(opt):
    '''
    Initialize everything and train
    '''
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    dst_src = init_dataset(opt, 'src', opt.data_dir, opt.num_samples, opt.val_ratio)
    dst_tar = init_dataset(opt, 'tar', opt.data_dir, opt.num_samples, opt.val_ratio)

    device = 'cuda:0' if torch.cuda.is_available() and options.cuda else 'cpu'
    model = getattr(models, options.model_name)()
    model = model.to(device)
    # print(model)
    
    fused_classifier = models.FinalClassifier(dim=256, num_classes=8).to(device)
    domain_extractor = models.DomainFeatureExtractor(
        input_dim=model.dim,  # Feature dimension of each modality
        output_dim=model.dim  # Output feature dimension
    ).to(device)
    print('Using ', device)
    param_list = [{"params": model.parameters(), "lr": options.learning_rate},
                  {"params": fused_classifier.parameters(), "lr": options.learning_rate},
                  {"params": domain_extractor.parameters(), "lr": options.learning_rate}]
    optim = torch.optim.Adam(params=param_list, lr=options.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=options.lr_scheduler_gamma,
                                           step_size=options.lr_scheduler_step)
    train_mix(opt=opt,
            src_dataset=dst_src,
            model=model,
            classifier=fused_classifier,
            domain_extractor=domain_extractor,
            optim=optim,
            lr_scheduler=lr_scheduler)
    # best_state, best_acc, train_loss, train_acc, val_loss, val_acc, val_matrix = res
    print('Testing with last model..')
    test_acc = test_mix(opt=opt,
                    tar_dataset=dst_tar[0],
                    model=model,
                    classifier = fused_classifier,
                    domain_extractor=domain_extractor)

    return test_acc

if __name__ == '__main__':
    options = parse_args()

    tasks = [0, 1, 2, 3]
    test_task = 0
    options.target_condition = [test_task]
    options.source_condition = [x for x in tasks if x!=test_task]

    test_mat_ls = []
    test_acc_ls = []

    options.manual_seed = 2024
    init_seed(options)
    
    res = main(options)
    test_acc, test_matrix = res
    test_mat_ls.append(test_matrix)
    test_acc_ls.append(test_acc)