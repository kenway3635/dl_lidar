#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:28:11 2023

@author: lab
"""
from model import autoencoder
from data_loader import kitti_dataset
from train import loader
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as chd
import ChamferDistancePytorch.fscore as fscore

import torch
from tqdm.auto import tqdm

import numpy as np
from statistics import mean, stdev
from torch.utils.data import DataLoader
import os


from data_loader import visualize_pcd
import pcl.pcl_visualization 

def cal_score(scores):
    mu = mean(scores)
    sigma = stdev(score)
    return mu,sigma


if __name__ == '__main__':
    test_loader = loader(path ='./dataset/',seq = ['21'], batch_size= 10,shuffle = True)
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model = autoencoder.to(device)
    
    visual = pcl.pcl_visualization.CloudViewing()
    scores = []
    test_pbar = tqdm(test_loader,position=0,leave=True)
    model.eval()
    for data in test_pbar:

        data = data.to(device)
        pred = model(data)

        cloud = visualize_pcd(visual,data[0],225)
        visual.ShowColorCloud(cloud,b'gt')
        pred_cloud = visualize_pcd(visual,pred[0],3329330)
        visual.ShowColorCloud(pred_cloud,b'pred')

        x = data.transpose(2,1)
        pred = pred.transpose(2,1)
        
        chamLoss = chd.chamfer_3DDist()
        dist1, dist2, idx1, idx2 = chamLoss(pred, x)
        f_score, precision, recall, dist = fscore.fscore(dist1, dist2)
        scores.append(f_score.cpu().numpy().tolist())
    
    mu,sigma = cal_score(scores)
    print('test result is :' ,mu,'accuracy with ',sigma,'standard devarition')
