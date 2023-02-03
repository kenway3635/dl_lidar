#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:53:24 2022

@author: kangli0306
"""

from data_loader import kitti_dataset_vstack,kitti_dataset_VlocNet
from model import deep_vo, transform,VlocNet

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm.auto import tqdm
import os

import argparse
from torch.utils.tensorboard import SummaryWriter 

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--dir",type = str,help = "kitti dataset directory",default = './')
parser.add_argument("--batch_size",type = int, help = "training batch size",default= 1)
parser.add_argument("--epoch", type = int, help = "training epoch number", default = 1)
parser.add_argument("--logdir",type = str,help = "training log directory", default = "./logs")
parser.add_argument("--savedir",type= str, help = "model save directory",default = "./model")
parser.add_argument("--flownet",type = str,help = "load pretrained flownet model",default = 'flownets_bn_EPE2.459.pth.tar')
parser.add_argument("--load_from",type = str, help = "load from last checkpoints",default = None)


def matplotlib_imshow(img,value, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    left = npimg[3:,:,:]
    #print(left.shape)
    
    right = npimg[:3,:,:]
    #print(right.shape)
    npimg = np.hstack((left,right))
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow((np.transpose(npimg, (1, 2, 0))*255).astype(np.uint8))
        plt.text(0, 0, value, bbox=dict(facecolor='red', alpha=0.5))
    plt.show()



    

if __name__ == '__main__':
    
    tf = transform() 
    tf.get_tf()
    args = parser.parse_args()
    
    train_seq = ['00','01','02','03','04','05','06']
    vali_seq = ['07','08','09']
    #train_seq = ['07']
    #vali_seq = ['07']
    
    train_loaders = []
    valid_loaders=[]
    for i in range(len(train_seq)):
        train_dataset = kitti_dataset_VlocNet(args.dir,train_seq[i], tfm = tf.train_tf,sequence_length=6)
        train_loaders.append(DataLoader(train_dataset, args.batch_size, shuffle = False, pin_memory = True))
    
    for i in range(len(vali_seq)):
        valid_dataset = kitti_dataset_VlocNet(args.dir, vali_seq[i],tfm = tf.test_tf,sequence_length=6)
        valid_loaders.append(DataLoader(valid_dataset, args.batch_size, shuffle = False, pin_memory = True))    
    
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    writer = SummaryWriter(args.logdir)

    
 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    n_epochs = args.epoch
    #model = deep_vo().to(device)
    model = VlocNet().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,factor = 0.1,patience = 30,min_lr = 1e-6)
    
    #load pretrained model
    if args.flownet:
        state_dict = torch.load(args.flownet)
        '''
        for k ,v in state_dict['state_dict'].items():
            print(k)
        '''
        model.load_weight(state_dict['state_dict'])
    
    if args.load_from:
        model.load_state_dict(torch.load(args.load_from))
        


    best_loss = 999999
    #torch.autograd.set_detect_anomaly(True)
    step = 0
    #model.hidden = model.init_hidden_no_stack()
    for epoch in range(n_epochs):
        model.train()
        
        train_losses = []
        valid_losess = []
        for i in range(len(train_seq)):
            train_loader = train_loaders[i]
            train_pbar = tqdm(train_loader, position=0, leave=True)
            
            for j,batch in enumerate(tqdm(train_pbar)):
                optimizer.zero_grad()
                
                left_image,last_pose, odom_rel,odom_global = batch
                

                #pred_rel, pred_global = model(left_image.to(device),last_pose.to(device))

                if j ==0:
                    pred_rel, pred_global = model(left_image.to(device),last_pose.to(device))
                else:
                    pred_rel, pred_global = model(left_image.to(device),pred_global.detach())
                #s ='GT : ' + f"{odom[0]}"+ ', PR : '+ f"{predicted[0]}"
                #matplotlib_imshow(left_image.cpu()[0],s)
                
                #s =' PR : '+ f"{predicted[0].cpu().detach().numpy()[0]}"
                #print(s)
                
                loss = model.get_loss(pred_rel,pred_global, odom_rel.to(device), odom_global.to(device))
                
                
                loss.backward()
                
                #clip the gradient norm for stable training
                #grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                
                train_losses.append(loss.item())
                
                clipping_value = 0.25 # arbitrary value of your choosing
                torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)

                
                optimizer.step()
                '''
                for name, param in model.named_parameters():
                    print(name, param.grad)
                '''
                
                step +=1
                train_pbar.set_description(f"[train | {epoch + 1:03d}/{n_epochs:03d}],learning rate = {optimizer.param_groups[0]['lr']}, loss = {loss.detach().item():.5f} ")

                
            train_loss = sum(train_losses)/len(train_losses)
            writer.add_scalar('Loss/train',train_loss,epoch)
            
        

        
        model.eval()

        with torch.no_grad():
            #model.hidden = model.init_hidden_no_stack()
            for i in range(len(vali_seq)):
                vali_pbar = tqdm(valid_loaders[i], position=0, leave=True)
                for j,batch in enumerate(tqdm(vali_pbar)):
                    left_image,last_pose, odom_rel,odom_global = batch
                    
                   
                    #pred_rel, pred_global = model(left_image.to(device),last_pose.to(device))
                    
                    if j ==0:
                        pred_rel, pred_global = model(left_image.to(device),last_pose.to(device))
                    else:
                        pred_rel, pred_global = model(left_image.to(device),pred_global.detach())
                    
                    
                    #s ='GT : ' + f"{odom[0]}"+ ', PR : '+ f"{predicted[0]}"
                    #matplotlib_imshow(left_image.cpu()[0],s)
                    
                    #s =' PR : '+ f"{predicted[0].cpu().detach().numpy()[0]}"
                    #print(s)
                    
                    loss = model.get_loss(pred_rel,pred_global, odom_rel.to(device), odom_global.to(device))
                    
                    #s ='GT : ' + f"{odom[0]}"+ ', PR : '+ f"{predicted[0]}"
                    
                    #matplotlib_imshow(left_image.cpu()[0],s)
                    
                    #s =' validate PR : '+ f"{predicted[0].cpu().detach().numpy()[0]}"
                    #print(s)
                    
                    valid_losess.append(loss.item())
                    vali_pbar.set_description(f"[validate | {epoch + 1:03d}/{n_epochs:03d}], loss = {loss.detach().item():.5f} ")
            valid_loss = sum(valid_losess)/len(valid_losess)
            scheduler.step(valid_loss)
            writer.add_scalar('Loss/valid',valid_loss,epoch)
        
        if (valid_loss<=best_loss):
            best_loss = valid_loss
            if not os.path.exists(args.savedir):
                os.makedirs(args.savedir)
            print('saving model...')
            torch.save(model.state_dict(), f"{args.savedir}/epoch_{epoch+1}.ckpt")
        elif (epoch%100 == 0):
            print('saving model for checkpoint...')
            torch.save(model.state_dict(), f"{args.savedir}/epoch_{epoch}.ckpt")
            
            