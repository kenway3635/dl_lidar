#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:09:03 2023

@author: lab
"""
from model import autoencoder
from data_loader import kitti_dataset

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from torch.utils.tensorboard import SummaryWriter

from data_loader import visualize_pcd
import pcl.pcl_visualization 

logdir = './logs'
if not os.path.exists(logdir):
    os.makedirs(logdir)
writer = SummaryWriter(logdir)

# a simple custom collate function, just to show the idea
def my_collate(batch):

    '''
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]
    '''
    data = []
    for item in batch:
        item_tensor = torch.tensor(item,dtype = torch.float32)
        item_tensor = item_tensor[:,:-1]
        data.append(item_tensor)
    data = pad_sequence(data, batch_first = True, padding_value = 0)
    data = data.permute(0,2,1)
    
    return data.to("cuda")


def loader(path,seq,batch_size,shuffle):
    data =  kitti_dataset(kitti_path =path,sequences = seq)
    _loader = DataLoader(data,batch_size,collate_fn=my_collate,shuffle = True,pin_memory = False)
    return _loader
    
def trainer(train_loader,vali_loader,batch_size):
    
    savedir = './model'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = autoencoder().to(device)
    num_epoch = 1000
    learning_rate = 1e-3
    best_loss = 9999999
    optimizer = torch.optim.Adam(model.parameters(), lr =learning_rate, weight_decay = 1e-5) 
    visual=pcl.pcl_visualization.CloudViewing()
    
    for epoch in range(num_epoch):
        
        model.train()
        
        train_losses = []
        valid_losses = []
        
        train_pbar = tqdm(train_loader,position = 0,leave = True)
        for data in train_pbar:
            optimizer.zero_grad()
            data = data.to(device)
            
            pred = model(data)
            
            
            cloud = visualize_pcd(visual,data[0],225)
            visual.ShowColorCloud(cloud,b'gt')
            pred_cloud = visualize_pcd(visual,pred[0],3329330)
            visual.ShowColorCloud(pred_cloud,b'pred')
            
            
            loss = model.get_loss(pred, data)
            train_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            
            train_pbar.set_description(f"[train | {epoch + 1:03d}/{num_epoch:03d}],learning rate = {optimizer.param_groups[0]['lr']}, loss = {loss.detach().item():.5f} ")
        train_loss = sum(train_losses)/len(train_losses)
        writer.add_scalar('Loss/train',train_loss,epoch)
        
        model.eval()
        
        with torch.no_grad():
            valid_pbar = tqdm(vali_loader,position = 0,leave = True)
            for data in valid_pbar:
                data = data.to(device)
                pred = model(data)
                
                
                cloud = visualize_pcd(visual,data[0],225)
                visual.ShowColorCloud(cloud,b'gt')
                pred_cloud = visualize_pcd(visual,pred[0],3329330)
                visual.ShowColorCloud(pred_cloud,b'pred')
                
            
                loss = model.get_loss(pred, data)
                valid_losses.append(loss.item())
                valid_pbar.set_description(f"[validate | {epoch + 1:03d}/{num_epoch:03d}], loss = {loss.detach().item():.5f} ")
        valid_loss = sum(valid_losses)/len(valid_losses)
        writer.add_scalar('Loss/valid',valid_loss,epoch)
        if (valid_loss<=best_loss):
            best_loss = valid_loss
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            print('saving model...')
            torch.save(model.state_dict(), f"{savedir}/epoch_{epoch+1}.ckpt")
        elif (epoch%100 == 0):
            print('saving model for checkpoint...')
            torch.save(model.state_dict(), f"{savedir}/epoch_{epoch}.ckpt")
            
if __name__ == '__main__':
    
    b_size = 30
    #sim_train_data = torch.randint(size = (50,3,2500),low = 0,high = 20,dtype = torch.float)
    
    train_loader = loader(path ='./dataset/',seq = ['00','02','04','06','08'], batch_size= b_size,shuffle = True)
    #train_loader = DataLoader(sim_train_data,10,shuffle = True,pin_memory = True )
    valid_loader = loader(path ='./dataset/',seq = ['01','03','05','07','09'], batch_size= b_size,shuffle = True)
    
    
    train_features = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")

    trainer(train_loader,valid_loader,b_size)
        
    