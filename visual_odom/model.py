#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:52:58 2022

@author: kangli0306
"""
import torch.nn as nn
import torch
import torchvision.transforms as transforms


class VlocNet(nn.Module):
    def __init__(self):
        
        super(VlocNet, self).__init__()
        
        self.conv1   = nn.Conv2d(6, 64, 7, 2, 3, bias = False)
        self.conv2   = nn.Conv2d(64, 128, 5, 2, 2, bias = False)
        self.conv3   = nn.Conv2d(128, 256, 5, 2, 2, bias = False)
        self.conv3_1 = nn.Conv2d(256, 256, 3, 1, 1, bias = False)
        self.conv4   = nn.Conv2d(256, 512, 3, 2, 1, bias = False)
        self.conv4_1 = nn.Conv2d(512, 512, 3, 1, 1, bias = False)
        self.conv5   = nn.Conv2d(512, 512, 3, 2, 1, bias = False)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1, bias = False)
        self.conv6   = nn.Conv2d(512, 1024, 3, 2, 1, bias = False)
        
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv4_bn = nn.BatchNorm2d(512)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv6_bn = nn.BatchNorm2d(1024)

        self.trans_rel = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,3),
            )
        self.rot_rel = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,3),
            )
        self.flat = nn.Linear(1024*6*20,1000)

        self.relu = nn.ReLU()
        
        self.trans_global = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,3),
            )
        self.rot_global = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,3),
            )
        self.concat = nn.Sequential(
            nn.Linear(12,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,6),
            )

        
        
    def forward(self, x,last_pose):
        

        x = (self.relu(self.conv1_bn(self.conv1(x))))
        x = (self.relu(self.conv2_bn(self.conv2(x))))
        x = (self.relu(self.conv3_bn(self.conv3(x))))
        x = (self.relu(self.conv3_1_bn(self.conv3_1(x))))
        x = (self.relu(self.conv4_bn(self.conv4(x))))
        x = (self.relu(self.conv4_1_bn(self.conv4_1(x))))
        x = (self.relu(self.conv5_bn(self.conv5(x))))
        x = (self.relu(self.conv5_1_bn(self.conv5_1(x))))
         
        x = self.conv6_bn(self.conv6(x)) # No relu at the last conv

        x = torch.flatten(x,start_dim=1)

        x=self.relu(x)
        x = self.flat(x)
        x=self.relu(x)
         
        trans_rel = self.trans_rel(x)
        rot_rel = self.rot_rel(x)
        out_rel = torch.cat((trans_rel,rot_rel),dim = 1)
        
        trans_global = self.trans_global(x)
        rot_global = self.rot_global(x)
        out_global = torch.cat((trans_global,rot_global,last_pose),dim = 1)
        out_global = self.concat(out_global)
        
    
        
        return out_rel, out_global

    def get_loss(self, pred_rel,pred_global , y_rel, y_global):
        ang_rel = torch.nn.functional.mse_loss(100*pred_rel[:,3:],100*y_rel[:,3:])
        trans_rel = torch.nn.functional.mse_loss(100*pred_rel[:,:3],100*y_rel[:,:3])
        
        ang_global = torch.nn.functional.mse_loss(10*pred_global[:,3:],10*y_global[:,3:])
        trans_global = torch.nn.functional.mse_loss(pred_global[:,:3],y_global[:,:3])
        loss = ang_rel +trans_rel + ang_global + trans_global
        return loss
    
    def load_weight(self,weights):
        
        self.conv1.weight.data = weights["conv1.0.weight"]
        self.conv1_bn.weight.data = weights["conv1.1.weight"]
        self.conv1_bn.bias.data = weights["conv1.1.bias"]
        self.conv1_bn.running_mean.data = weights["conv1.1.running_mean"]
        self.conv1_bn.running_var.data = weights["conv1.1.running_var"]

        self.conv2.weight.data = weights["conv2.0.weight"]
        self.conv2_bn.weight.data = weights["conv2.1.weight"]
        self.conv2_bn.bias.data = weights["conv2.1.bias"]
        self.conv2_bn.running_mean.data = weights["conv2.1.running_mean"]
        self.conv2_bn.running_var.data = weights["conv2.1.running_var"]

        self.conv3.weight.data = weights["conv3.0.weight"]
        self.conv3_bn.weight.data = weights["conv3.1.weight"]
        self.conv3_bn.bias.data = weights["conv3.1.bias"]
        self.conv3_bn.running_mean.data = weights["conv3.1.running_mean"]
        self.conv3_bn.running_var.data = weights["conv3.1.running_var"]

        self.conv3_1.weight.data = weights["conv3_1.0.weight"]
        self.conv3_1_bn.weight.data = weights["conv3_1.1.weight"]
        self.conv3_1_bn.bias.data = weights["conv3_1.1.bias"]
        self.conv3_1_bn.running_mean.data = weights["conv3_1.1.running_mean"]
        self.conv3_1_bn.running_var.data = weights["conv3_1.1.running_var"]

        self.conv4.weight.data = weights["conv4.0.weight"]
        self.conv4_bn.weight.data = weights["conv4.1.weight"]
        self.conv4_bn.bias.data = weights["conv4.1.bias"]
        self.conv4_bn.running_mean.data = weights["conv4.1.running_mean"]
        self.conv4_bn.running_var.data = weights["conv4.1.running_var"]

        self.conv4_1.weight.data = weights["conv4_1.0.weight"]
        self.conv4_1_bn.weight.data = weights["conv4_1.1.weight"]
        self.conv4_1_bn.bias.data = weights["conv4_1.1.bias"]
        self.conv4_1_bn.running_mean.data = weights["conv4_1.1.running_mean"]
        self.conv4_1_bn.running_var.data = weights["conv4_1.1.running_var"]

        self.conv5.weight.data = weights["conv5.0.weight"]
        self.conv5_bn.weight.data = weights["conv5.1.weight"]
        self.conv5_bn.bias.data = weights["conv5.1.bias"]
        self.conv5_bn.running_mean.data = weights["conv5.1.running_mean"]
        self.conv5_bn.running_var.data = weights["conv5.1.running_var"]

        self.conv5_1.weight.data = weights["conv5_1.0.weight"]
        self.conv5_1_bn.weight.data = weights["conv5_1.1.weight"]
        self.conv5_1_bn.bias.data = weights["conv5_1.1.bias"]
        self.conv5_1_bn.running_mean.data = weights["conv5_1.1.running_mean"]
        self.conv5_1_bn.running_var.data = weights["conv5_1.1.running_var"]

        self.conv6.weight.data = weights["conv6.0.weight"]
        self.conv6_bn.weight.data = weights["conv6.1.weight"]
        self.conv6_bn.bias.data = weights["conv6.1.bias"]
        self.conv6_bn.running_mean.data = weights["conv6.1.running_mean"]
        self.conv6_bn.running_var.data = weights["conv6.1.running_var"]


        
class deep_vo(nn.Module):
    def __init__(self):
        
        super(deep_vo, self).__init__()
        
        self.conv1   = nn.Conv2d(6, 64, 7, 2, 3, bias = False)
        self.conv2   = nn.Conv2d(64, 128, 5, 2, 2, bias = False)
        self.conv3   = nn.Conv2d(128, 256, 5, 2, 2, bias = False)
        self.conv3_1 = nn.Conv2d(256, 256, 3, 1, 1, bias = False)
        self.conv4   = nn.Conv2d(256, 512, 3, 2, 1, bias = False)
        self.conv4_1 = nn.Conv2d(512, 512, 3, 1, 1, bias = False)
        self.conv5   = nn.Conv2d(512, 512, 3, 2, 1, bias = False)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1, bias = False)
        self.conv6   = nn.Conv2d(512, 1024, 3, 2, 1, bias = False)
        
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv4_bn = nn.BatchNorm2d(512)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv6_bn = nn.BatchNorm2d(1024)

        self.trans = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256,3),
            )
        self.rot = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256,3),
            )
        self.flat = nn.Linear(1024*6*20,1000)
        self.lstm = nn.LSTM(input_size=1000, hidden_size=1000, num_layers=2,batch_first=True)
        self.relu = nn.ReLU()
        self.rnn_dropout = nn.Dropout(0.2)
        self.fc_nn = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            )
        
    def init_hidden(self,batch_size):
        h = ((torch.zeros(2,batch_size,1000,device = 'cuda').requires_grad_()),
               (torch.zeros(2,batch_size,1000,device = 'cuda').requires_grad_()))
        return h
    def init_hidden_no_stack(self):
        h = ((torch.zeros(2,1000,device = 'cuda').requires_grad_()),
               (torch.zeros(2,1000,device = 'cuda').requires_grad_()))
        return h
        
    def forward(self, x_3d):
        self.hidden = self.init_hidden(x_3d.shape[0])
        
        #print(x_3d.shape)
        for i in range(x_3d.shape[1]):
            #self.hidden = tuple(state.to('cuda') for state in self.hidden)
            x=x_3d[:,i,:,:,:]
            x = (self.relu(self.conv1_bn(self.conv1(x))))
            x = (self.relu(self.conv2_bn(self.conv2(x))))
            x = (self.relu(self.conv3_bn(self.conv3(x))))
            x = (self.relu(self.conv3_1_bn(self.conv3_1(x))))
            x = (self.relu(self.conv4_bn(self.conv4(x))))
            x = (self.relu(self.conv4_1_bn(self.conv4_1(x))))
            x = (self.relu(self.conv5_bn(self.conv5(x))))
            x = (self.relu(self.conv5_1_bn(self.conv5_1(x))))
            
            x = self.conv6_bn(self.conv6(x)) # No relu at the last conv
           # print(out.shape)
            x = torch.flatten(x,start_dim=1)
           # print(out.shape)
            x = self.flat(x)
            x = torch.unsqueeze(x,1)

            if i ==0:
                out_3d = x
            else:
                out_3d = torch.cat((out_3d,x),dim =1)
        '''
        x = (self.relu(self.conv1_bn(self.conv1(x))))
        x = (self.relu(self.conv2_bn(self.conv2(x))))
        x = (self.relu(self.conv3_bn(self.conv3(x))))
        x = (self.relu(self.conv3_1_bn(self.conv3_1(x))))
        x = (self.relu(self.conv4_bn(self.conv4(x))))
        x = (self.relu(self.conv4_1_bn(self.conv4_1(x))))
        x = (self.relu(self.conv5_bn(self.conv5(x))))
        x = (self.relu(self.conv5_1_bn(self.conv5_1(x))))
         
        x = self.conv6_bn(self.conv6(x)) # No relu at the last conv
        # print(out.shape)
        x = torch.flatten(x,start_dim=1)
        # print(out.shape)
        x=self.relu(x)
        x = self.flat(x)
        x=self.relu(x)
         
        '''

        #print(out_3d.shape)
        x,self.hidden = self.lstm(out_3d,self.hidden)
        

        
        #print(f"hidden = {hid[0].cpu().detach().numpy()[:,-1,:]}")
        #print(f"output size = {out.shape}")
        
        x =x[:,-1,:]
        #x =self.rnn_dropout(x)
        #x = self.fc_nn(x)
        #out = self.fc(out)
        trans = self.trans(x)
        rot = self.rot(x)
        out = torch.cat((trans,rot),dim = 1)
        
        self.hidden = tuple(state.detach() for state in self.hidden)  #detach for backward propergation

        return out

    def get_loss(self, pred, y):
        angle_loss = torch.nn.functional.mse_loss(pred[:,3:],y[:,3:])
        translation_loss = torch.nn.functional.mse_loss(pred[:,:3],y[:,:3])
        loss = 100*angle_loss +100*translation_loss
        return loss
    
    def load_weight(self,weights):
        
        self.conv1.weight.data = weights["conv1.0.weight"]
        self.conv1_bn.weight.data = weights["conv1.1.weight"]
        self.conv1_bn.bias.data = weights["conv1.1.bias"]
        self.conv1_bn.running_mean.data = weights["conv1.1.running_mean"]
        self.conv1_bn.running_var.data = weights["conv1.1.running_var"]

        self.conv2.weight.data = weights["conv2.0.weight"]
        self.conv2_bn.weight.data = weights["conv2.1.weight"]
        self.conv2_bn.bias.data = weights["conv2.1.bias"]
        self.conv2_bn.running_mean.data = weights["conv2.1.running_mean"]
        self.conv2_bn.running_var.data = weights["conv2.1.running_var"]

        self.conv3.weight.data = weights["conv3.0.weight"]
        self.conv3_bn.weight.data = weights["conv3.1.weight"]
        self.conv3_bn.bias.data = weights["conv3.1.bias"]
        self.conv3_bn.running_mean.data = weights["conv3.1.running_mean"]
        self.conv3_bn.running_var.data = weights["conv3.1.running_var"]

        self.conv3_1.weight.data = weights["conv3_1.0.weight"]
        self.conv3_1_bn.weight.data = weights["conv3_1.1.weight"]
        self.conv3_1_bn.bias.data = weights["conv3_1.1.bias"]
        self.conv3_1_bn.running_mean.data = weights["conv3_1.1.running_mean"]
        self.conv3_1_bn.running_var.data = weights["conv3_1.1.running_var"]

        self.conv4.weight.data = weights["conv4.0.weight"]
        self.conv4_bn.weight.data = weights["conv4.1.weight"]
        self.conv4_bn.bias.data = weights["conv4.1.bias"]
        self.conv4_bn.running_mean.data = weights["conv4.1.running_mean"]
        self.conv4_bn.running_var.data = weights["conv4.1.running_var"]

        self.conv4_1.weight.data = weights["conv4_1.0.weight"]
        self.conv4_1_bn.weight.data = weights["conv4_1.1.weight"]
        self.conv4_1_bn.bias.data = weights["conv4_1.1.bias"]
        self.conv4_1_bn.running_mean.data = weights["conv4_1.1.running_mean"]
        self.conv4_1_bn.running_var.data = weights["conv4_1.1.running_var"]

        self.conv5.weight.data = weights["conv5.0.weight"]
        self.conv5_bn.weight.data = weights["conv5.1.weight"]
        self.conv5_bn.bias.data = weights["conv5.1.bias"]
        self.conv5_bn.running_mean.data = weights["conv5.1.running_mean"]
        self.conv5_bn.running_var.data = weights["conv5.1.running_var"]

        self.conv5_1.weight.data = weights["conv5_1.0.weight"]
        self.conv5_1_bn.weight.data = weights["conv5_1.1.weight"]
        self.conv5_1_bn.bias.data = weights["conv5_1.1.bias"]
        self.conv5_1_bn.running_mean.data = weights["conv5_1.1.running_mean"]
        self.conv5_1_bn.running_var.data = weights["conv5_1.1.running_var"]

        self.conv6.weight.data = weights["conv6.0.weight"]
        self.conv6_bn.weight.data = weights["conv6.1.weight"]
        self.conv6_bn.bias.data = weights["conv6.1.bias"]
        self.conv6_bn.running_mean.data = weights["conv6.1.running_mean"]
        self.conv6_bn.running_var.data = weights["conv6.1.running_var"]

    

class transform():
    def __init__(self):
        self.test_tf = None
        self.train_tf = None
    
    def get_tf(self):
        #add image transformation for training
        
        self.train_tf = transforms.Compose([
            transforms.ToTensor(),
            
            transforms.Normalize(
            mean=(0.5,)*3,
            std = (1,)*3,)
            
            ]
            )
        
        
        #self.train_tf = transforms.ToTensor()
        
        #add image transformation for validating

        
        self.test_tf = transforms.Compose([
            transforms.ToTensor(),
            
            transforms.Normalize(
            mean=(0.5,)*3,
            std = (1,)*3,)
            ]
            
            )
        