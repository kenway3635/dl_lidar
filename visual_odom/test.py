#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:53:24 2022

@author: kangli0306
"""

from data_loader import kitti_dataset_vstack,kitti_dataset_VlocNet
from model import deep_vo, transform, VlocNet

from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import numpy as np

import argparse
import matplotlib.pyplot as plt
from transform import euler_from_matrix, euler_matrix,translation_matrix
from train import matplotlib_imshow


parser = argparse.ArgumentParser()

parser.add_argument("--dir",type = str,help = "kitti dataset directory",default = './')
parser.add_argument("--load_model",type= str, help = "model save directory",default = './1115_vlonet_model/epoch_185.ckpt')







if __name__ == '__main__':
    
    tf = transform() 
    tf.get_tf()
    args = parser.parse_args()
    
    test_seq = '10'

    test_dataset = kitti_dataset_VlocNet(args.dir, test_seq,tfm = tf.test_tf,sequence_length=6)
    test_loader=DataLoader(test_dataset, 1, shuffle = False, pin_memory = True)


    pose =np.zeros(3,dtype = np.float32)
    gd_true = np.zeros(4,dtype = np.float32)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"


    #model = deep_vo().to(device)
    model = VlocNet().to(device)

    pred_x = []
    pred_y = []
    gd_true_x = []
    gd_true_y = []
    #load pretrained model
    ang = np.identity(3,dtype = np.float32)
    ang_gnd = np.identity(3,dtype = np.float32)
    if args.load_model == False:
        print('error, could not find the model in directory')
    

    
    else:
        
        model.load_state_dict(torch.load(args.load_model))

        model.eval()
        #model.hidden = model.init_hidden_no_stack()
        test_pbar = tqdm(test_loader, position=0, leave=True)
        trans_gnd=np.zeros(3)
        trans = np.zeros(3)
        pred=[]
        gt = []
        with torch.no_grad():
            for i,batch in enumerate(tqdm(test_pbar)):
                
                left_image,last_pose, odom_rel,odom_global = batch
                
                print(i)
                
                if i ==0:
                    pred_rel, pred_global = model(left_image.to(device),last_pose.to(device))
                else:
                    pred_rel, pred_global = model(left_image.to(device),pred_global)
                
                
                #s ='GT : ' + f"{odom[0]}"+ ', PR : '+ f"{predicted[0]}"
                #matplotlib_imshow(left_image.cpu()[0],s)
                
                #s =' PR : '+ f"{predicted[0].cpu().detach().numpy()[0]}"
                #print(s)
                
                loss = model.get_loss(pred_rel,pred_global, odom_rel.to(device), odom_global.to(device))
                
                s ='loss : ' + f"{loss.item()}"
                print(s)
                #matplotlib_imshow(left_image.cpu()[0][-1],s)
                #print(predicted)
                
                x = pred_global[0].cpu().detach().numpy()
                #x=x.resize(6)
                y = odom_global[0].cpu().detach().numpy()
                #y=y.resize(6)
                
                
                #print('predicted = ',x)
                #ang =  (euler_matrix(x[3],x[4],x[5],axes = 'sxzy')).dot(ang)
                #trans =ang.dot(x[:3])+trans
                pred_x.append(x[0])
                pred_y.append(x[2])
    

                #ang_gnd =  (euler_matrix(y[3],y[4],y[5],axes = 'sxzy')).dot(ang_gnd)
                #trans_gnd =ang_gnd.dot(y[:3])+trans_gnd
                #gd_true =ang_gnd[:,3]
                
                gd_true_x.append(y[0])
                gd_true_y.append(y[2])
                

        
   

        plt.plot(pred_x,pred_y,'--g',label = 'deepVO')
        plt.plot(gd_true_x,gd_true_y,'-k',label = 'GroundTrue')
        #plt.xlim([-300,400])
        
        
        plt.show()

            
       
        
                

            
            