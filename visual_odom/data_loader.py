# -*- coding: utf-8 -*-

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy
import torch
import math
from transform import euler_matrix, euler_from_matrix

class kitti_dataset_VlocNet(Dataset):
    
    def __init__(self,kitti_path,sequence, tfm,sequence_length):
        self.path = kitti_path
        left_image_path = kitti_path +'sequences/'+sequence+'/image_2/'
        self.left_image = sorted([os.path.join(left_image_path,x) for x in os.listdir(left_image_path) if x.endswith(".png")])
        self.odometry = []
        self.odom_last = None
        f=open(kitti_path + 'poses/' + sequence + '.txt', 'r' )
        for line in f.read().splitlines():
            odom = line.split(' ')
            self.odometry.append(odom)
        
        self.transform = tfm
        self.seq_len = sequence_length
    
    def stack_image(self,index):
        
        left_file = self.left_image[index]
        left_img = Image.open(left_file)
        left_img = self.transform(left_img)
        
        if (index == 0):
                left_img = torch.cat((left_img,left_img))
        else:
                last_left_img = Image.open(self.left_image[index-1])
                last_left_img = self.transform(last_left_img)
                left_img =  torch.cat((left_img,last_left_img),dim =0)
        return left_img
        
    

    def __getitem__(self,index):
    
        left_img_stack=self.stack_image(index)

        

        odom = numpy.array(self.odometry[index], dtype = numpy.float32).reshape((3,4))
        
        odom_rot,odom_trans = odom[:3,:3],odom[:,-1]
        
        if (index == 0):
            rot_relative = odom_rot
            trans_rel = numpy.linalg.inv(odom_rot).dot(odom_trans)
            
            last_pose =  numpy.concatenate((odom_trans,euler_from_matrix(odom_rot,axes = 'sxzy')), dtype = numpy.float32)
            
        else:
            odom_last = numpy.array(self.odometry[index-1], dtype = numpy.float32).reshape((3,4))
            #odom_last = numpy.vstack((odom_last,[0,0,0,1]))
            odom_rot_last,odom_trans_last = odom_last[:3,:3],odom_last[:,-1]
            last_pose =  numpy.concatenate((odom_trans_last,euler_from_matrix(odom_rot_last,axes = 'sxzy')), dtype = numpy.float32)
            
            rot_relative = odom_rot.dot(numpy.linalg.inv(odom_rot_last))
            trans_rel = numpy.linalg.inv(odom_rot).dot(odom_trans -odom_trans_last)
        
        euler = euler_from_matrix(rot_relative,axes = 'sxzy')
        pose_relative = numpy.concatenate((trans_rel,euler),dtype = numpy.float32)
        
        
        
        pose_global = numpy.concatenate((odom_trans,euler_from_matrix(odom_rot,axes = 'sxzy')), dtype = numpy.float32)

        return left_img_stack,last_pose,pose_relative, pose_global
    
    def __len__(self):
        return len(self.left_image)


class kitti_dataset_vstack(Dataset):
    
    def __init__(self,kitti_path,sequence, tfm,sequence_length):
        self.path = kitti_path
        left_image_path = kitti_path +'sequences/'+sequence+'/image_2/'
        self.left_image = sorted([os.path.join(left_image_path,x) for x in os.listdir(left_image_path) if x.endswith(".png")])
        self.odometry = []
        self.odom_last = None
        f=open(kitti_path + 'poses/' + sequence + '.txt', 'r' )
        for line in f.read().splitlines():
            odom = line.split(' ')
            self.odometry.append(odom)
        
        self.transform = tfm
        self.seq_len = sequence_length
    
    def stack_image(self,index):
        
        left_file = self.left_image[index]
        left_img = Image.open(left_file)
        left_img = self.transform(left_img)
        
        if (index == 0):
                left_img = torch.cat((left_img,left_img))
        else:
                last_left_img = Image.open(self.left_image[index-1])
                last_left_img = self.transform(last_left_img)
                left_img =  torch.cat((left_img,last_left_img),dim =0)
        left_img = torch.unsqueeze(left_img, dim=0)
        return left_img
        
    

    def __getitem__(self,index):


        
        
        for i in range(self.seq_len):
            
            
            _left_img = self.stack_image(index-i)
            #last_left_img = self.transform(_left_img)
            #last_left_img = torch.unsqueeze(last_left_img,0)
            if i ==0:
                left_img_stack = _left_img
            else:
                left_img_stack =  torch.cat((_left_img,left_img_stack),dim =0)
        
       # left_img_stack=self.stack_image(index)

        

        odom = numpy.array(self.odometry[index], dtype = numpy.float32).reshape((3,4))
        
        odom_rot,odom_trans = odom[:3,:3],odom[:,-1]
        
        if (index == 0):
            rot_relative = odom_rot
            trans_rel = numpy.linalg.inv(odom_rot).dot(odom_trans)
        else:
            odom_last = numpy.array(self.odometry[index-1], dtype = numpy.float32).reshape((3,4))
            #odom_last = numpy.vstack((odom_last,[0,0,0,1]))
            odom_rot_last,odom_trans_last = odom_last[:3,:3],odom_last[:,-1]
            rot_relative = odom_rot.dot(numpy.linalg.inv(odom_rot_last))
            trans_rel = numpy.linalg.inv(odom_rot).dot(odom_trans -odom_trans_last)
            
        #translate_vector = odom_relative[:-1,-1]
        #rotation_matrix = odom_relative[:3,:3]
        
            

        #trans_rel[0],trans_rel[1],trans_rel[2] = trans_rel[0],trans_rel[2],trans_rel[1]
        
        euler = euler_from_matrix(rot_relative,axes = 'sxzy')
        pose_relative = numpy.concatenate((trans_rel,euler),dtype = numpy.float32)
            

        return left_img_stack,pose_relative
    
    def __len__(self):
        return len(self.left_image)



if __name__ == '__main__':
    import torchvision.transforms as transforms
    from matplotlib import pyplot as plt
    from model import transform
    t= transform()
    t.get_tf()
    
    dst =  kitti_dataset_vstack(kitti_path ='./',sequence = '07',tfm =t.train_tf,sequence_length=6)
    '''
    for i in dst.odometry:
        i = numpy.array(i,dtype = numpy.float32)
        #plt.xlim([-300,400])
        plt.plot(i[3],i[11],marker = "o")
        
        #print('x=',i[7],'y=',i[11])
    '''
    #plt.show
    ang = numpy.identity(3,dtype = numpy.float32)
    trans = numpy.zeros(3,dtype = numpy.float32)
    for i in range(len(dst)):
    
        x1 = dst[i][1]
        ang =  (euler_matrix(x1[3],x1[4],x1[5],axes = 'sxzy')).dot(ang)
        trans =ang.dot(x1[:3])+trans
        print('sequence', i,',  x = ',trans,'true pose = ', dst[i][2])
        #print('sequence', i,',  rot = ',ang,'true rot = ', dst[i][3])
        #print('sequence', i,',  rot = ',euler_from_matrix(ang),'true rot = ', euler_from_matrix(dst[i][3]))

        
