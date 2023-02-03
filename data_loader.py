# -*- coding: utf-8 -*-

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import pcl.pcl_visualization
#from pclpy import pcl
import open3d as o3d
import time
import pcl



class kitti_dataset(Dataset):
    
    def __init__(self,kitti_path,sequences):
        self.path = kitti_path
        self.pointclouds =[]
        for sequence in sequences:
            left_image_path = kitti_path +'sequences/'+sequence+'/image_2/'
            lidar_path = kitti_path+'sequences/'+sequence+'/velodyne/'
            #self.left_image = sorted([os.path.join(left_image_path,x) for x in os.listdir(left_image_path) if x.endswith(".png")])
            self.pointclouds += (sorted([os.path.join(lidar_path,x) for x in os.listdir(lidar_path) if x.endswith(".bin")]))
        self.odometry = []

        '''
        f=open(kitti_path + 'poses/' + sequence + '.txt', 'r' )
        
        for line in f.read().splitlines():
            odom = line.split(' ')
            self.odometry.append(odom)
        '''
    
    def image(self,index):
        
        left_file = self.left_image[index]
        left_img = Image.open(left_file)
        return left_img
    
    def points(self,index):
        
        points = np.fromfile(self.pointclouds[index],dtype=np.float32).reshape(-1,4)
        
        cloud = pcl.PointCloud_PointXYZRGB(points)
        cloud1=pcl.PointCloud_PointXYZRGB.make_voxel_grid_filter(cloud)
        pcl.VoxelGridFilter_PointXYZRGB.set_leaf_size(cloud1, 0.4, 0.4, 0.4)
        cloud2=pcl.VoxelGridFilter_PointXYZRGB.filter(cloud1)

        arr = cloud2.to_array()
        
        arr[:,3]=255
        return arr
    

    def __getitem__(self,index):
        _points = self.points(index)
        
        return _points
    
    def __len__(self):
        return len(self.pointclouds)
    
    def get_num_scans(self):
        return len(self.pointclouds)        

def visualize_pcd(visual,arr,color):
    
    #visual=pcl.pcl_visualization.CloudViewing()
    arr = arr.detach().cpu().numpy()
    #arr = arr.expand_dims(axis = 1)
    arr = np.concatenate((arr,np.ones((1,arr.shape[1])))).astype(np.float32)
    
    arr = arr.transpose(1,0)
    arr[:,3] = color

    
    #pcd = pcl.PointCloud_PointXYZI().from_array(arr)
    color_cloud = pcl.PointCloud_PointXYZRGB(arr)
    return color_cloud
    #visual.ShowColorCloud(color_cloud,b'cloud')
    
    

if __name__ == '__main__':
    
    visual=pcl.pcl_visualization.CloudViewing()
    seq = ['00','01','02']
    dst =  kitti_dataset(kitti_path ='./dataset/',sequences = seq)
    pcd = pcl.PointCloud_PointXYZRGB(dst[0])

    points = np.fromfile('./000000.bin',dtype=np.float32).reshape(-1,4)
    points[:,3] = 255
    cloud = pcl.PointCloud_PointXYZRGB(points)
    #visual.ShowColorCloud(cloud,b'cloud')
    
    cloud1=pcl.PointCloud_PointXYZRGB.make_voxel_grid_filter(cloud)
    pcl.VoxelGridFilter_PointXYZRGB.set_leaf_size(cloud1, 0.4, 0.4, 0.4)
    cloud2=pcl.VoxelGridFilter_PointXYZRGB.filter(cloud1)
    visual.ShowColorCloud(cloud2,b'cloud')
    arr = cloud2.to_array()
    
    '''
    for i in range(dst.get_num_scans()):
        print(i)
        color_cloud = pcl.PointCloud_PointXYZRGB(dst[i])
        visual.ShowColorCloud(color_cloud,b'cloud')
        time.sleep(0.1)
    '''
        
        

        
