#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:22:37 2023

@author: lab
"""
import numpy as np
import pcl.pcl_visualization
import os


points = np.fromfile(path,dtype=np.float32).reshape(-1,4)
points[:,3]=255

#color_cloud = pcl.PointCloud_PointXYZRGB(points)
color_cloud = pcl.PointCloud_PointXYZRGB(points)
visual=pcl.pcl_visualization.CloudViewing()
visual.ShowColorCloud(color_cloud,b'cloud')
flag=True
while flag:
    flag != visual.WasStopped()
    
