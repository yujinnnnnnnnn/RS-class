# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:56:43 2022

@author: USER
"""
# import libraries
import numpy as np
import os
import rasterio
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt

f_name="C:/Users/USER/Desktop/yeahje.jpg"
img=rasterio.open(f_name)
I=img.read()
I=np.swapaxes(I,0,2)
new_rgb=I.reshape(920*920,3)
k=10

data=pd.read_csv("C:/Users/USER/Desktop/train_set.csv")
data_array=data.to_numpy()
river=data_array[:794,:2].astype(int)
urban=data_array[:,2:4].astype(int)
tree=data_array[:462,4:].astype(int)

old_rgb=np.zeros((2168,3))
old_group=np.zeros((2168,1))

for i in range(np.shape(river)[0]):
    old_rgb[i,:]=I[river[i,0],river[i,1],:]
    old_group[i,0]=1
    
for i in range(np.shape(urban)[0]):
    old_rgb[794+i,:]=I[urban[i,0],urban[i,1],:]
    old_group[794+i,0]=2
    
for i in range(np.shape(tree)[0]):
    old_rgb[1706+i,:]=I[tree[i,0],tree[i,1],:]
    old_group[1706+i,0]=3
    
dis=distance.cdist(new_rgb,old_rgb,'euclidean')
ind_sort=np.argsort(dis)[:,:k]
new_group=np.zeros(np.shape(ind_sort)).astype(int)

# k=10
for p in range(846400):
    for q in range(k):
        new_group[p,q]=old_group[ind_sort[p,q]]
# new_group=new_group.astype(int)

new_def=np.zeros((846400,1))
for i in range(846400):
    new_def[i,0]=np.argmax(np.bincount(new_group[i,:]))
pic=new_def.reshape(920,920,1).astype(int) 

pic=np.swapaxes(pic,0,1)

# # palette = np.array([[  9,  41,  36],    # blue(river))
# #                     [135, 212, 192],    # green(tree)
# #                     [252, 244, 212]])   # red(urban)
# palette = np.array([[  0,   0, 255],   # blue(water))
#                     [  255, 0,   0],   # green(vegetation)
#                     [0,   255,   0]] )  # red(building)

# photo=np.zeros((920,920,3))
# for a in range(np.shape(pic)[0]):
#     for b in range(np.shape(pic)[1]):
#         if pic[a,b]==1:
#             photo[a,b]=palette[0]
#         elif pic[a,b]==2:
#             photo[a,b]=palette[1]
#         else:
#             photo[a,b]=palette[2]

            
plt.figure(figsize=(30,20))
plt.rc('font', size=30)
plt.title('KNN RESULT')
plt.imshow(pic,cmap='terrain')
# plt.imshow(photo)
