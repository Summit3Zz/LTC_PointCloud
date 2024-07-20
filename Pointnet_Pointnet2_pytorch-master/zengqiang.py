import numpy as np
import provider
import os
import json
import warnings
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

def rotate_point_cloud_with_normal(batch_xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    '''
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    shape_pc = batch_xyz_normal[:,0:3]
    shape_normal = batch_xyz_normal[:,3:6]
    batch_xyz_normal[:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    batch_xyz_normal[:,3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal

root = './data/huajiao'
cat = []
for i in range(16, 18):
    cat.append('r' + str(i) + '_normal')

print(cat)
for i in range(len(cat)):
    # print('category', item)
    dir_point = os.path.join(root, cat[i])
    fns = sorted(os.listdir(dir_point))
    # print(os.path.basename(fns))
    meta=[]
    for fn in fns:
        token = (os.path.splitext(os.path.basename(fn))[0])
        meta.append(os.path.join(dir_point, token + '.txt'))
    for k in range(len(meta)):
        fn = meta[k]
        data = np.loadtxt(fn)
        point_set = data[:, 0:6]
        set=data[:,6:7]
#set=torch.from_numpy(set)
        point_set=rotate_point_cloud_with_normal(point_set)
#point_set=torch.from_numpy(point_set)
        set=torch.from_numpy(set)
        point_set=torch.from_numpy(point_set)
        data=torch.cat([point_set,set],dim=1)
        mydir='data/huajiao/'+cat[i]+'/fanzhuan'+str(k)+'.txt'
        np.savetxt(mydir,data,fmt='%.8f')