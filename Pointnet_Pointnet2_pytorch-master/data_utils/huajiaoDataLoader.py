# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/huajiao', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.cat = []
        self.normal_channel = normal_channel
        self.cla=['ZhiGan', 'ShuYe', 'GuoShi']
        if split == 'trainval':
            for i in range(1, 16):
                self.cat.append('r' + str(i) + '_normal')
        elif split == 'train':
            for i in range(1, 16):
                self.cat.append('r' + str(i) + '_normal')
        elif split == 'val':
            for i in range(16, 18):
                self.cat.append('r' + str(i) + '_normal')
        elif split == 'test':
            for i in range(16,18):
                self.cat.append('r' + str(i) + '_normal')
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)


        # print(self.cat)
        self.meta =[]
        for i in range(len(self.cat)):
            # print('category', item)
            dir_point = os.path.join(self.root, self.cat[i])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta.append(os.path.join(dir_point, token + '.txt'))



        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'ZhiGan': [0], 'ShuYe': [1], 'GuoShi': [2]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.meta[index]
            data = np.loadtxt(fn).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            cls = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        cls=cls[choice]
        cls = np.array([cls]).astype(np.int32)
        return point_set, cls, seg

    def __len__(self):
        return len(self.meta)
