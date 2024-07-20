import os
import json
import numpy as np

catfile = os.path.join('./shapenetcore_partanno_segmentation_benchmark_v0_normal', 'synsetoffset2category.txt')
cat = {}
with open(catfile, 'r') as f:
    for line in f:
        ls = line.strip().split()
        cat[ls[0]] = ls[1]
cat = {k: v for k, v in cat.items()}
classes_original = dict(zip(cat, range(len(cat))))
meta=[]
for fn in cat:
    meta.append(fn)
print(meta)
classes = {}
for i in cat.keys():
    classes[i] = classes_original[i]

meta = {}
with open(os.path.join('./shapenetcore_partanno_segmentation_benchmark_v0_normal', 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
    train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
with open(os.path.join('./shapenetcore_partanno_segmentation_benchmark_v0_normal', 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
    val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
with open(os.path.join('./shapenetcore_partanno_segmentation_benchmark_v0_normal', 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
    test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

for item in cat:
    meta[item] = []
    dir_point = os.path.join('./shapenetcore_partanno_segmentation_benchmark_v0_normal', cat[item])
    fns = sorted(os.listdir(dir_point))
    fns = [fn for fn in fns]
    for fn in fns:
        token = (os.path.splitext(os.path.basename(fn))[0])
        meta[item].append(os.path.join(dir_point, token + '.txt'))

# datapath = []
# for item in cat:
#     for fn in meta[item]:
#         datapath.append((item, fn))
# classes = {}
# for i in cat.keys():
#     classes[i] = classes_original[i]
# cache = {}  # from index to (point_set, cls, seg) tuple
# cache_size = 20000
# # if index in cache:
# #     point_set, cls, seg = cache[index]
# # else:
# print(datapath)
# for i in range(len(datapath)):
#     fn =datapath[i]
#     cat = datapath[i][0]
#     cls = classes[cat]
#     cls = np.array([cls]).astype(np.int32)
#     data = np.loadtxt(fn[1]).astype(np.float32)
#     point_set = data[:, 0:6]
#     seg = data[:, -1].astype(np.int32)
# choice = np.random.choice(len(seg), self.npoints, replace=True)
# # resample
# point_set = point_set[choice, :]
# seg = seg[choice]