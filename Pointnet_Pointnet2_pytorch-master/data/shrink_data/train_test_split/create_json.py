import os
import random
from traceback import print_list
import json
# filePath ='/home/lion/pointnet2_python3_priky/data/prickly_ash_data3/prickly'
filePath = '/home/lion/shrink_data'
file_list_pre = os.listdir(filePath)  # 获取当前文件夹下的文件夹名

data_list = []
train_list = []
test_list = []
val_list = [] 
for i in file_list_pre:  # 依次打开子文件夹，获取子文件夹的文件名。这将是一个二维列表
    if  "train" in i:
        pass
    else:
        for root, dirs, files in os.walk(filePath+"/"+i):
            for name in files:
                base_name=os.path.splitext(name)[0]  #去掉后缀 .txt
                data_list.append(os.path.join("shape_data/"+i,base_name))
random.shuffle(data_list)
#print(len(data_list))


# #划分训练集，测试集，验证集。比例6：2：2
# #总共896个数据集，其中538个训练集，179个测试集，179个验证集
for i in data_list:
    if data_list.index(i)<538:
        train_list.append(i)
    elif data_list.index(i)>=538 and data_list.index(i)<717:
        test_list.append(i)
    elif data_list.index(i)>=717 and data_list.index(i)<896:
        val_list.append(i)
#再次打乱
random.shuffle(train_list)
random.shuffle(test_list)
random.shuffle(val_list)


with open('shuffled_test_file_list.json','w') as file_obj:
    json.dump(test_list,file_obj)
with open('shuffled_train_file_list.json','w') as file_obj:
    json.dump(train_list,file_obj)
with open('shuffled_val_file_list.json','w') as file_obj:
    json.dump(val_list,file_obj)
