import sys
sys.path.append('..')

import scipy.io
import pickle
import numpy as np
from utils.datasets import load

# datus = load("SJAFFE", return_X_y=False)
# X=datus.feature
# length=len(X)
# n_feature=X.shape[1]
# data=X
# # label=mat['DRML_multi']
# label=datus.logical_label
# n_label=label.shape[1]
# sparse=1
# D=datus.label_distribution
# dict_data= {"data":X,"n_label":n_label,"n_feature":n_feature,"sparse":sparse,"label":label,"length":length}
# dict_data1=D
# # print(n_label)
# with open("SJAFFE.plk", 'wb') as fo:  # 将数据写入pkl文件
#     pickle.dump(dict_data, fo)
# fo.close()
with open("Datasets/datas/SJAFFE/SJAFFE.plk", 'rb') as fo:  # 读取pkl文件数据
    dict_data = pickle.load(fo)

with open("SJAFFE_d.plk", 'wb') as fo:  # 将数据写入pkl文件
    pickle.dump(dict_data1, fo)
fo.close()
with open("SJAFFE_d.plk", 'rb') as fo:  # 读取pkl文件数据
    dict_data1 = pickle.load(fo)
fo.close()
print(dict_data.keys())  # 测试我们读取的文件
print(dict_data)
print(dict_data["label"])
print(dict_data1.keys())  # 测试我们读取的文件
print(dict_data1)
print(dict_data1["distributions"])