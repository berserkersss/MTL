#!/usr/bin/env python
# coding: utf-8

# # Fed-Learning in Wireless Environment
# 此函数用于构造所需数据集，并计算P(x)
# ## Import Libraries

# In[1]:

import pandas as pd
import numpy as np



def mnist_noniid(dataset_idx, num_label, num_img):
    """
    Sample non-I.I.D client data from MNIST dataset
    """
    dict_users = np.array([], dtype='int64')
    for k in range(len(num_label)):
        if num_label[k] >= 0:
            idxs = np.where(np.array(dataset_idx[1, :]) == num_label[k])
            idxs = idxs[0]
        else:
            idxs = dataset_idx[1, :]
        rand_set = np.random.choice(idxs, num_img[k], replace=False)
        train_idx = dataset_idx[0, :]

        dict_users = np.concatenate((dict_users, train_idx[rand_set]), axis=0)

        dataset_idx = np.delete(dataset_idx, rand_set, axis=1)

    return dict_users, dataset_idx


if __name__ == '__main__':
    csv_path_train_data = 'data/training_image.csv'
    csv_path_train_label = 'data/training_label.csv'

    # 设置用户数据集大小和标签种类
    num_img = [[700, 100], [400, 400], [700, 100], [400, 400],
               [700, 100], [700, 100]]
    num_label = [[0, -1], [0, -1], [1, -1], [1, -1], [2, -1], [3, -1]]

    # 导入MNIST数据集，数据集由 60,000 个训练样本和 10,000 个测试样本组成，每个样本
    # 为一个28*28的图片，读入时我们将这个图片转换为1*784的向量
    # header=None 表示文件一开始就是数据
    X_all_train = pd.read_csv(csv_path_train_data, header=None)
    y_all_train = pd.read_csv(csv_path_train_label, header=None)

    # In[3]:
    X_all_train = X_all_train.values
    y_all_train = y_all_train.values

    total = 60000
    idx_shard = [i for i in range(total)]
    dataset_idx = np.vstack((idx_shard, y_all_train.T))

    for k in range(len(num_img)):
        while True:
            index_set, dataset_idx = mnist_noniid(dataset_idx, num_label[k], num_img[k])
            X_train = X_all_train[index_set, :]
            y_train = y_all_train[index_set, :]
            filename_idx = 'csv/' + 'user' + str(k) + 'train_index_' + '.csv'
            np.savetxt(filename_idx, index_set, delimiter=',')
            break

        # filename = 'user_csv/' + 'user' + str(k) + 'train_label' + label + '.csv'
        # filename1 ='user_csv/' + 'user' + str(k) + 'train_img' + label + '.csv'
        # np.savetxt(filename1, X_train, delimiter=',')
        # np.savetxt(filename, y_train, delimiter=',')
