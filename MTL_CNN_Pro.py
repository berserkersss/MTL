#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# 用于CNN的平衡与不平衡数据的仿真, 手写体只需要一个通道
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import math
from scipy.linalg import sqrtm

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Update import CLUpdate
from models.Nets import CNNMnist_MTL, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.Fed import FedAvg_Optimize
from models.test import test_img

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=False, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=False, transform=trans_mnist)

    # sample users
    num_img = [800, 800, 800, 800, 800, 800]
    num_label = [[0, -1], [0, -1], [1, -1], [1, -1], [2, -1], [3, -1]]
    lambda_1 = 0.00001
    dict_users = {}
    for k in range(len(num_img)):
        #  导入unbalance数据集
        csv_path_train_data = 'csv/' + 'user' + str(k) + 'train_index_' + '.csv'
        train_index = pd.read_csv(csv_path_train_data, header=None)

        # 修剪数据集使得只有图片和标签,把序号剔除
        train_index = train_index.values
        train_index = train_index.T
        dict_users[k] = np.array(train_index[0].astype(int))

    net_glob = CNNMnist_MTL(args=args).to(args.device)
    net_test = copy.deepcopy(net_glob)
    net_glob.train()

    w_cluster = [net_glob.state_dict() for i in range(len(num_img))]  # 存储FL后的w（Fm）
    indicator_cluster = [[i] for i in range(len(num_img))]  # 存储每个簇的用户索引
    indicator_user = [i for i in range(len(num_img))]  # 指示用户属于哪一类
    Lm_select = [net_glob.state_dict() for i in range(len(num_img))]  # 存储用户选中簇的w（Lm）

    acc_train_cl_his, acc_train_fl_his, acc_train_cl_his_iid = [], [], []
    acc_train_cl_his2, acc_train_fl_his2 = [], []

    for iter in range(args.epochs):  # num of iterations
        # CL setting
        w_locals, loss_locals = [[], [], [], [], [], []], [[], [], [], [], [], []]
        for user in range(len(num_img)):
            # 获取用户的Lm
            if iter == 0:
                init_Lm = w_cluster[user]
            else:
                init_Lm = Lm_select[user]
            for cluster in range(len(w_cluster)):
                # 初始化用户本轮的w
                init_Fm = w_cluster[cluster]
                init_w = init_Fm
                init_w['linear.weight'] = init_Lm['linear.weight']
                init_w['linear.bias'] = init_Lm['linear.bias']
                if user not in indicator_cluster[cluster]:
                    # 构造大L矩阵
                    # Lm_user = init_Lm['linear.weight'].numpy()[0]
                    Lm_list = []
                    for i in indicator_cluster[cluster]:
                        # Lm_user = np.vstack((Lm_user, Lm_select[i]['linear.weight'].numpy()[0]))
                        Lm_list.append(Lm_select[i]['linear.weight'])
                else:
                    # Lm_user = init_Lm['linear.weight'].numpy()[0]
                    Lm_list = []
                    for i in indicator_cluster[cluster]:
                        if i != user:
                            # Lm_user = np.vstack((Lm_user, Lm_select[i]['linear.weight'].numpy()[0]))
                            Lm_list.append(Lm_select[i]['linear.weight'])
                if iter == 0:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user])  # data select
                    net_glob.load_state_dict(init_w)
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device),
                                          class_labels=num_label[user])
                    w_locals[user].append(copy.deepcopy(w))  # collect local model
                    loss_locals[user].append(copy.deepcopy(loss))  # collect local loss fucntion
                else:
                    # # 计算Omega矩阵
                    # if len(Lm_list) > 1:
                    #     S = sqrtm(np.dot(Lm_user, Lm_user.T))
                    #     Omega = S / np.trace(S)
                    #     Omega = np.linalg.inv(Omega)  # 可能出现为0的情况
                    #     Omega = Omega[0, :]
                    #     print(Omega)
                    # else:
                    #     Omega = [1]

                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user])  # data select
                    net_glob.load_state_dict(init_w)
                    w, loss = local.train_Om(net=copy.deepcopy(net_glob).to(args.device),
                                             class_labels=num_label[user], Lm=Lm_list, lambda_1=lambda_1)
                    w_locals[user].append(copy.deepcopy(w))  # collect local model
                    loss_locals[user].append(copy.deepcopy(loss))  # collect local loss fucntion

        # select the cluster
        idx = []
        Lm = w_locals
        for user in range(len(num_img)):
            min_idx = loss_locals[user].index(min(loss_locals[user]))
            idx.append(min_idx)
        set_idx = list(set(idx))
        indicator_cluster = [[] for i in range(len(set_idx))]
        w_cluster_temp = [[] for i in range(len(set_idx))]
        for k in range(len(set_idx)):
            for user in range(len(num_img)):
                if idx[user] == set_idx[k]:
                    indicator_user[user] = k
                    Lm_select[user] = w_locals[user][idx[user]]
                    indicator_cluster[k].append(user)
                    w_cluster_temp[k].append(w_locals[user][idx[user]])

        # cluster FL
        w_cluster = [0 for i in range(len(set_idx))]
        for k in range(len(set_idx)):
            w_cluster[k] = FedAvg(w_cluster_temp[k])

        print("iter=", iter)
        print(indicator_user)
        for user in range(len(num_img)):
            init_Fm = w_cluster[indicator_user[user]]
            init_w = init_Fm
            init_w['linear.weight'] = Lm_select[user]['linear.weight']
            init_w['linear.bias'] = Lm_select[user]['linear.bias']
            net_test.load_state_dict(init_w)
            acc_test_fl2 = test_img(net_test, dataset_test, args, num_label[user])
            print("user", user, "--Testing accuracy: {:.2f}".format(acc_test_fl2))
            acc_train_fl_his2.append(acc_test_fl2)

        # print("iter=", iter, 'loss=', loss_locals)
