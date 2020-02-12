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
import random

from Dataset_select import Mtl_sample
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
    print(torch.cuda.is_available())
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

    num_avg = 10
    data_diff = 100
    lambda_1 = 0.000005
    for multi in range(num_avg):
        # sample users
        dict_users = {}
        num_img_init = [2000, 2000, 2000, 2000, 2000, 2000]
        num_label_init = [1, 1, 1, 1, 1, 1]

        num_img = [[] for i in range(args.num_users)]
        num_label = [[] for i in range(args.num_users)]

        label_class = range(10)
        for k in range(args.num_users):
            diff = int((multi+1) * data_diff)
            num_img[k].append(num_img_init[k] - diff)
            num_img[k].append(int(diff))
            num_label[k] = random.sample(label_class, num_label_init[k])
            num_label[k].append(-1)

        num_label[0] = num_label[1]
        num_label[2] = num_label[3]
        num_label[4] = num_label[5]
        print(num_label)

        dict_users = Mtl_sample(dict_users, num_label, num_img)

        net_glob = CNNMnist_MTL(args=args).to(args.device)
        net_test = copy.deepcopy(net_glob)
        net_glob.train()

        w_cluster = [net_glob.state_dict() for i in range(len(num_img))]  # 存储FL后的w（Fm）
        indicator_cluster = [[i] for i in range(len(num_img))]  # 存储每个簇的用户索引
        indicator_user = [i for i in range(len(num_img))]  # 指示用户属于哪一类
        Lm_select = [net_glob.state_dict() for i in range(len(num_img))]  # 存储用户选中簇的w（Lm）

        w_idep = [copy.deepcopy(net_glob) for i in range(len(num_img))]
        w_fl = [copy.deepcopy(net_glob) for i in range(len(num_img))]

        acc_train_cl_his, acc_train_fl_his, acc_train_cl_his_iid = [], [], []
        acc_train_cl_his2, acc_train_fl_his2 = [], []

        acc_train = [[] for i in range(len(num_img))]
        acc_train_idep = [[] for i in range(len(num_img))]
        for iter in range(args.epochs):  # num of iterations
            # CL setting
            w_locals, loss_locals = [[], [], [], [], [], []], [[], [], [], [], [], []]
            w_locals_idep = []


            for user in range(len(num_img)):
                # idep用户更新
                local = CLUpdate(args=args, dataset=dataset_train, idxs=dict_users[user])  # data select
                w_idep_temp, loss_idep = local.cltrain(net=w_idep[user].to(args.device), class_labels=num_label[user])
                w_idep[user].load_state_dict(w_idep_temp)
                w_locals_idep.append(loss_idep)
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
                        xxx = w_cluster[0]
                        xx = w_cluster[1]
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
                        xxx = w_cluster[0]
                        xx = w_cluster[1]
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
            print("Idep Loss:", w_locals_idep)
            for user in range(len(num_img)):
                init_Fm = w_cluster[indicator_user[user]]
                init_w = init_Fm
                init_w['linear.weight'] = Lm_select[user]['linear.weight']
                init_w['linear.bias'] = Lm_select[user]['linear.bias']
                net_test.load_state_dict(init_w)
                acc_test_fl2 = test_img(net_test, dataset_test, args, num_label[user])
                print("cluster: user", user, "--Testing accuracy: {:.2f}".format(acc_test_fl2))
                # acc_train_fl_his2.append(acc_test_fl2)
                acc_train[user].append(acc_test_fl2)

                # 输出idep的结果

                acc_test_fl2 = test_img(w_idep[user], dataset_test, args, num_label[user])
                print("idep: user", user, "--Testing accuracy: {:.2f}".format(acc_test_fl2))
                # acc_train_fl_his2.append(acc_test_fl2)
                acc_train_idep[user].append(acc_test_fl2)

        print(acc_train)
        colors = ["navy", "red", "black", "orange", "violet", "blue"]
        # labels = ["FedAvg_unbalance", "FedAvg_Optimize_unbalance", "FedAvg_balance", "FedAvg_Optimize_balance", "CL_iid", ""]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for user in range(len(num_img)):
            ax.plot(acc_train[user], c=colors[user], label=str(user))
        ax.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('Figure/Accuracy.png')

        print(acc_train_idep)
        colors = ["navy", "red", "black", "orange", "violet", "blue"]
        # labels = ["FedAvg_unbalance", "FedAvg_Optimize_unbalance", "FedAvg_balance", "FedAvg_Optimize_balance", "CL_iid", ""]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for user in range(len(num_img)):
            ax.plot(acc_train_idep[user], c=colors[user], label=str(user))
        ax.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('Figure/Accuracy2.png')

        print(acc_train_idep)
        colors = ["navy", "red", "black", "orange", "violet", "blue"]
        # labels = ["FedAvg_unbalance", "FedAvg_Optimize_unbalance", "FedAvg_balance", "FedAvg_Optimize_balance", "CL_iid", ""]
        for user in range(len(num_img)):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(acc_train_idep[user], c=colors[0], label=str(user))
            ax.plot(acc_train[user], c=colors[1], linestyle="dashed", label=str(user))
            ax.legend()
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            filename = 'Figure/' + "accuracy_c" + str(user) + ".png"
            plt.savefig(filename)

        filename = 'result/' + "accuracy.csv"
        np.savetxt(filename, [])
        filename = 'result/' + "accuracy_idep.csv"
        np.savetxt(filename, [])

        filename = 'result/' + "accuracy.csv"
        avg = 0
        for user in range(len(num_img)):
            with open(filename, "a") as myfile:
                myfile.write(str(max(acc_train[user])) + ',')
            avg += max(acc_train[user]) / len(num_img)
        with open(filename, "a") as myfile:
            myfile.write(str(avg) + ',')

        avg = 0
        filename = 'result/' + "accuracy_idep.csv"
        for user in range(len(num_img)):
            with open(filename, "a") as myfile:
                myfile.write(str(max(acc_train_idep[user])) + ',')
            avg += max(acc_train_idep[user]) / len(num_img)
        with open(filename, "a") as myfile:
            myfile.write(str(avg) + ',')
        # print("iter=", iter, 'loss=', loss_locals)
