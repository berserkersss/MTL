#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# 这个用于CNN的仿真, 手写体只需要一个通道
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    '''
    num = 5
    colors = ["navy", "red", "black", "orange", "violet"]
    labels = ["FedAvg_S", "FedAvg_Optimize_S", "FedAvg_L", "FedAvg_Optimize_L", "CL_iid"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #  导入unbalance数据集
    for i in range(num):
        csv_path_accuracy = 'result/MLP/' + 'Accuracy_' + labels[i] + '_MLP.csv'
        accuracy = pd.read_csv(csv_path_accuracy, header=None)
        accuracy = accuracy.values
        ax.plot(accuracy[0], c=colors[i], label=labels[i])
        ax.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('Figure/Accuracy.png')
    '''
    '''
    num = 5
    colors = ["navy", "red", "black", "orange", "violet"]
    labels = ["FedAvg_S", "FedAvg_Optimize_S", "FedAvg_L", "FedAvg_Optimize_L", "CL_iid"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #  导入unbalance数据集
    for i in range(num):
        csv_path_accuracy = 'result/MLP/' + 'Accuracy_' + labels[i] + '_MLP.csv'
        accuracy = pd.read_csv(csv_path_accuracy, header=None)
        accuracy = accuracy.values
        if i == 1 or i == 3:
            accuracy = np.delete(accuracy, -1 , axis = 1)
            A_Len = len(accuracy[0])
            index = np.argmax(accuracy)
            R_array = accuracy[0][index-10:index]
            A_array = np.tile(R_array, (1, int(A_Len/10 - index/10)))
            accuracy = np.delete(accuracy, index + np.arange(A_Len-index), axis = 1)
            accuracy = np.hstack((accuracy, A_array))
        ax.plot(accuracy[0], c=colors[i], label=labels[i])
        ax.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('Figure/Accuracy.png')
    '''
    '''
    num = 4
    colors = ["navy", "red", "black", "orange", "violet"]
    labels = ["FedAvg_S", "FedAvg_Optimize_S", "FedAvg_L", "FedAvg_Optimize_L"]
    linestyles = ["dashed", "dashdot", "dotted", "--", "solid"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #  导入unbalance数据集
    for i in range(num):
        csv_path_accuracy = 'result/CNN_D/' + 'Accuracy_' + labels[i] + '_CNN.csv'
        accuracy = pd.read_csv(csv_path_accuracy, header=None)
        accuracy = accuracy.values
        x = np.arange(int((len(accuracy[0])) / 2)) * 2
        ax.plot(x, accuracy[0][x], c=colors[i], label=labels[i])
        ax.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('Figure/Accuracy_CNN_D.png')
        plt.savefig('Figure/Accuracy_CNN_D.eps')
    '''

    num = 4
    colors = ["navy", "red", "black", "orange", "violet"]
    labels = ["FedAvg_unbalance", "FedAvg_Optimize_unbalance", "FedAvg_balance", "FedAvg_Optimize_balance",
              "CL_iid"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #  导入unbalance数据集
    for i in range(num):
        csv_path_accuracy = 'result/CNN/' + 'Accuracy_' + labels[i] + '_CNN.csv'
        accuracy = pd.read_csv(csv_path_accuracy, header=None)
        accuracy = accuracy.values
        x = np.arange(int((len(accuracy[0])) / 2)) * 2
        ax.plot(x, accuracy[0][x], c=colors[i], label=labels[i])
        ax.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('Figure/Accuracy_CNN2.png')
        plt.savefig('Figure/Accuracy_CNN2.eps')


    # SVM
    '''
    num = 5
    colors = ["navy", "red", "black", "orange", "violet"]
    labels = ["FedAvg_S", "FedAvg_Optimize_S", "FedAvg_L", "FedAvg_Optimize_L", "CL_iid"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #  导入unbalance数据集
    for i in range(num):
        csv_path_accuracy = 'result/SVM/' + 'Accuracy_' + labels[i] + '_SVM.csv'
        accuracy = pd.read_csv(csv_path_accuracy, header=None)
        accuracy = accuracy.values
        ax.plot(accuracy[0], c=colors[i], label=labels[i])
        ax.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('Figure/Accuracy_SVM.png')
        plt.savefig('Figure/Accuracy_SVM.eps')
    '''

    '''
    num = 5
    colors = ["navy", "red", "black", "orange", "violet"]
    labels = ["FedAvg_unbalance", "FedAvg_Optimize_unbalance", "FedAvg_balance", "FedAvg_Optimize_balance",
              "CL_iid"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #  导入unbalance数据集
    for i in range(num):
        csv_path_accuracy = 'result/SVM/' + 'Accuracy_' + labels[i] + '_SVM.csv'
        accuracy = pd.read_csv(csv_path_accuracy, header=None)
        accuracy = accuracy.values
        if i == 1 or i == 3:
            accuracy = np.delete(accuracy, -1, axis=1)
            A_Len = len(accuracy[0])
            index = np.argmax(accuracy)
            R_array = accuracy[0][index - 10:index]
            A_array = np.tile(R_array, (1, int(A_Len / 10 - index / 10)))
            accuracy = np.delete(accuracy, index + np.arange(A_Len - index), axis=1)
            accuracy = np.hstack((accuracy, A_array))
        x = np.arange(int((len(accuracy[0])) / 2)) * 2
        ax.plot(x, accuracy[0][x], c=colors[i], label=labels[i])
        ax.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('Figure/Accuracy_SVM2.png')
        plt.savefig('Figure/Accuracy_SVM2.eps')
    '''