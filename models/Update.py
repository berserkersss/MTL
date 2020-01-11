#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.Reg import Regularization

import numpy as np
import pandas as pd
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        img_idx = self.idxs[item]
        return image, label, img_idx


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.BCEWithLogitsLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs),
                                    batch_size=int(len(DatasetSplit(dataset, idxs)) / self.args.local_num),
                                    shuffle=True)

    def train(self, net, class_labels):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, img_idxs) in enumerate(self.ldr_train):
                # 标签转换
                temp = labels
                label_1 = (temp == class_labels[0]).nonzero()
                temp[label_1] = 10
                label_2 = (temp != 10).nonzero()
                temp[label_2] = 0
                temp[label_1] = 1

                images, labels = images.to(self.args.device), temp.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)  # predicted label
                log_probs = log_probs.squeeze(dim=-1)
                loss = self.loss_func(log_probs, labels.float())
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_Om(self, net, class_labels, Lm, lambda_1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        epoch_MTL_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_loss_MTL = []
            for batch_idx, (images, labels, img_idxs) in enumerate(self.ldr_train):
                reg_loss = Regularization(net, 0.1, p=Lm).to(self.args.device)

                # 标签转换
                temp = labels
                label_1 = (temp == class_labels[0]).nonzero()
                temp[label_1] = 10
                label_2 = (temp != 10).nonzero()
                temp[label_2] = 0
                temp[label_1] = 1

                images, labels = images.to(self.args.device), temp.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)  # predicted label
                log_probs = log_probs.squeeze(dim=-1)
                loss = self.loss_func(log_probs, labels.float())
                MTL_loss = loss
                loss = loss + lambda_1 * (reg_loss(net)) / len(labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                batch_loss_MTL.append(MTL_loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_MTL_loss.append(sum(batch_loss_MTL) / len(batch_loss_MTL))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class CLUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.BCEWithLogitsLoss()
        self.cl_train = DataLoader(DatasetSplit(dataset, idxs),
                                   batch_size=int(len(DatasetSplit(dataset, idxs)) / self.args.local_num), shuffle=True)
        # self.cl_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def cltrain(self, net, class_labels):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss_g = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, img_idxs) in enumerate(self.cl_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # print(images.shape)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                w_g = net.state_dict()
                if self.args.verbose and batch_idx % 60 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss_g.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss_g) / len(epoch_loss_g)
