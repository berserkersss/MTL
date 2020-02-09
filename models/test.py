#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args, user_labels):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        temp = target
        label_1 = (temp == user_labels[0]).nonzero()
        temp[label_1] = 10
        label_2 = (temp != 10).nonzero()
        temp[label_2] = 0
        temp[label_1] = 1

        target = temp
        # if args.gpu != -1:
        #     data, target = data.cuda(), target.cuda()

        data, target = data.to(args.device), target.to(args.device)

        log_probs = net_g(data)
        # sum up batch loss
        # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability

        temp = log_probs.data
        label_1 = (log_probs >= 0).nonzero()
        temp[label_1] = 1
        label_2 = (temp < 0).nonzero()
        temp[label_2] = 0
        temp[label_1] = 1

        y_pred = temp
        target = target.float()
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        break

    # test_loss /= len(data_loader.dataset)
    accuracy = 100.0000 * correct.item() / args.bs
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy

