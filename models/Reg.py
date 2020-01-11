import torch
from torch import nn
import torch.nn.functional as F


class Regularization(nn.Module):
    def __init__(self, model, weight_decay, p):
        """
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        """
        super(Regularization, self).__init__()

        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        """
        指定运行模式
        :param device: cude or cpu
        :return:
        """
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        """
        获得模型的权重列表
        :param model:
        :return:
        """
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p):
        """
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        """
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0

        Omega = weight_decay
        Lm = p

        for name, w in weight_list:
            if name == 'linear.weight':
                if len(Omega) > 0:
                    for i in range(len(Lm)):
                        l2_reg = 2 * Omega[i] * torch.mm(w, Lm[i].t())
                        reg_loss = reg_loss + l2_reg
                else:
                    reg_loss = 0
        return reg_loss

    def weight_info(self, weight_list):
        """
        打印权重列表信息
        :param weight_list:
        :return:
        """
        #print("---------------regularization weight---------------")
        #for name, w in weight_list:
           #print(name)
        #print("---------------------------------------------------")
