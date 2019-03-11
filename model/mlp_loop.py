import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import cv2
import numpy as np
from model.modules import findLoopsModule
from torch.nn import functional as F
from model.resnet import create_model as create_resnet

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

class CMP(nn.Module):

    def __init__(self, options, feat_size=168, num_classes=1000):
        super(CMP, self).__init__()

        self.C1 = nn.Conv2d(feat_size, 512, kernel_size=5, stride=2, padding=1)
        self.C2 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=1)
        self.C3 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=1)
        self.C4 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=1)
        self.C5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=1)
        self.C6 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.fc_0 = nn.Linear(512, 512)
        self.fc_1 = nn.Linear(512, 512)
        self.fc_2 = nn.Linear(512, 1)
        #self.fc_3 = nn.Linear(2048, num_classes)

        return 

    def forward(self, x):

        # Conv block 1
        x = self.C1(x)
        x = self.relu(x)

        # Conv block 2
        x = self.C2(x)
        x = self.relu(x)

        # Conv block 3
        x = self.C3(x)
        x = self.relu(x)

        # Conv block 4
        x = self.C4(x)
        x = self.relu(x)

        # Conv block 5
        x = self.C5(x)
        x = self.relu(x)

        # Conv block 6
        x = self.C6(x)
        x = self.relu(x)

        x = x.view(-1, 512)

        z = self.fc_0(x)
        z = self.relu(z)

        z = self.fc_1(z)
        z = self.relu(z)

        # z = self.fc_2(z)
        # z = self.relu(z)

        z = self.fc_2(z)
        z = torch.sigmoid(z).view(-1)

        return z

class MLP(nn.Module):

    def __init__(self, options, num_classes=1000):
        super(MLP, self).__init__()

        # nn.BatchNorm1d(16)
        # nn.BatchNorm1d(16)
        # nn.BatchNorm1d(16)
        # nn.BatchNorm1d(16)
        # self.nlb_1 = NONLocalBlock1D(20, sub_sample=False, bn_layer=False)
        # self.nlb_2 = NONLocalBlock1D(20, sub_sample=False, bn_layer=False)

        self.fc_agg_0 = nn.Linear(524, 2048)
        self.fc_agg_1 = nn.Linear(2048, 2048)
        self.fc_agg_2 = nn.Linear(2048, 2048)
        self.fc_agg_3 = nn.Linear(2048, 512)

        self.relu = nn.ReLU(inplace=True)
        self.fc_0 = nn.Linear(1536, 2048)
        self.fc_1 = nn.Linear(2048, 2048)
        self.fc_2 = nn.Linear(2048, 2048)
        self.fc_3 = nn.Linear(2048, num_classes)

        return 

    def forward(self, xs):

        # xs = self.nlb_1(xs)
        # xs = self.nlb_2(xs)
        # xs = xs.view(-1, 512*20)

        ys = []
        for x in xs:
            y = self.relu(self.fc_agg_0(x))
            y = self.relu(self.fc_agg_1(y))
            y = self.relu(self.fc_agg_2(y))
            y = self.relu(self.fc_agg_3(y))
            y = torch.cat([torch.max(y, 0)[0], torch.min(y, 0)[0], torch.sum(y, 0)])
            ys.append(y)
        ys = torch.stack(ys)

        z = self.fc_0(ys)
        z = self.relu(z)

        z = self.fc_1(z)
        z = self.relu(z)

        z = self.fc_2(z)
        z = self.relu(z)

        z = self.fc_3(z)
        z = torch.sigmoid(z).view(-1)

        return z


def create_model(options):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #model = create_resnet(options)
    model = CMP(options, num_classes=1)

    return model