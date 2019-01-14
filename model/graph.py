import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.attention import attend
from torch.autograd import Variable

# # GNN - Graph Neural Network
# class EdgeClassifier(nn.Module):
#     def __init__(self):
#         super(EdgeClassifier, self).__init__()

#         self.enc1_1 = nn.Conv2d(4, 128, kernel_size=3, stride=2)
#         self.enc1_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
#         self.enc1_3 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
#         self.enc1_4 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
#         self.enc1_5 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
#         self.enc1_6 = nn.Conv2d(128, 128, kernel_size=3, stride=2)

#         self.conv1_bn = nn.GroupNorm(32, 128)
#         self.conv2_bn = nn.GroupNorm(32, 128)
#         self.conv3_bn = nn.GroupNorm(32, 128)
#         self.conv4_bn = nn.GroupNorm(32, 128)
#         self.conv5_bn = nn.GroupNorm(32, 128)
#         self.conv6_bn = nn.GroupNorm(32, 128)

#         self.enc2_1 = nn.Linear(1152, 512, bias=True)
#         self.enc2_2 = nn.Linear(512, 512, bias=True)
#         self.enc2_3 = nn.Linear(1024, 512, bias=True)
#         self.enc2_4 = nn.Linear(512, 512, bias=True)
#         self.enc2_5 = nn.Linear(512, 2, bias=True)
#         self.softmax = nn.Softmax(dim=-1)

#         self.dense1_bn = nn.BatchNorm1d(512)
#         self.dense2_bn = nn.BatchNorm1d(512)

#     def forward(self, x):

        
#         # corners updates
#         Ne = x.shape[1]
        
#         feats_s = []
#         for k in range(3, x.shape[1]):

#             x_s = torch.cat([x[:, :3, :, :], x[:, k, :, :].unsqueeze(1)], 1)
#             x_s = F.relu(self.conv1_bn(self.enc1_1(x_s)))
#             x_s = F.relu(self.conv2_bn(self.enc1_2(x_s)))
#             x_s = F.relu(self.conv3_bn(self.enc1_3(x_s)))
#             x_s = F.relu(self.conv4_bn(self.enc1_4(x_s)))
#             x_s = F.relu(self.conv5_bn(self.enc1_5(x_s)))
#             x_s = F.relu(self.conv6_bn(self.enc1_6(x_s)))
#             x_s = x_s.view(-1, 1152)
#             x_s = F.relu(self.enc2_1(x_s))
#             x_s = F.relu(self.enc2_2(x_s))
#             feats_s.append(x_s)

#         # from PIL import Image
#         # import matplotlib.pyplot as plt
#         # im = x[0, :3, :, :]*255.0
#         # im = im.cpu().numpy().transpose(1, 2, 0)
#         # im = Image.fromarray(im.astype('uint8'))
#         # im_s0 = Image.fromarray(x[0, 3, :, :].cpu().numpy()*255.0)
#         # im_s1 = Image.fromarray(x[0, 4, :, :].cpu().numpy()*255.0)

#         # plt.figure()
#         # plt.imshow(im)
#         # plt.figure()
#         # plt.imshow(im_s0)
#         # plt.figure()
#         # plt.imshow(im_s1)
#         # plt.show()

#         feats_s = torch.cat(feats_s, -1)
#         feats_s = F.relu(self.dense1_bn(self.enc2_3(feats_s)))
#         feats_s = F.relu(self.dense2_bn(self.enc2_4(feats_s)))
#         logits = self.enc2_5(feats_s)
#         probs = self.softmax(logits)

#         return  probs, logits

class EdgeClassifier(nn.Module):
    def __init__(self):
        super(EdgeClassifier, self).__init__()

        self.enc1_1 = nn.Conv2d(6, 128, kernel_size=3, stride=2)
        self.enc1_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.enc1_3 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.enc1_4 = nn.Conv2d(128, 512, kernel_size=3, stride=2)
        self.enc1_5 = nn.Conv2d(512, 512, kernel_size=3, stride=2)
        self.enc1_6 = nn.Conv2d(512, 512, kernel_size=3, stride=2)
        self.enc1_7 = nn.Conv2d(512, 512, kernel_size=3, stride=2)

        self.enc3_1 = nn.Linear(288, 512, bias=True)
        self.enc3_2 = nn.Linear(512, 512, bias=True)

        self.enc2_1 = nn.Linear(512, 512, bias=True)
        self.enc2_2 = nn.Linear(512, 512, bias=True)
        self.enc2_3 = nn.Linear(512, 512, bias=True)
        self.enc2_4 = nn.Linear(512, 2, bias=True)
        self.softmax = nn.Softmax(dim=-1)

        # self.conv1_bn = nn.GroupNorm(32, 64)
        # self.conv2_bn = nn.GroupNorm(32, 64)
        # self.conv3_bn = nn.GroupNorm(32, 64)
        # self.conv4_bn = nn.GroupNorm(32, 64)
        # self.conv5_bn = nn.GroupNorm(32, 64)
        # self.conv6_bn = nn.GroupNorm(32, 512)

        # self.dense1_bn = nn.BatchNorm1d(512)
        # self.dense2_bn = nn.BatchNorm1d(512)
        # self.dense3_bn = nn.BatchNorm1d(512)

    def forward(self, x, y):

        # corners updates        
        x = F.relu(self.enc1_1(x))
        x = F.relu(self.enc1_2(x))
        x = F.relu(self.enc1_3(x))
        x = F.relu(self.enc1_4(x))
        x = F.relu(self.enc1_5(x))
        x = F.relu(self.enc1_6(x))
        x = F.relu(self.enc1_7(x))
        x = x.view(-1, 512)

        # # split corners
        # y1_c = y[:, :72]
        # y2_c = y[:,72:144]
        # y1_n = y[:,144:216]
        # y2_n = y[:,216:288]

        # #  concatenate change in corners
        # y1 = torch.cat([y1_c, y1_n], -1)
        # y2 = torch.cat([y2_c, y2_n], -1)

        # # encode corner 1
        # y1 = F.relu(self.enc3_1(y1))
        # y1 = F.relu(self.enc3_2(y1))

        # # encode corner 2
        # y2 = F.relu(self.enc3_1(y2))
        # y2 = F.relu(self.enc3_2(y2))

        # # combine corners
        # y = y1 + y2

        # encode corners
        # y = F.relu(self.enc3_1(y))
        # y = F.relu(self.enc3_2(y))

        # combine bins and image
        #z = torch.cat([x, y], -1)
        z = x
        z = F.relu(self.enc2_1(z))
        z = F.relu(self.enc2_2(z))
        z = F.relu(self.enc2_3(z))

        # # corners updates        
        # x_s = F.relu(self.conv1_bn(self.enc1_1(x)))
        # x_s = F.relu(self.conv2_bn(self.enc1_2(x_s)))
        # x_s = F.relu(self.conv3_bn(self.enc1_3(x_s)))
        # x_s = F.relu(self.conv4_bn(self.enc1_4(x_s)))
        # x_s = F.relu(self.conv5_bn(self.enc1_5(x_s)))
        # x_s = F.relu(self.conv6_bn(self.enc1_6(x_s)))

        # x_s = x_s.view(-1, 512)
        # x_s = F.relu(self.dense1_bn(self.enc2_1(x_s)))
        # x_s = F.relu(self.dense2_bn(self.enc2_2(x_s)))
        # x_s = F.relu(self.dense3_bn(self.enc2_3(x_s)))

        logits = self.enc2_4(z)
        probs = self.softmax(logits)

        return  probs, logits

