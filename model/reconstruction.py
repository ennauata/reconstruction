import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import glob
import numpy as np
import pickle as p
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset.custom_dataloader_rec import GraphData
from torch.utils.data import DataLoader
from dataset.collate import PadCollate
from utils.losses import balanced_binary_cross_entropy
import os
from dataset.metrics import Metrics
from collections import OrderedDict
import svgwrite
from cairosvg import svg2png
from utils.utils import compose_im
from sklearn import linear_model
from utils.utils import reconstruct
from model.graph import EdgeClassifier


class ReconstructionModule():
    def __init__(self):
        super(ReconstructionModule, self).__init__()
        return

    def draw_edges(self, edges_on, e_xys):
        im = np.zeros((128, 128))
        for k in range(edges_on.shape[0]):
            if edges_on[k] == 1:
                xs, ys = e_xys[k]
                pos_inds = np.where(xs >= 0)[0]
                xs = xs[pos_inds]
                ys = ys[pos_inds]
                im[xs, ys] += 1.0
        return im

    def greedy_over_edges(self, current_edges, corner_edge, e_xys, im, edge_classifier):

        # enumerate all actions
        current_edges = np.zeros_like(current_edges)

        keep_track = (-1, -1)
        for _ in range(10):
            #inds = np.where(current_edges == 0)[0]
            for k in range(corner_edge.shape[1]):

                # draw current state
                im_s0 = self.draw_edges(current_edges, e_xys)

                # draw new state
                new_state = np.array(current_edges)
                new_state[k] = abs(1.0-new_state[k])
                im_s1 = self.draw_edges(new_state, e_xys)

                # predict next state
                im_sx = np.concatenate([im.transpose(2, 0, 1)/255.0, im_s0[np.newaxis, :, :], im_s1[np.newaxis, :, :]])
                with torch.no_grad():
                    _input = torch.from_numpy(im_sx).float().cuda().unsqueeze(0)
                    probs, logits = edge_classifier(_input)

                prob = probs[0][1]
                # if prob > keep_track[0]:
                #      keep_track = (prob, k)
                
                # # Debug
                # im0 = Image.fromarray((im_s0>0)*255.0)
                # im1 = Image.fromarray((im_s1>0)*255.0)
                # print(prob)
                # plt.figure()
                # plt.imshow(im0)
                # plt.figure()
                # plt.imshow(im1)
                # plt.show() 

                # update current state
                if prob > .5:
                    #current_edges[keep_track[1]] = abs(1.0-current_edges[keep_track[1]])
                    current_edges[k] = abs(1.0-current_edges[k])

        return current_edges