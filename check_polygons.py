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



##############################################################################################################
############################################### Define Model #################################################
##############################################################################################################

edge_classifier = EdgeClassifier()
edge_classifier = edge_classifier.cuda()
edge_classifier = edge_classifier.eval()

epoch = 10
src = 'saved_models'
edge_classifier.load_state_dict(torch.load('./{}/edge_classifier_iter_{}.pth'.format(src, epoch)))


##############################################################################################################
############################################# Setup Training #################################################
##############################################################################################################
PREFIX = '/home/nelson/Workspace'
RGB_FOLDER = '{}/building_reconstruction/la_dataset_new/rgb'.format(PREFIX)
OUT_FOLDER = '{}/building_reconstruction/la_dataset_new/outlines'.format(PREFIX)
ANNOTS_FOLDER = '{}/building_reconstruction/la_dataset_new/annots'.format(PREFIX)
EDGES_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_annots_dets/edges'.format(PREFIX)
CORNERS_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_annots_dets/corners'.format(PREFIX)
#EDGES_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/expanded_primitives_dets/edges'
#CORNERS_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/expanded_primitives_dets/corners'
with open('{}/building_reconstruction/la_dataset_new/train_list_prime.txt'.format(PREFIX)) as f:
    valid_list = [line.strip() for line in f.readlines()]

# create loaders
dset_valid = GraphData(valid_list, ANNOTS_FOLDER, EDGES_FOLDER, CORNERS_FOLDER, with_augmentation=True)
valid_loader = DataLoader(dset_valid, batch_size=1, shuffle=False, num_workers=1)

##############################################################################################################
################################################ Start Search ################################################
##############################################################################################################
def draw_relations(corners, edges, edges_conf, connections, _id):
    tresh = .5
    # # Draw corners
    # for i in range(corners.shape[1]):
    #     y, x = corners[0, i, :2]
    #     y, x = float(y), float(x)
    #     dwg.add(dwg.circle(center=(x,y),r=2, stroke='yellow', fill='white', stroke_width=1, opacity=.8))
    
    # Collect relations
    edges_to_draw = []
    corners_to_draw = set()
    all_corners = set()
    for i in range(edges.shape[0]):
        y1, x1, y2, x2 = edges[i, :]
        if edges_conf[i] > tresh:
            inds = np.where(connections[:, i] == 1)
            if inds[0].shape[0] > 2:
                print('ERROR', _id)
                print(inds[0].shape)
            c1_idx = inds[0][0]
            c2_idx = inds[0][1]
            pt1 = corners[c1_idx, :2]
            pt2 = corners[c2_idx, :2]
            edges_to_draw.append([float(pt1[1]), float(pt1[0]), float(pt2[1]), float(pt2[0])])
            corners_to_draw.update([(float(pt1[1]), float(pt1[0]), c1_idx)])
            corners_to_draw.update([(float(pt2[1]), float(pt2[0]), c2_idx)])

    
    for k in range(corners.shape[0]):
        pt = corners[k, :2]
        all_corners.update([(float(pt[1]), float(pt[0]), k)])

    # Draw edges and corner
    for e in edges_to_draw:
        x1, y1, x2, y2 = e
        dwg.add(dwg.line((x1, y1), (x2, y2), stroke='magenta', stroke_width=1, opacity=.7))

    for c in corners_to_draw:
        x, y, c_id  = c
        dwg.add(dwg.circle(center=(x,y),r=1.5, stroke='blue', fill='white', stroke_width=0.8, opacity=.8))

    for c in all_corners:
        if c not in corners_to_draw:
            x, y, c_id  = c
            dwg.add(dwg.circle(center=(x,y),r=1.5, stroke='red', fill='white', stroke_width=0.8, opacity=.8))
    return

def _steps(im_arr, bin_state):
    probs, logits = edge_classifier(im_arr, bin_state)
    return probs

fract = 0.0
preds = []
targets = []
# for each building
for l, data in enumerate(valid_loader):

    # get the inputs
    _id = valid_list[l]
    corners_det, edges_det, corner_edge, e_xys, current_edges, ce_angles_bins = data

    # format input
    ce_angles_bins = ce_angles_bins.squeeze(0)
    corners_det = corners_det.squeeze(0)
    edges_det = edges_det.squeeze(0)
    corner_edge = corner_edge.squeeze(0)
    e_xys = e_xys.squeeze(0)
    current_edges = current_edges.squeeze(0)
    current_edges = torch.zeros_like(current_edges)

    edges_deg = torch.sum(corner_edge, 0) 
    inds = torch.nonzero(edges_deg == 1)
    print(inds.size()[0])
    if inds.size()[0] > 0:
        print('ERR')
        print(_id)