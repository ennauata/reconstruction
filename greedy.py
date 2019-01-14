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

epoch = 40
src = 'saved_models'
edge_classifier.load_state_dict(torch.load('./{}/edge_classifier_iter_{}.pth'.format(src, epoch)))


##############################################################################################################
############################################# Setup Training #################################################
##############################################################################################################
PREFIX = '/home/nelson/Workspace'
RGB_FOLDER = '{}/building_reconstruction/la_dataset_new/rgb'.format(PREFIX)
OUT_FOLDER = '{}/building_reconstruction/la_dataset_new/outlines'.format(PREFIX)
ANNOTS_FOLDER = '{}/building_reconstruction/la_dataset_new/annots'.format(PREFIX)
EDGES_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_dets/edges'.format(PREFIX)
CORNERS_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_dets/corners'.format(PREFIX)
#EDGES_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/expanded_primitives_dets/edges'
#CORNERS_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/expanded_primitives_dets/corners'
with open('{}/building_reconstruction/la_dataset_new/valid_list.txt'.format(PREFIX)) as f:
    valid_list = [line.strip() for line in f.readlines()]

# create loaders
dset_valid = GraphData(valid_list, ANNOTS_FOLDER, EDGES_FOLDER, CORNERS_FOLDER, with_augmentation=False)
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

    
    for k in range(corners.shape[1]):
        pt = corners[k, :2]
        all_corners.update([(float(pt[1]), float(pt[0]), k)])


    # Draw edges and corner
    for e in edges_to_draw:
        x1, y1, x2, y2 = e
        dwg.add(dwg.line((x1, y1), (x2, y2), stroke='magenta', stroke_width=1, opacity=.7))

    for c in corners_to_draw:
        x, y, c_id  = c
        dwg.add(dwg.circle(center=(x,y),r=1.5, stroke='blue', fill='white', stroke_width=0.8, opacity=.8))

    return

def draw_edges(edges_on, e_xys):
    im = np.zeros((128, 128))
    for k in range(edges_on.shape[0]):
        if edges_on[k] == 1:
            xs, ys = e_xys[k]
            im[xs, ys] += 1.0
    return im

def _steps(im_arr):
    probs, logits = edge_classifier(im_arr)
    return probs
                
fract = 0.0
preds = []
targets = []
# for each building
for l, data in enumerate(valid_loader):

    # get the inputs
    _id = valid_list[l]
    corners_det, edges_det, corner_edge, e_xys, current_edges = data

    # format input
    corners_det = corners_det.squeeze(0)
    edges_det = edges_det.squeeze(0)
    corner_edge = corner_edge.squeeze(0)
    e_xys = e_xys.squeeze(0)
    current_edges = current_edges.squeeze(0)
    current_edges = torch.zeros_like(current_edges)

    # get corners degree roughly
    #inds = torch.nonzero(corner_edge==1)
    # deg = torch.zeros_like(corner_edge)
    # deg[inds[0].data, inds[1].data] = current_edges[inds[1].data]
    # deg = torch.sum(deg, 1)

    # get image
    im = np.array(Image.open("{}/{}.jpg".format(RGB_FOLDER, _id)).resize((128, 128)))
    out = np.array(Image.open("{}/{}.jpg".format(OUT_FOLDER, _id)).resize((128, 128)).convert('L'))
    im = np.concatenate([im, out[:, :, np.newaxis]], -1)
    
    # enumerate all actions
    print('reconstructing: {}'.format(_id))
    for i in range(corner_edge.shape[0]):

        inds = np.where((corner_edge[i, :] == 1) & (current_edges == 0))[0]
        for _ in inds:
            keep_track = (-1, -1)
            for k in inds:

                # draw current state
                im_s0 = draw_edges(current_edges, e_xys)

                # draw new state
                new_state = np.array(current_edges)
                new_state[k] = abs(1.0-new_state[k])
                im_s1 = draw_edges(new_state, e_xys)

                # predict next state
                im_sx = np.concatenate([im.transpose(2, 0, 1)/255.0, im_s0[np.newaxis, :, :], im_s1[np.newaxis, :, :]])
                with torch.no_grad():
                    _input = torch.from_numpy(im_sx).float().cuda().unsqueeze(0)
                    probs = _steps(_input)

                prob = probs[0][1]
                if prob > keep_track[0]:
                     keep_track = (prob, k)
                
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
            if keep_track[0] > .5:
                current_edges[keep_track[1]] = abs(1.0-current_edges[keep_track[1]])

    # draw final state
    im_path = os.path.join(RGB_FOLDER, _id + '.jpg')
    dwg = svgwrite.Drawing('svg/{}.svg'.format(_id), (256, 256))
    dwg.add(svgwrite.image.Image(im_path, size=(256, 256)))
    draw_relations(corners_det, edges_det, current_edges, corner_edge, _id)
    dwg.save()