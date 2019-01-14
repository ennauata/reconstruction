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
from model.regressor import DistanceRegressor
import time

##############################################################################################################
############################################### Define Model #################################################
##############################################################################################################

edge_classifier = EdgeClassifier()
edge_classifier = edge_classifier.cuda()
edge_classifier = edge_classifier.eval()

distance_regressor = DistanceRegressor()
distance_regressor = distance_regressor.cuda()
distance_regressor = distance_regressor.eval()

epoch = 240
src = 'saved_models'
edge_classifier.load_state_dict(torch.load('./{}/edge_classifier_iter_{}.pth'.format(src, epoch)))

epoch = 240
src = 'saved_models'
distance_regressor.load_state_dict(torch.load('../distance_regression/{}/edge_classifier_iter_{}.pth'.format(src, epoch)))

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
    count = 0
    corners_det, edges_det, corner_edge, e_xys, ini_edges = data
    im = np.array(Image.open("{}/{}.jpg".format(RGB_FOLDER, _id)).resize((128, 128)))
    out = np.array(Image.open("{}/{}.jpg".format(OUT_FOLDER, _id)).resize((128, 128)).convert('L'))
    im_comb = np.concatenate([im, out[:, :, np.newaxis]], -1)

    # format input
    corners_det = corners_det.squeeze(0)
    edges_det = edges_det.squeeze(0)
    corner_edge = np.array(corner_edge.squeeze(0))
    e_xys = e_xys.squeeze(0)

    # init states
    init_current_corner = 0
    init_current_edges = np.array(ini_edges.view(-1)) #np.zeros(corner_edge.shape[1])

    # start timer
    start = time.time()

    # enumerate all actions
    best_dist = 99999
    print('reconstructing: {}'.format(_id))
    keep_track = []
    states = [(init_current_corner, init_current_edges)]
    while len(states) > 0:

        # retrieve next state
        current_corner, current_edges = states.pop()

        # check timer
        end = time.time()
        if end - start > 600:
            break

        # exit check
        if current_corner < corner_edge.shape[0]:

            # find next states
            inds = np.where(corner_edge[current_corner, :] == 1)[0]# & (current_edges == 0))[0]
            for k in inds:

                # draw current state
                im_s0 = draw_edges(current_edges, e_xys)

                # draw new state
                new_state = np.array(current_edges)
                new_state[k] = abs(1.0-new_state[k])
                im_s1 = draw_edges(new_state, e_xys)

                # predict next state
                im_sx = np.concatenate([im_comb.transpose(2, 0, 1)/255.0, im_s0[np.newaxis, :, :], im_s1[np.newaxis, :, :]])
                with torch.no_grad():
                    _input = torch.from_numpy(im_sx).float().cuda().unsqueeze(0)
                    probs = _steps(_input)
                    prob = probs[0][1]
                
                # # Debug
                # im0 = Image.fromarray((im_s0>0)*255.0)
                # im1 = Image.fromarray((im_s1>0)*255.0)
                # print(prob)
                # plt.figure()
                # plt.imshow(im0)
                # plt.figure()
                # plt.imshow(im1)
                # plt.figure()
                # plt.imshow(Image.fromarray(im))
                # plt.figure()
                # plt.imshow(Image.fromarray(out))
                # plt.show() 

                # append next state
                if prob > 0.5:
                    to_add = (current_corner, new_state)
                    states.append(to_add)

            to_add = (current_corner+1, np.array(current_edges))
            states.append(to_add)

        # draw final state
        else:

            # get corners degree roughly
            inds = np.where(corner_edge==1)
            deg = np.zeros_like(corner_edge)
            deg[inds[0], inds[1]] = current_edges[inds[1]]
            deg = np.sum(deg, 1)
            inds = np.where(deg == 1)[0]

            # im_s = draw_edges(current_edges, e_xys)
            # im_curr = torch.tensor(np.concatenate([im.transpose(2, 0, 1)/255.0, im_s[np.newaxis, :, :]])).float().cuda()
            # curr_dist = distance_regressor(im_curr.unsqueeze(0))
            # curr_dist = curr_dist.cpu()[0][0]

            # if curr_dist < best_dist:
            #best_dist = curr_dist

            if (inds.shape[0] == 0) and (tuple(current_edges) not in keep_track):

                keep_track.append(tuple(current_edges))
                dest = "svg/{}".format(_id)
                if not os.path.exists(dest):
                    os.makedirs(dest)
                im_path = os.path.join(RGB_FOLDER, _id + '.jpg')
                dwg = svgwrite.Drawing('svg/{}/inst_{}.svg'.format(_id, count), (128, 128))
                dwg.add(svgwrite.image.Image(im_path, size=(128, 128)))
                draw_relations(corners_det/2.0, edges_det/2.0, current_edges, corner_edge, _id)
                dwg.save() 
                count+=1