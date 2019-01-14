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
from dataset.custom_dataloader import GraphData
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
from model.graph import Graph, Encoder, Decoder, BipartiteGraph
from sklearn.metrics import precision_score, recall_score

## HELPER FUNCTIONS ##
def rotate_coords(image_shape, xy, angle):
    org_center = (image_shape-1)/2.
    rot_center = (image_shape-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a), -org[0]*np.sin(a) + org[1]*np.cos(a)])
    return new+rot_center

def get_annots(_id):
    annot_path = os.path.join(ANNOTS_FOLDER, _id +'.npy')
    annot = np.load(open(annot_path, 'rb'), encoding='bytes')
    annot = OrderedDict(annot[()])
    return np.array(list(annot.keys())), annot

##############################################################################################################
############################################### Define Model #################################################
##############################################################################################################

enc = Encoder()
dec = Decoder()
corner_edge_net = BipartiteGraph()

corner_edge_net = corner_edge_net.cuda()
dec = dec.cuda()
enc = enc.cuda()

corner_edge_net = corner_edge_net.eval()
enc = enc.eval()
dec = dec.eval()

epoch = 240
src = 'saved_models'
#src = 'checkpoints/dets+gt/splitted_edges_dist_16'
corner_edge_net.load_state_dict(torch.load('./{}/corner_edge_net_iter_{}.pth'.format(src, epoch)))
dec.load_state_dict(torch.load('./{}/decoder_iter_{}.pth'.format(src, epoch)))
enc.load_state_dict(torch.load('./{}/encoder_iter_{}.pth'.format(src, epoch)))

##############################################################################################################
############################################# Setup Training #################################################
##############################################################################################################

RGB_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset_new/rgb'
ANNOTS_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset_new/annots'
# EDGES_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset_new/expanded_primitives_gt_only/edges'
# CORNERS_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/expanded_primitives_gt_only/corners'
EDGES_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset_new/expanded_primitives_dets/edges'
CORNERS_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset_new/expanded_primitives_dets/corners'
with open('/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset_new/valid_list.txt') as f:
    valid_list = [line.strip() for line in f.readlines()]

# create loaders
dset_valid = GraphData(valid_list, ANNOTS_FOLDER, EDGES_FOLDER, CORNERS_FOLDER, with_augmentation=False, with_filter=True)
valid_loader = DataLoader(dset_valid, batch_size=1, shuffle=False, num_workers=1, collate_fn=PadCollate())

##############################################################################################################
############################################### Start Training ###############################################
##############################################################################################################

def _steps(V1, V2, V1_c, V2_c, corner_edge_net, cr_mat, c_xys, e_xys, cea_bs, T=1):    
    E12 = Variable(torch.zeros((V1.shape[0], V1.shape[1], V2.shape[1], 128))).float()
    V1_pred = Variable(torch.ones(*V1.shape[:2])).float()
    V2_pred = Variable(torch.ones(*V2.shape[:2])).float()
    V1, V2 = enc(V1.cuda(), V2.cuda(), V1_c.cuda(), V2_c.cuda(), c_xys.cuda(), e_xys.cuda())
    for _ in range(T):

        # forward + backward + optimize
        V1, V2, E12, V1_pred, V2_pred = corner_edge_net(V1.cuda(), V2.cuda(), E12.cuda(), c_xys.cuda(), e_xys.cuda(), cr_mat.cuda(), V1_pred.cuda(), V2_pred.cuda(), dist_3.cuda(), cea_bs.cuda())
    return V2_pred    

def get_annots(_id):
    annot_path = os.path.join(ANNOTS_FOLDER, _id +'.npy')
    annot = np.load(open(annot_path, 'rb'), encoding='bytes')
    annot = OrderedDict(annot[()])
    return np.array(list(annot.keys())), annot

def isHit(e1, e2, thresh=4.0):
    
    y1, x1, y2, x2 = e1
    y3, x3, y4, x4 = e2

    # make order
    d1 = np.sqrt((y3-y1)**2 + (x3-x1)**2)
    d2 = np.sqrt((y4-y1)**2 + (x4-x1)**2)
    d3 = np.sqrt((y3-y2)**2 + (x3-x2)**2)
    d4 = np.sqrt((y4-y2)**2 + (x4-x2)**2)

    hit = False
    if ((d1 < thresh) and (d4 < thresh)) or ((d2 < thresh) and (d3 < thresh)):
        hit = True
    dist = min(d1+d4, d2+d3)

    # im = np.zeros((256, 256, 3)).astype('uint8')
    # im = Image.fromarray(im)
    # draw = ImageDraw.Draw(im)
    # draw.line((x1, y1, x2, y2), fill='blue')
    # draw.line((x3, y3, x4, y4), fill='red')
    # print(hit)
    # print(d1, d2, d3, d4)
    # plt.imshow(im)
    # plt.show()

    return hit, dist



def compute_metric(annots, pred, coords):
    
    pos = np.where(pred>0.5)
    coords = coords[pos]

    # grab all edges
    edge_set = set()
    for c1 in annots:
        for c2 in annots[c1]:
            
            # make an order
            x1, y1 = c1
            x2, y2 = c2
            if x1 > x2:
                x1, x2, y1, y2 = x2, x1, y2, y1  # swap
            elif x1 == x2 and y1 > y2:
                x1, x2, y1, y2 = x2, x1, y2, y1  # swap
            else:
                pass

            edge = (y1, x1, y2, x2)
            edge_set.add(edge)

    # compute scores
    tp = 0
    fp = 0
    pos = len(edge_set)
    keep_track = []

    # for each detection
    for e1 in coords:
        y1, x1, y2, x2 = e1
        found = 0
        closest = 99999
        edge_found = None

        # grab closest edge
        for e2 in edge_set:
            y3, x3, y4, x4 = e2
            hit, dist = isHit((y1, x1, y2, x2), (y3, x3, y4, x4))
            if hit == True and dist < closest:
                closest = dist
                edge_found = e2

        # check if closest edge is a true detection
        if (edge_found is not None) and (edge_found not in keep_track):
            keep_track.append(edge_found)
            tp += 1
        else:
            fp += 1

    return tp, fp, pos

# each epoch has a training and validation phase
gt_percentage = []
gt_complete = []
gt_edges = []
pred_edges = []
true_pos = []
false_pos = []
pos = []
neg = []
for k, data in enumerate(valid_loader):

    # get the inputs
    _id = valid_list[k]
    embs_c, embs_e, gt_c, gt_e, dets_c, dets_e, cr_mat, e_xys, dist_3, cea_bs, ca_gt = data
    embs_c, embs_e, gt_c, gt_e = Variable(embs_c.float()), Variable(embs_e.float()), Variable(gt_c.float()), Variable(gt_e.float())
    c_coords, e_coords = Variable(dets_c.float()), Variable(dets_e.float())
    cr_mat = cr_mat.float()
    e_xys = e_xys.float()
    cea_bs = cea_bs.float()
    ca_gt = ca_gt.float()
    dist_3 = dist_3.float()

    # forward
    V1 = embs_c
    V2 = embs_e
    V1_c = c_coords
    V2_c = e_coords
    V1_c = torch.cat([c_coords, c_coords**2], -1)
    V2_c = torch.cat([e_coords, e_coords**2], -1)

    x, y, _, _ = torch.chunk(c_coords, 4, -1)
    scale = 64.0
    c_xys = torch.cat([x*scale, y*scale], -1).long()

    # forward
    with torch.no_grad():
        ce_t = _steps(V1, V2, V1_c, V2_c, corner_edge_net, cr_mat, c_xys, e_xys, cea_bs)

    # load data
    dets_annot, edges_annots = np.array(get_annots(_id))

    # compute metrics
    tp, fp, p = compute_metric(edges_annots, ce_t, e_coords*256.0)
    true_pos.append(tp)
    false_pos.append(fp)
    pos.append(p)
    print(tp/p, tp/(tp+fp+1e-10))
    # edges_dets_gt = gt_e
    # pos_edges_gt = np.where(edges_dets_gt > 0)[-1]

    # gt_percentage.append(pos_edges_gt.shape[0]/float(edges_annots.shape[0]))
    # gt_complete.append([1 if pos_edges_gt.shape[0] >= edges_annots.shape[0] - 3 else 0])

    # gt_edges.append(edges_dets_gt.numpy().ravel())
    # ce_t[ce_t>=.5] = 1
    # ce_t[ce_t<.5] = 0
    # pred_edges.append(ce_t.cpu().numpy().ravel())
    # print(edges_annots.shape)
    # print(pos_edges_gt.shape)
    # pos_edges_gt_with_filter = edges_dets_gt_no_filter[edges_filter]
    # print(pos_edges_gt.shape)
    # print(pos_edges_gt_with_filter.shape)

# gt_edges = np.concatenate(gt_edges, -1)
# pred_edges = np.concatenate(pred_edges, -1)
# hits = (pred_edges == gt_edges)
# tp = ((gt_edges == np.ones_like(gt_edges)) and (pred_edges == np.ones_like(gt_edges)))
# P = (gt_edges == np.ones_like(gt_edges))
recall = np.sum(true_pos)/np.sum(pos)
precision = np.sum(true_pos)/(np.sum(true_pos)+np.sum(false_pos))

print('recall: {}'.format(recall))
print('precision: {}'.format(precision))
