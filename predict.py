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
from dataset.custom_dataloader_dynamic_4 import GraphData
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
import torchvision.models as models

def create_png(svg, target):
    lines = open(svg).readlines()[1:][0]
    svg2png(bytestring=lines, write_to=target)

def fit_edge(mask):

    idx = np.where(mask > 0)
    y = idx[0][:, np.newaxis].astype('float')
    X = idx[1][:, np.newaxis].astype('float')
    # dx = np.array([np.min(X), np.max(X)])[:, np.newaxis]
    # ransac = linear_model.RANSACRegressor()
    # ransac.fit(X, y)
    # print(ransac.estimator_.coef_)
    # print(ransac.estimator_.intercept_)
    # dy = ransac.predict(dx)
    dx = np.array([X[0], X[-1]])
    dy = np.array([y[0], y[-1]])
    return dx, dy

##############################################################################################################
############################################### Define Model #################################################
##############################################################################################################

edge_classifier = EdgeClassifier()
edge_classifier = edge_classifier.cuda()
edge_classifier = edge_classifier.eval()

epoch = 1000
src = 'saved_models'
edge_classifier.load_state_dict(torch.load('./{}/edge_classifier_iter_{}.pth'.format(src, epoch)))


###############################
###############################################################################
############################################# Setup Training #################################################
##############################################################################################################
PREFIX = '/home/nelson/Workspace'
LADATA_FOLDER = '{}/building_reconstruction/la_dataset_new/'.format(PREFIX)
RGB_FOLDER = '{}/building_reconstruction/la_dataset_new/rgb'.format(PREFIX)
ANNOTS_FOLDER = '{}/building_reconstruction/la_dataset_new/annots'.format(PREFIX)
EDGES_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_dets/edges'.format(PREFIX)
CORNERS_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_dets/corners'.format(PREFIX)
#EDGES_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/expanded_primitives_dets/edges'
#CORNERS_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/expanded_primitives_dets/corners'
with open('{}/building_reconstruction/la_dataset_new/train_list_prime.txt'.format(PREFIX)) as f:
    valid_list = [line.strip() for line in f.readlines()][-50:]

# create loaders
dset_valid = GraphData(valid_list, ANNOTS_FOLDER, EDGES_FOLDER, CORNERS_FOLDER, LADATA_FOLDER, with_augmentation=False)
valid_loader = DataLoader(dset_valid, batch_size=1, shuffle=False, num_workers=1, collate_fn=PadCollate())

##############################################################################################################
############################################### Start Training ###############################################
##############################################################################################################

def _steps(im_arr, gt, bins):
    probs, logits = edge_classifier(im_arr, bins)
    return probs
                
def get_annots(_id):
    annot_path = os.path.join(ANNOTS_FOLDER, _id +'.npy')
    annot = np.load(open(annot_path, 'rb'), encoding='bytes')
    annot = OrderedDict(annot[()])
    return np.array(list(annot.keys())), annot


# Each epoch has a training and validation phase
fract = 0.0
preds = []
targets = []
for k, data in enumerate(valid_loader):

    # get the inputs
    im_arr, gt, bins = data
    im_arr = Variable(im_arr.float())
    im_arr = im_arr.view(-1, 6, 256, 256)
    #inds = np.array([0, 1, 2, 4, 5])
    #im_arr = im_arr[:, inds, :, :]

    gt = gt.float().view(-1)
    bins = Variable(bins.float().view(-1, 288))

    # forward
    with torch.no_grad():
        pred = _steps(im_arr.cuda(), gt.cuda(), bins.cuda())

    preds.append(pred)
    targets.append(gt)

    for n, im in enumerate(im_arr):
        if gt[n] != np.argmax(pred[n]).float():
            im = np.array(im).transpose(1, 2, 0)
            im1 = im[:, :, :3]*255.0
            im2 = im[:, :, :3]*255.0
            im3 = im[:, :, :3]*255.0

            inds = np.array(np.where(im[:, :, 3]>0))
            if inds.shape[1] > 0:
                im1[inds[0], inds[1], 0] = 255.0
                im1[inds[0], inds[1], 1] = 0.0
                im1[inds[0], inds[1], 2] = 0.0
            inds = np.array(np.where(im[:, :, 4]>0))
            if inds.shape[1] > 0:
                im2[inds[0], inds[1], 0] = 255.0
                im2[inds[0], inds[1], 1] = 0.0
                im2[inds[0], inds[1], 2] = 0.0
            inds = np.array(np.where(im[:, :, 5]>0))
            if inds.shape[1] > 0:
                im3[inds[0], inds[1], 0] = 255.0
                im3[inds[0], inds[1], 1] = 0.0
                im3[inds[0], inds[1], 2] = 0.0

            im1 = Image.fromarray(im1.astype('uint8'))
            im2 = Image.fromarray(im2.astype('uint8'))
            im3 = Image.fromarray(im3.astype('uint8'))

            pred_path = './visualization/{}_predicted.jpg'.format(n)
            correct_path = './visualization/{}_correct.jpg'.format(n)
            if np.argmax(pred[n]) == 0:
                im2.save(pred_path)
                im3.save(correct_path)
            else:
                im3.save(correct_path)
                im2.save(pred_path)
            print(np.argmax(pred[n]), gt[n])

            # plt.figure()
            # plt.imshow(im1)
            # plt.figure()
            # plt.imshow(im2)
            # plt.figure()
            # plt.imshow(im3)
            # plt.show()

preds = np.concatenate(preds, 0)
preds = np.argmax(preds, -1)

#preds = np.random.randint(0, 3, preds.shape)
targets = np.concatenate(targets, 0)

# filter current state
inds = np.where(targets == 0)
t0 = targets[inds]
p0 = preds[inds]
h0 = np.sum(p0 == t0)

inds = np.where(targets == 1)
t1 = targets[inds]
p1 = preds[inds]
h1 = np.sum(p1 == t1)

# inds = np.where(targets == 2)
# t2 = targets[inds]
# p2 = preds[inds]
# h2 = np.sum(p2 == t2)
print(t0.shape[0], t1.shape[0])
print(h0/t0.shape[0])
print(h1/t1.shape[0])
#print(h2/t2.shape[0])