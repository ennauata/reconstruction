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
from model.graph import EdgeClassifier
from dataset.collate import PadCollate
from utils.losses import balanced_binary_cross_entropy
import os
from dataset.metrics import Metrics
from collections import OrderedDict

##############################################################################################################
############################################### Define Model #################################################
##############################################################################################################
edge_classifier = EdgeClassifier()
edge_classifier = edge_classifier.cuda()
edge_classifier = edge_classifier.train()
sample_eval = EdgeClassifier()
sample_eval = sample_eval.cpu()
sample_eval = sample_eval.eval()
##############################################################################################################
############################################# Setup Training #################################################
##############################################################################################################
PREFIX = '/home/nelson/Workspace'
LADATA_FOLDER = '{}/building_reconstruction/la_dataset_new/'.format(PREFIX)
ANNOTS_FOLDER = '{}/building_reconstruction/la_dataset_new/annots'.format(PREFIX)
#EDGES_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/dets/edges'
#CORNERS_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/dets/corners'
EDGES_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_dets/edges'.format(PREFIX)
CORNERS_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_dets/corners'.format(PREFIX)
with open('{}/building_reconstruction/la_dataset_new/train_list_prime.txt'.format(PREFIX)) as f:
    train_list = [line.strip() for line in f.readlines()][:-50]
with open('{}/building_reconstruction/la_dataset_new/valid_list.txt'.format(PREFIX)) as f:
    valid_list = [line.strip() for line in f.readlines()]

# select optimizer
params = list(edge_classifier.parameters())
optimizer = optim.Adam(params, lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
best_score = 0.0
mt = Metrics()

##############################################################################################################
############################################### Start Training ###############################################
##############################################################################################################

def _steps(im_arr, gt, bins):

    loss = 0.0
    probs, logits = edge_classifier(im_arr, bins)

    # computer edge loss
    ind = torch.nonzero(gt >= 0)
    if ind.size()[0] > 0:

        # print(gt.shape)
        logits = logits[ind.data].squeeze(1)
        target = gt[ind.data].squeeze(1)

        # weight = torch.ones_like(target).cuda()
        # pos = torch.nonzero(target==1)
        # weight[pos.data] = 10.0
        # print(torch.nonzero(target==1).shape)
        # print(torch.nonzero(target==0).shape)
        # print(torch.nonzero(target>1).shape)
        # print(torch.nonzero(target<0).shape)
        loss += F.cross_entropy(logits, target.long().cuda())#, weight=torch.tensor([1.0, 5.0]).cuda())

    return loss

def get_annots(_id):
    annot_path = os.path.join(ANNOTS_FOLDER, _id +'.npy')
    annot = np.load(open(annot_path, 'rb'), encoding='bytes')
    annot = OrderedDict(annot[()])
    return np.array(list(annot.keys()))

update_cnn_flag = False
for epoch in range(3000):

    # load cnn
    src = 'saved_models/*'
    if update_cnn_flag:
        list_of_files = glob.glob(src) 
        latest_file = max(list_of_files, key=os.path.getctime)
        sample_eval.load_state_dict(torch.load(latest_file))
        update_cnn_flag = False
        print('CNN Updated...')
        print(latest_file)

    # create loaders
    dset_train = GraphData(train_list, ANNOTS_FOLDER, EDGES_FOLDER, CORNERS_FOLDER, LADATA_FOLDER, sample_eval, with_augmentation=True)
    train_loader = DataLoader(dset_train, batch_size=64, shuffle=True, num_workers=1, collate_fn=PadCollate())

    # for each epoch
    scheduler.step()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        print('iter {}'.format(i))
        # get the inputs
        im_arr, gt, bins = data

        im_arr = Variable(im_arr.float())
        im_arr = im_arr.view(-1, 6, 256, 256)
        #inds = np.array([0, 1, 2, 4, 5])
        #im_arr = im_arr[:, inds, :, :]

        gt = gt.float().view(-1)
        bins = Variable(bins.float().view(-1, 288))

        # # filter padded
        # inds = torch.nonzero(gt >= 0)[0]
        # im_array = im_arr[inds.data]
        # gt = gt[inds.data]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = _steps(im_arr.cuda(), gt.cuda(), bins.cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss

    # print epoch loss
    print('[%d] lr: %.5f \nloss: %.5f' %
          (epoch + 1, optimizer.param_groups[0]['lr'], running_loss / len(train_loader)))

    if (epoch > 0) and (epoch % 5 == 0):
        torch.save(edge_classifier.state_dict(), './saved_models/edge_classifier_iter_{}.pth'.format(epoch)) 
        update_cnn_flag = True