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
from dataset.custom_dataloader_dynamic import GraphData
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
    train_list = [line.strip() for line in f.readlines()]
with open('{}/building_reconstruction/la_dataset_new/valid_list.txt'.format(PREFIX)) as f:
    valid_list = [line.strip() for line in f.readlines()]

# create loaders
dset_train = GraphData(train_list, ANNOTS_FOLDER, EDGES_FOLDER, CORNERS_FOLDER, LADATA_FOLDER, with_augmentation=True)
dset_valid = GraphData(valid_list, ANNOTS_FOLDER, EDGES_FOLDER, CORNERS_FOLDER, LADATA_FOLDER, with_augmentation=False)
train_loader = DataLoader(dset_train, batch_size=4, shuffle=True, num_workers=1, collate_fn=PadCollate())
valid_loader = DataLoader(dset_valid, batch_size=1, shuffle=False, num_workers=1, collate_fn=PadCollate())
dset_list = {'train': dset_train, 'val': dset_valid}
dset_loader = {'train': train_loader, 'val': valid_loader}

# select optimizer
params = list(edge_classifier.parameters())
optimizer = optim.Adam(params, lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
best_score = 0.0
mt = Metrics()

##############################################################################################################
############################################### Start Training ###############################################
##############################################################################################################

def _steps(im_arr, gt):

    loss = 0.0
    probs, logits = edge_classifier(im_arr)

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

def hard_sample_mining(im_arr, gt, epoch, step=10, src='saved_models', last_n=128):

	sample_eval = EdgeClassifier()
	sample_eval = sample_eval.cuda()
	sample_eval = sample_eval.eval()
	ref = max(0, epoch-1)
	sample_eval.load_state_dict(torch.load('./{}/edge_classifier_iter_{}.pth'.format(src, step*(ref//step))))

	# forward
	inds = torch.nonzero(gt >= 0).squeeze(-1)
	im_arr_slice = im_arr[inds.data]
	gt_slice = gt[inds.data]
	with torch.no_grad():
	    probs, logits = sample_eval(im_arr_slice.cuda())
	    probs_1 = probs[:, 1]
	    diff = torch.abs(probs_1-gt_slice.cuda())
	    vals, inds = torch.sort(diff)
	    inds = inds[-last_n:]
	    im_arr_slice = im_arr_slice[inds.data]
	    gt_slice = gt_slice[inds.data]
	   
	return im_arr_slice, gt_slice

def get_annots(_id):
    annot_path = os.path.join(ANNOTS_FOLDER, _id +'.npy')
    annot = np.load(open(annot_path, 'rb'), encoding='bytes')
    annot = OrderedDict(annot[()])
    return np.array(list(annot.keys()))

for epoch in range(3000):

    # Each epoch has a training and validation phase
    for phase in ['train']:

        scheduler.step()
        running_loss = 0.0
        for i, data in enumerate(dset_loader[phase]):

            # get the inputs
            im_arr, gt = data
            im_arr = Variable(im_arr.float())
            im_arr = im_arr.view(-1, 6, 128, 128)
            gt = gt.float().view(-1)

            # im_arr, gt = hard_sample_mining(im_arr, gt, epoch)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = _steps(im_arr.cuda(), gt.cuda())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss

        # print epoch loss
        print('[%d] %s lr: %.5f \nloss: %.5f' %
              (epoch + 1, phase, optimizer.param_groups[0]['lr'], running_loss / len(dset_loader[phase])))

    if epoch % 10 == 0:
        torch.save(edge_classifier.state_dict(), './saved_models/edge_classifier_iter_{}.pth'.format(epoch)) 