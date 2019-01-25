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
from dataset.incremental_dataloader import GraphData
from torch.utils.data import DataLoader
from model.graph import EdgeClassifier
from model.resnet import CustomResNet
from dataset.collate import PadCollate
from utils.losses import balanced_binary_cross_entropy
import os
from dataset.metrics import Metrics
from collections import OrderedDict
import cv2
from tqdm import tqdm
from utils.utils import tileImages
from searcher import Searcher

import argparse

parser = argparse.ArgumentParser(description='PlaneFlow')

parser.add_argument('--data_path', dest='data_path',
                    help='path to data',
                    default='.', type=str)
parser.add_argument('--corner_type', dest='corner_type',
                    help='corner type for the search algorithm',
                    default='annots_only', type=str)
parser.add_argument('--batch_size', dest='batch_size',
                    help='batch size',
                    default=64, type=int)
parser.add_argument('--num_edges', dest='num_edges',
                    help='num edges',
                    default=0, type=int)
parser.add_argument('--num_training_images', dest='num_training_images',
                    help='num training images',
                    default=10000, type=int)
parser.add_argument('--num_testing_images', dest='num_testing_images',
                    help='num testing images',
                    default=200, type=int)
parser.add_argument('--restore', dest='restore',
                    help='restore',
                    default=1, type=int)
parser.add_argument('--LR', dest='LR',
                    help='learning rate',
                    default=1e-3, type=float)
parser.add_argument('--task', dest='task',
                    help='task',
                    default='train', type=str)
parser.add_argument('--max_num_edges', dest='max_num_edges',
                    help='maximum number of_edges',
                    default=20, type=int)
parser.add_argument('--suffix', dest='suffix',
                    help='the suffix to distinguish experiments with different configurations',
                    default='', type=str)


options = parser.parse_args()
options.keyname = 'incremental'
options.keyname += '_' + options.corner_type
if options.suffix != '':
    options.keyname += '_' + options.suffix
    pass
options.test_dir = 'test/' + options.keyname
options.checkpoint_dir = 'checkpoint/' + options.keyname

if not os.path.exists(options.checkpoint_dir):
    os.system("mkdir -p %s"%options.checkpoint_dir)
    pass
if not os.path.exists(options.test_dir):
    os.system("mkdir -p %s"%options.test_dir)
    pass


def testOneEpoch(options, edge_classifier, dataset, num_edges):
    #edge_classifier.eval()
    
    epoch_losses = []
    ## Same with train_loader but provide progress bar
    data_loader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False, num_workers=1)    
    data_iterator = tqdm(data_loader, total=int(np.ceil(float(len(dataset)) / options.batch_size)))
    statistics = np.zeros(4)
    for sample_index, sample in enumerate(data_iterator):
        im_arr, label_gt, IDs = sample[0].cuda().squeeze(1), sample[1].cuda().squeeze(1), sample[2]
        
        logits = edge_classifier(im_arr)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits, label_gt)
        ## Progress bar
        loss_values = [loss.data.item()]
        epoch_losses.append(loss_values)
        status = 'val loss: '
        for l in loss_values:
            status += '%0.5f '%l
            continue
        data_iterator.set_description(status)

        label_pred = probs.max(-1)[1]
        label_gt = label_gt.detach().cpu().numpy()
        label_pred = label_pred.detach().cpu().numpy()
        
        statistics[0] += np.logical_and(label_pred == 1, label_gt == 1).sum()
        statistics[1] += np.logical_and(label_pred == 0, label_gt == 1).sum()                  
        statistics[2] += np.logical_and(label_pred == 1, label_gt == 0).sum()                  
        statistics[3] += np.logical_and(label_pred == 0, label_gt == 0).sum()                  
        #statistics[4] += len(label_pred)
        
        if sample_index % 500 == 0:
            im_arr = im_arr.detach().cpu().numpy()
            probs = probs.detach().cpu().numpy()

            index_offset = 0
            for batch_index in range(len(im_arr)):
                if label_pred[batch_index] != label_gt[batch_index]:
                    #print(index_offset, IDs[batch_index])
                    visualizeExample(options, im_arr[batch_index], label_pred[batch_index], label_gt[batch_index], confidence=probs[batch_index][label_pred[batch_index]], prefix=str(num_edges) + '_val_', suffix='_' + str(index_offset))
                    index_offset += 1
                    pass
                continue
            pass
        continue
    print('statistics', statistics[0] / (statistics[0] + statistics[1]), statistics[3] / (statistics[2] + statistics[3]))
    print('validation loss', np.array(epoch_losses).mean(0))
    edge_classifier.train()
    return

def visualizeExample(options, im, label_pred, label_gt, confidence=1, prefix='', suffix=''):
    image = (im[:3] * 255).transpose((1, 2, 0)).astype(np.uint8)
    image[im[3] > 0.5] = np.array([255, 0, 0], dtype=np.uint8)
    left_image = image.copy()
    left_image[im[4] > 0.5] = np.array([0, 0, 255])
    right_image = image.copy()
    right_image[im[5] > 0.5] = np.array([0, 0, 255])
    if label_pred == 1:
        images = [left_image, right_image]
    else:
        images = [right_image, left_image]
        pass
    if label_pred == label_gt:
        #background_color = np.array([0, 0, 0], dtype=np.uint8)
        background_color = int(round(confidence * 255))
    else:
        background_color = np.array([0, 0, int(round(255 * confidence))], dtype=np.uint8)
        pass
    image = tileImages([images], background_color=background_color)
    cv2.imwrite(options.test_dir + '/' + prefix + 'image' + suffix + '.png', image)
    return


##############################################################################################################
############################################### Define Model #################################################
##############################################################################################################
#edge_classifier = EdgeClassifier()
edge_classifier = CustomResNet(pretrained=options.restore == 0, num_classes=2)
edge_classifier = edge_classifier.cuda()
edge_classifier = edge_classifier.train()

# select optimizer
params = list(edge_classifier.parameters())
optimizer = optim.Adam(params, lr=options.LR)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# sample_eval = EdgeClassifier()
# sample_eval = sample_eval.cpu()
# sample_eval = sample_eval.eval()
##############################################################################################################
############################################# Setup Training #################################################
##############################################################################################################
PREFIX = options.data_path
with open('{}/building_reconstruction/la_dataset_new/train_list_prime.txt'.format(PREFIX)) as f:
    file_list = [line.strip() for line in f.readlines()]
    train_list = file_list[:-50]
    valid_list = file_list[-50:]
    pass
#with open('{}/building_reconstruction/la_dataset_new/valid_list.txt'.format(PREFIX)) as f:
#valid_list = [line.strip() for line in f.readlines()]

best_score = 0.0
mt = Metrics()

##############################################################################################################
############################################### Start Training ###############################################
##############################################################################################################

if options.num_edges >= 0:
    all_num_edges = [options.num_edges]
else:
    all_num_edges = range(options.max_num_edges)
    pass

## Train incrementally
for num_edges in all_num_edges:
    if options.restore == 1:
        edge_classifier.load_state_dict(torch.load(options.checkpoint_dir + '/' + str(num_edges) + '_checkpoint.pth'))
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/' + str(num_edges) + '_optim.pth'))
        pass
    
    dset_val = GraphData(options, valid_list, split='val', num_edges=-1 if 'mixed' in options.suffix else num_edges)
    
    if options.task == 'test':
        testOneEpoch(options, edge_classifier, dset_val, num_edges=-1 if 'mixed' in options.suffix else num_edges)
        exit(1)
        pass
    
    dset_train = GraphData(options, train_list, num_edges=num_edges)
    #train_loader = DataLoader(dset_train, batch_size=64, shuffle=True, num_workers=1, collate_fn=PadCollate())

    if options.task == 'search':
        searcher = Searcher(options)
        for split, dataset in [('train', dset_train), ('val', dset_val)]:
            all_statistics = np.zeros(options.max_num_edges + 1)
            for building in dataset.buildings:
                statistics = searcher.search(building, num_edges_target=num_edges + 1)
                all_statistics += statistics
                if statistics.sum() < 1:
                    print(building._id, statistics.sum(), building.num_edges, building.num_edges_gt)
                    pass
                building.save()
                continue
            print(split, all_statistics / len(dataset.buildings))
            continue
        exit(1)
        pass

    
    for epoch in range(20):
        os.system('rm ' + options.test_dir + '/' + str(num_edges) + '_*')
        dset_train.reset()
        train_loader = DataLoader(dset_train, batch_size=options.batch_size, shuffle=True, num_workers=16)    
        epoch_losses = []
        ## Same with train_loader but provide progress bar
        data_iterator = tqdm(train_loader, total=int(np.ceil(float(len(dset_train)) / options.batch_size)))
        for sample_index, sample in enumerate(data_iterator):
            optimizer.zero_grad()

            im_arr, label_gt = sample[0].cuda().squeeze(1), sample[1].cuda().squeeze(1)
            logits = edge_classifier(im_arr)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            loss = F.cross_entropy(logits, label_gt)

            ## Progress bar
            loss_values = [loss.data.item()]
            epoch_losses.append(loss_values)
            status = str(epoch + 1) + ' loss: '
            for l in loss_values:
                status += '%0.5f '%l
                continue
            data_iterator.set_description(status)

            loss.backward()
            optimizer.step()

            if sample_index % 500 == 0:
                label_pred = probs.max(-1)[1]
                im_arr = im_arr.detach().cpu().numpy()
                label_gt = label_gt.detach().cpu().numpy()
                label_pred = label_pred.detach().cpu().numpy()
                probs = probs.detach().cpu().numpy()

                index_offset = 0
                for batch_index in range(len(im_arr)):
                    if label_pred[batch_index] != label_gt[batch_index]:
                        visualizeExample(options, im_arr[batch_index], label_pred[batch_index], label_gt[batch_index], confidence=probs[batch_index][label_pred[batch_index]], prefix=str(num_edges) + '_', suffix='_' + str(index_offset))
                        index_offset += 1
                        pass
                    continue
                pass
            continue

        print('loss', np.array(epoch_losses).mean(0))
        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint_' + str(epoch // 10) + '.pth')
        #     torch.save(semantic_model.state_dict(), options.checkpoint_dir + '/checkpoint_semantic_' + str(epoch // 10) + '.pth')                
        #     #torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim_' + str(epoch // 10) + '.pth')
        #     pass
        torch.save(edge_classifier.state_dict(), options.checkpoint_dir + '/' + str(num_edges) + '_checkpoint.pth')
        torch.save(optimizer.state_dict(), options.checkpoint_dir + '/' + str(num_edges) + '_optim.pth')
        testOneEpoch(options, edge_classifier, dset_val, num_edges)        
        continue

    searcher = Searcher(options)
    for split, dataset in [('train', dset_train), ('val', dset_val)]:
        all_statistics = np.zeros(options.max_num_edges + 1)
        for building in dataset.buildings:
            statistics = searcher.search(building, num_edges_target=num_edges + 1)
            all_statistics += statistics
            if statistics.sum() < 1:
                print(building._id, statistics.sum(), building.num_edges, building.num_edges_gt)
                pass
            building.save()
            continue
        print(split, all_statistics / len(dataset.buildings))
        continue
    
    continue
