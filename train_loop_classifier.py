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
from dataset.loop_dataloader import LoopData
from dataset.shuffle_dataloader import ShuffleData
from torch.utils.data import DataLoader
from model.graph_model import GraphModel
from dataset.collate import PadCollate
from utils.losses import balanced_binary_cross_entropy
import os
from collections import OrderedDict
import cv2
from tqdm import tqdm
from utils.utils import tileImages
from validator import Validator
from options import parse_args
# from model.resnet_loop import create_model, GraphModelCustom
from model.mlp_loop import create_model
from model.modules import findLoopsModule
from dataset.metrics import Metrics

# sample run
# python3 train_loop_classifier.py --data_path ~/Workspace/ --predicted_edges_path ~/Workspace/building_reconstruction/working_model/action_evaluation_master/cache/predicted_edges/annots_only

def main(options):

    ##############################################################################################################
    ############################################### Define Model #################################################
    ##############################################################################################################

    model = create_model(options)
    model = model.cuda()
    model = model.train()

    # select optimizer
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=options.LR)

    ##############################################################################################################
    ############################################# Setup Training #################################################
    ##############################################################################################################
    PREFIX = options.data_path
    with open('{}/dataset_atlanta/processed/train_list.txt'.format(PREFIX)) as f:
        train_list = [line.strip() for line in f.readlines()]
    with open('{}/dataset_atlanta/processed/valid_list.txt'.format(PREFIX)) as f:
        valid_list = [line.strip() for line in f.readlines()]

    dset_train = LoopData(options, train_list, split='train', num_edges=0, load_heatmaps=True)
    dset_val = LoopData(options, valid_list, split='val', num_edges=0, load_heatmaps=True)

    for epoch in range(20):

        dset_train.reset()
        train_loader = DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=4)    
        epoch_losses = []

        ## Same with train_loader but provide progress bar
        data_iterator = tqdm(train_loader, total=len(dset_train))
        optimizer.zero_grad()     

        for sample_index, sample in enumerate(data_iterator):
                
            im_arr, loop_gt, loop_edges, loop_corners, loop_accs, building_index = sample[0].squeeze(0).cuda().float(), sample[1].squeeze(0).cuda().float(), sample[2].squeeze(0), sample[3], sample[4], sample[5]
            # sample[0].cuda().squeeze(0).float()
            #im_arr = [x.cuda().float() for x in im_arr]

            loop_corners = [lp.cuda().squeeze(0).float()/255.0 for lp in loop_corners]
            building = dset_train.buildings[building_index]
            # p_inds = np.where(loop_gt.detach().cpu()==1)[0]
            # n_inds = np.where(loop_gt.detach().cpu()==0)[0]
            # print('pos: ', p_inds.shape)
            # print('neg: ', n_inds.shape)

            # for k in range(im_arr.shape[0]):
            #     lim = im_arr[k, 0, :, :].detach().cpu().numpy()
            #     rgb = im_arr[k, 1:, :, :].detach().cpu().numpy() *255.0
            #     rgb = np.transpose(rgb, (1, 2, 0)).astype('uint8')
                
            #     print(loop_gt[k])
            #     # print(rgb.shape)
            #     # print(lim.shape)
            #     inds = np.where(lim == 1)
            #     rgb[inds[0], inds[1], :] = [0, 0, 255]

            #     cv2.imshow('loop', lim)
            #     cv2.imshow('rgb', rgb)
            #     cv2.waitKey(0)
            
            loop_pred = model(im_arr) #model(im_arr, loop_corners)
            #loop_pred = torch.sigmoid(loop_pred).view(-1)

            weight = loop_gt * 4.0 + 1.0
            # print(loop_pred.shape)
            # print(loop_gt.shape)
            edge_loss = F.binary_cross_entropy(loop_pred, loop_gt, weight=weight)
            losses = [edge_loss]
            loss = sum(losses)        
            loss.backward()

            if (sample_index + 1) % options.batch_size == 0:
                loss_values = [l.data.item() for l in losses]
                epoch_losses.append(loss_values)
                status = str(epoch + 1) + ' loss: '
                for l in loss_values:
                    status += '%0.5f '%l
                    continue
                data_iterator.set_description(status)
                optimizer.step()
                optimizer.zero_grad()

        print('loss', np.array(epoch_losses).mean(0))
        torch.save(model.state_dict(), options.checkpoint_dir + '/loop_checkpoint_{}_epoch_{}.pth'.format(options.suffix, epoch))
        torch.save(optimizer.state_dict(), options.checkpoint_dir + '/loop_optim_{}_epoch_{}.pth'.format(options.suffix, epoch))


       
if __name__ == '__main__':
    args = parse_args()
    args.keyname = 'batch'
    args.keyname += '_' + args.corner_type
    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass
    if args.conv_type != '':
        args.keyname += '_' + args.conv_type
        pass    
    args.test_dir = 'test/' + args.keyname
    args.checkpoint_dir = 'checkpoint/' + args.keyname

    if not os.path.exists(args.checkpoint_dir):
        os.system("mkdir -p %s"%args.checkpoint_dir)
        pass
    if not os.path.exists(args.test_dir):
        os.system("mkdir -p %s"%args.test_dir)
        pass
    if not os.path.exists(args.test_dir + '/cache'):
        os.system("mkdir -p %s"%args.test_dir + '/cache')
        pass

    main(args)
