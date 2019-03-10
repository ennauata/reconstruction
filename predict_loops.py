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
from model.mlp_loop import create_model
from model.modules import findLoopsModule
from dataset.metrics import Metrics
from dataset.building import draw_edges
from utils.utils import loop_nms

# sample run
# python3 predict_loops.py --data_path ~/Workspace/ --predicted_edges_path ~/Workspace/building_reconstruction/working_model/action_evaluation_master/cache/predicted_edges/annots_only --testing_corner_type annots_only
def main(options):

    ##############################################################################################################
    ############################################### Define Model #################################################
    ##############################################################################################################

    model = create_model(options)
    model = model.cuda()
    model = model.train()
    epoch = 19
    model.load_state_dict(torch.load(options.checkpoint_dir + '/loop_checkpoint_{}_epoch_{}.pth'.format(options.suffix, epoch)))

    ##############################################################################################################
    ############################################# Setup Training #################################################
    ##############################################################################################################
    PREFIX = options.data_path
    with open('{}/dataset_atlanta/processed/valid_list.txt'.format(PREFIX)) as f:
        valid_list = [line.strip() for line in f.readlines()]
    dset_val = LoopData(options, valid_list, split='val', num_edges=0, load_heatmaps=True)
    data_iterator = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=1)    
    epoch_losses = []

    ## Same with train_loader but provide progress bar
    #data_iterator = tqdm(loader, total=len(dset_val))
    all_images = []
    row_images = []
    for sample_index, sample in enumerate(data_iterator):
            
        im_arr, loop_gt, edges_loops, loop_corners, _, building_index = sample[0].squeeze(0).cuda().float(), sample[1].squeeze(0).cuda().float(), sample[2].squeeze(0), sample[3], sample[4].squeeze(0), sample[5]
        #im_arr = [x.cuda().float() for x in im_arr]

        loop_corners_nms = [x.cuda().squeeze(0).float() for x in loop_corners]
        loop_corners_norm = [x.cuda().squeeze(0).float()/255.0 for x in loop_corners]  
        edges_loop_baseline = np.zeros(edges_loops.shape[-1])
        edges_loop_cls = np.zeros(edges_loops.shape[-1])
        loop_accs = np.zeros(edges_loops.shape[0])
        building = dset_val.buildings[building_index]
        rgb = np.array(Image.open("{}/rgb/{}.jpg".format(building.dataset_folder, building._id)))
        with np.load('{}/{}_0_False.npz'.format(building.options.predicted_edges_path, building._id)) as data:
            edges_baseline = data['arr_0']

        with torch.no_grad():
            
            loop_pred = model(im_arr)
            edges_loops = np.array([x.detach().cpu().numpy() for x in edges_loops])
            loop_corners_nms = [x.detach().cpu().numpy() for x in loop_corners_nms]  
            #loop_pred = torch.sigmoid(loop_pred).view(-1)
            loop_pred = loop_pred.detach().cpu().numpy()
            #loop_accs = loop_accs.detach().cpu().numpy()

            # Get min baseline
            for k, loop in enumerate(edges_loops):
                inds = np.where(loop==1)
                confs = edges_baseline[inds]
                loop_accs[k] = np.min(confs) 

            # Apply NMS
            edges_nms_baseline, loop_confs_baseline = loop_nms(loop_accs, edges_loops, loop_corners_nms)
            edges_nms_cls, loop_confs_cls = loop_nms(loop_pred, edges_loops, loop_corners_nms)

            # edges_nms_baseline = np.sum(loop_edges_baseline, axis=0)
            # edges_nms_cls = np.sum(loop_edges_cls, axis=0)

            inds = np.where(loop_accs > .5)
            edges_no_nms_baseline = edges_loops[inds]
            inds = np.where(loop_pred > .5)
            edges_no_nms_cls = edges_loops[inds]

            # for k, lp in enumerate(loop_pred):
            #     edge_in_loop = edges_loops[k]
            #     im = draw_edges(edge_in_loop, building.edges_det)

            #     print(edge_in_loop)
            #     print("baseline: ", loop_accs[k])
            #     print("loop cls: ", lp)
            #     cv2.imshow('loop', im)
            #     cv2.imshow('rgb', rgb)
            #     cv2.waitKey(0)

        # im = draw_edges(edges_out, building.edges_det)
        # cv2.imshow('image', im)
        # cv2.imshow('rgb', rgb)
        # cv2.waitKey(0)

        # Per Edge Classification
        per_edge_imgs, _ = building.visualize(mode='draw_annot', edge_state=edges_baseline, building_idx=building_index, post_processing=False)
        row_images.append(per_edge_imgs[0])

        # Baseline no NMS
        image = building.visualize_multiple_loops(edges_no_nms_baseline)
        row_images.append(image)

        # Baseline NMS
        image = building.visualize_multiple_loops(edges_nms_baseline)
        row_images.append(image)

        # Loop Classifier no NMS
        image = building.visualize_multiple_loops(edges_no_nms_cls)
        row_images.append(image)
        
        # Loop Classifier NMS
        image = building.visualize_multiple_loops(edges_nms_cls)                            
        row_images.append(image)
        row_images.append(per_edge_imgs[2])

        if len(row_images) >= 10:
            all_images.append(row_images)
            row_images = []

    image = tileImages(all_images, background_color=0)
    cv2.imwrite(options.test_dir + '/results_on_detections_at_{}.png'.format(epoch), image)      

        # ## DEBUG ##
        # for k, loop in enumerate(edges_loops):
            # import matplotlib.pyplot as plt

            # im = Image.fromarray(im_arr[k].squeeze(0).detach().cpu().numpy())

            # print(edges_loops[k])

            # plt.imshow(im)
            # plt.show()

            
            # plt.imshow(im)
            # print(edges_confidence.shape)
            # print(loop_labels[k])
            # print(edges_loops[k])
            # plt.show() 

            # print(loop_pred)
   
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
