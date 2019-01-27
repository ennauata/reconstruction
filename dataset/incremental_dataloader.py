import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as p
import glob
from collections import OrderedDict
from utils.intersections import doIntersect
from math import atan2,degrees
import cv2
from dataset.building import Building
import time

class GraphData(Dataset):

    def __init__(self, options, id_list, num_edges, split='train'):
        self.options = options
        self.num_images = len(id_list)        
        if split == 'train':
            if options.num_training_images > 0:
                self.num_images = options.num_training_images
                pass
        else:
            if options.num_testing_images > 0:
                self.num_images = options.num_testing_images
                pass
            pass                
        self.buildings = []
        for _id in id_list:
            building = Building(options, _id, with_augmentation=split == 'train')
            if 'uniform' not in options.suffix and 'single' not in options.suffix:            
                if num_edges > building.current_num_edges():
                    continue
                elif num_edges < building.current_num_edges() and num_edges >= 0:
                    building.reset(num_edges)
                    pass
                pass
            self.buildings.append(building)
            if len(self.buildings) >= self.num_images:
                break
            continue
        print('num images', split, len(self.buildings))
        self.split = split
        self.num_edges = num_edges
        return

    def reset(self):
        self.id_graph_map = {}
        return
    
    def __getitem__(self, index):
        # retrieve id
        if self.split == 'train':
            t = int(time.time() * 1000000)
            np.random.seed(((t & 0xff000000) >> 24) +
                           ((t & 0x00ff0000) >> 8) +
                           ((t & 0x0000ff00) << 8) +
                           ((t & 0x000000ff) << 24))            
            index = np.random.randint(len(self.buildings))
        else:
            index = index % len(self.buildings)
            pass
        building = self.buildings[index]

        if 'mixed' in self.options.suffix:
            num_edges = np.random.randint(building.current_num_edges() + 1)
            if self.num_edges >= 0:
                num_edges = min(num_edges, self.num_edges)
                pass
        else:
            num_edges = self.num_edges
            pass
        sample, label = building.create_sample(num_edges_source=num_edges)

        return sample, label, building._id

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.num_images


    # def generate_bins(self, edges_state, ce_angles_bins, n_bins=72):
    #     ce_bins = np.zeros((ce_angles_bins.shape[0], n_bins))
    #     for c_ind in range(ce_angles_bins.shape[0]):
    #         e_inds = np.where((ce_angles_bins[c_ind] >= 0) & ((edges_state == 1))) 
    #         bins = ce_angles_bins[c_ind, e_inds].ravel().astype(np.int32)
    #         ce_bins[c_ind, bins] += 1.0
    #     return ce_bins

    
    # def compose(self, im, mask):
    #     for i in range(im.shape[1]):
    #         for j in range(im.shape[2]):
    #             im[:, i, j] = (1.0-mask[i, j])*im[:, i, j] + mask[i, j]*np.array([255.0, 0, 0])
    #     return im
    

    

    # def compute_overlaps_masks(self, masks1, masks2):
    #     '''Computes IoU overlaps between two sets of masks.
    #     masks1, masks2: [Height, Width, instances]
    #     '''
    #     # flatten masks
    #     masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    #     masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    #     area1 = np.sum(masks1, axis=0)
    #     area2 = np.sum(masks2, axis=0)

    #     # intersections and union
    #     intersections = np.dot(masks1.T, masks2)
    #     union = area1[:, None] + area2[None, :] - intersections
    #     overlaps = intersections / union

    #     return overlaps
