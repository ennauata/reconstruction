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

    def __init__(self, options, id_list, num_edges, split='train', load_heatmaps=False):
        self.options = options
        self.num_images = len(id_list)
        self.load_heatmaps = load_heatmaps
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
            #_id = '1549084252.4'
            corner_types = []
            corner_type = options.corner_type if split == 'train' else options.testing_corner_type
            if 'annots_dets' in corner_type:
                corner_types.append('annots_dets')
            else:
                if 'dets' in corner_type:
                    corner_types.append('dets_only')
                    pass
                if 'annots' in corner_type:
                    corner_types.append('annots_only')
                    pass
                pass
            for corner_type in corner_types:
                building = Building(options, _id, with_augmentation=split == 'train', corner_type=corner_type)
                if 'uniform' not in options.suffix and 'single' not in options.suffix:            
                    if num_edges > building.current_num_edges():
                        continue
                    elif num_edges < building.current_num_edges() and num_edges >= 0:
                        #building.reset(num_edges)
                        pass
                    pass
                self.buildings.append(building)
                pass
            if len(self.buildings) >= self.num_images:
                break
            continue

        # if split != 'train':
        #     self.buildings = self.buildings[-50:]
        #     self.num_images = 50
        #     pass
        
        #self.buildings = self.buildings[-6:-5]
        #self.num_images = len(self.buildings)
        
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
        sample = building.create_sample_graph(load_heatmaps=self.load_heatmaps)
        return sample + [index, ]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.num_images
