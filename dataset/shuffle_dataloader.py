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

class ShuffleData(Dataset):

    def __init__(self, options, id_list, num_edges, split='train', load_heatmaps=False):
        self.options = options
        self.load_heatmaps = load_heatmaps
        self.buildings = []
        self.building_edge_indices = []
        for _id in id_list:
            building = Building(options, _id, with_augmentation=split == 'train')
            self.building_edge_indices += [(len(self.buildings), edge_index) for edge_index in range(building.num_edges)]
            self.buildings.append(building)
            continue
        self.num_images = len(self.building_edge_indices)
        print('num images', split, self.num_images)
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
            index = np.random.randint(len(self.building_edge_indices))
        else:
            index = index % len(self.building_edge_indices)
            pass
        building_index, edge_index = self.building_edge_indices[index]
        building = self.buildings[building_index]
        sample = building.create_sample_edge(edge_index, load_heatmaps=self.load_heatmaps)
        return sample + [building_index, edge_index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.num_images
