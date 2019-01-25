import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import glob
import numpy as np
import os
from dataset.metrics import Metrics
from model.resnet import CustomResNet
#from model.regressor import DistanceRegressor
import time

class Searcher():
    def __init__(self, options):
        self.options = options
        edge_classifier = CustomResNet(pretrained=False, num_classes=2)
        edge_classifier = edge_classifier.cuda()
        edge_classifier = edge_classifier.eval()

        self.edge_classifiers = {}
        # if options.num_edges == 0:
        #     all_num_edges = []
        if options.num_edges >= 0:
            all_num_edges = [options.num_edges]
        else:
            all_num_edges = range(options.max_num_edges)
            pass
        for num_edges in all_num_edges:
            filename = options.checkpoint_dir + '/' + str(num_edges) + '_checkpoint.pth'
            if os.path.exists((filename)):
                edge_classifier.load_state_dict(torch.load(filename))
                self.edge_classifiers[num_edges] = edge_classifier
            else:
                break
            continue            

        # distance_regressor = DistanceRegressor()
        # distance_regressor = distance_regressor.cuda()
        # distance_regressor = distance_regressor.eval()        
        # epoch = 240
        # src = 'saved_models'
        # distance_regressor.load_state_dict(torch.load('../distance_regression/{}/edge_classifier_iter_{}.pth'.format(src, epoch)))
        return

    def search(self, building, num_edges_target=-1):
        statistics = np.zeros(self.options.max_num_edges + 1, dtype=np.int32)
        while True:
            _input, _labels = building.create_samples()
            current_num_edges = building.current_num_edges()
            with torch.no_grad():
                _input = _input.cuda()
                logits = self.edge_classifiers[current_num_edges](_input)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs = probs[:, 1]
                probs = probs.detach().cpu().numpy()
                pass
            
            if probs.max() < 0.5:
                if _labels.max() < 0.5:
                    statistics[current_num_edges] += 1
                    pass
                break
            best_edge = probs.argmax()
            building.update_edge(best_edge)
            statistics[current_num_edges] += _labels[best_edge]
            if num_edges_target > 0 and building.current_num_edges() == num_edges_target:
                break
            continue
        return statistics
