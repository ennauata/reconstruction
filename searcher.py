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
        
        # if options.num_edges >= 0:
        #     all_num_edges = [options.num_edges]
        # else:
        #     all_num_edges = range(options.max_num_edges)
        #     pass
        all_num_edges = range(options.max_num_edges)
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
        visited_edges = {}
        while True:
            _input, _labels = building.create_samples(mode='inference')
            current_num_edges = building.current_num_edges()
            with torch.no_grad():
                _input = _input.cuda()
                if 'uniform' in self.options.suffix or 'single' in self.options.suffix:
                    logits = self.edge_classifiers[0](_input)
                else:
                    if current_num_edges in self.edge_classifiers:
                        logits = self.edge_classifiers[current_num_edges](_input)
                    else:
                        break
                        logits = self.edge_classifiers[max(list(self.edge_classifiers.keys()))](_input)
                        pass
                    pass
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs = probs[:, 1]
                probs = probs.detach().cpu().numpy()
                pass
            _labels = _labels.detach().cpu().numpy()

            if 'single' in self.options.suffix:
                probs[np.nonzero(building.current_edges())[0]] = 0
                pass
            for edge_index, count in visited_edges.items():
                if count >= 2:
                    probs[edge_index] = 0
                    pass
                continue
            if probs.max() < 0.5:
                if _labels.max() < 0.5:
                    if current_num_edges < self.options.max_num_edges:
                        statistics[current_num_edges] += 1
                        pass
                    statistics[-1] += 1                    
                    pass
                break
            if 'batch' in self.options.suffix:
                indices = probs.argsort()
                for edge_index in indices:
                    if probs[edge_index] > 0.5:
                        building.update_edge(edge_index)
                        pass
                    continue
                break
            best_edge = probs.argmax()
            building.update_edge(best_edge)
            #print(best_edge, _labels[best_edge], building.current_num_edges(), probs.max())
            if current_num_edges < self.options.max_num_edges:
                statistics[current_num_edges] += _labels[best_edge]
                pass
            if num_edges_target > 0 and building.current_num_edges() == num_edges_target:
                break
            if best_edge not in visited_edges:
                visited_edges[best_edge] = 0
                pass
            visited_edges[best_edge] += 1
            continue
        return statistics
