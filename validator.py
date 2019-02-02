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

class Validator():
    def __init__(self, options):
        self.options = options
        edge_classifier = CustomResNet(pretrained=False, num_classes=2)
        edge_classifier = edge_classifier.cuda()
        edge_classifier = edge_classifier.eval()

        edge_classifier.load_state_dict(torch.load('checkpoint/incremental_dets_only_mixed/0_checkpoint.pth'))
        self.edge_classifier = edge_classifier
        return

    def validate(self, building):
        filename = 'cache/edge_confidence_' + self.options.corner_type + '/' + building._id + '.pth'
        if os.path.exists(filename):
            return torch.load(filename)
        _input, _labels = building.create_samples(mode='inference')
        _input = _input.cuda()
        logits = []
        batch_size = 16
        for batch_index in range((len(_input) - 1) // batch_size + 1):
            logits.append(self.edge_classifier(_input[batch_index * batch_size:min((batch_index + 1) * batch_size, len(_input))]).detach())
            continue
        logits = torch.cat(logits, dim=0)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs[:, 1]
        probs = probs
        torch.save(probs, filename)
        return probs
