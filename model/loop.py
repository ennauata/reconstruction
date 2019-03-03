import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import cv2
import numpy as np
from model.modules import findLoopsModule

class LoopEncoder(nn.Module):
    def __init__(self, num_input_channels):
        super(LoopEncoder, self).__init__()
        self.padding = nn.ReflectionPad1d((loop_kernel_size - 1) // 2)
        self.conv_0 = nn.Sequential(self.padding, nn.Conv1d(num_input_channels, 64, kernel_size=3), nn.ReLU(inplace=True))
        self.conv_1 = nn.Sequential(self.padding, nn.Conv1d(64, 64, kernel_size=3), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(self.padding, nn.Conv1d(64, 64, kernel_size=3), nn.ReLU(inplace=True))
        return
    
    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.max(-1)[0]
        return x
        
class LoopModel(nn.Module):
    def __init__(self, options, num_classes=1):
        super(LoopModel, self).__init__()
        self.options = options
        self.inplanes = 64

        block = BasicBlock
        layers = [3, 4, 6, 3]
        
        self.relu = nn.ReLU(inplace=True)            
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.image_layer_0 = nn.Sequential(nn.Conv2d(4, self.num_channels, kernel_size=7, stride=2, padding=3,
                                                     bias=False),
                                           nn.BatchNorm2d(self.num_channels),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))                
        self.image_layer_1 = self._make_layer(block, 64, layers[0])
        self.image_layer_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.image_layer_3 = self._make_layer(block, 256, layers[2], stride=2)
        self.image_layer_4 = self._make_layer(block, 512, layers[3], stride=2)

        self.coord_layer_1 = nn.Sequential(nn.Conv2d(4, 16, kernel_size=1), nn.BatchNorm2d(16), nn.ReLU())
        self.coord_layer_2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1), nn.BatchNorm2d(32), nn.ReLU())
        self.coord_layer_3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU())
        self.coord_layer_4 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU())

        self.edge_pred_1 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.edge_pred_2 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.edge_pred_3 = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 1))
        self.edge_pred_4 = nn.Sequential(nn.Linear(1024, 64), nn.ReLU(), nn.Linear(64, 1))

        self.coord_encoder_1 = LoopEncoder(128)
        self.coord_encoder_2 = LoopEncoder(256)
        self.coord_encoder_3 = LoopEncoder(512)
        self.coord_encoder_4 = LoopEncoder(1024)

        
        self.image_aggregate_1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.image_aggregate_2 = nn.Sequential(nn.Conv2d(128 * 4, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.image_aggregate_3 = nn.Sequential(nn.Conv2d(256 * 4, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.image_aggregate_4 = nn.Sequential(nn.Conv2d(512 * 4, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True))

        self.coord_aggregate_1 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.coord_aggregate_2 = nn.Sequential(nn.Conv2d(128 * 4, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.coord_aggregate_3 = nn.Sequential(nn.Conv2d(256 * 4, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.coord_aggregate_4 = nn.Sequential(nn.Conv2d(512 * 4, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True))                                

        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        return
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            if _ == blocks - 1 and 'sharing' in self.options.suffix:
                layers.append(block(self.inplanes, planes))
            else:
                layers.append(block(self.inplanes, planes))
                pass
            continue

        return nn.Sequential(*layers)

    def aggregate(self, x, corner_edge_pairs, edge_corner, num_corners, aggregator=None):
        corner_features = torch.zeros((num_corners, x.shape[1], x.shape[2], x.shape[3])).cuda()
        corner_features.index_add_(0, corner_edge_pairs[:, 0], x[corner_edge_pairs[:, 1]])
        count = torch.zeros(num_corners).cuda()
        count.index_add_(0, corner_edge_pairs[:, 0], torch.ones(len(corner_edge_pairs)).cuda())
        corner_features = corner_features / torch.clamp(count.view((-1, 1, 1, 1)), min=1)
        left_x = corner_features[edge_corner[:, 0]]
        right_x = corner_features[edge_corner[:, 1]]            
        #global_x = x.mean(0, keepdim=True).repeat(len(x), 1)
        global_x = x.max(0, keepdim=True)[0]
        if aggregator == None:
            x = x + left_x + right_x + global_x
        else:
            x = aggregator(torch.cat([x, left_x, right_x, global_x], dim=-1))
            pass
        return x

    def aggregate_loop(image_x, coord_x, edge_pred, loop_pred, loop_encoder, corners, corner_edge_pairs, edge_corner, num_corners):
        edge_x = torch.cat([image_x.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0], coord_x], dim=1)
        edge_confidence = torch.sigmoid(edge_pred(edge_x))
        
        all_loops = findLoopsModule(edge_confidence, edge_corner, num_corners, self.options.max_num_loop_corners, corners=corners, disable_colinear=True)

        corner_features = torch.zeros((num_corners, x.shape[1], x.shape[2], x.shape[3])).cuda()
        corner_features.index_add_(0, corner_edge_pairs[:, 0], x[corner_edge_pairs[:, 1]])
        count = torch.zeros(num_corners).cuda()
        count.index_add_(0, corner_edge_pairs[:, 0], torch.ones(len(corner_edge_pairs)).cuda())
        corner_features = corner_features / torch.clamp(count.view((-1, 1, 1, 1)), min=1)

        loop_features = []
        loop_edges = []
        for confidence, loops in all_loops:
            for loop_index in range(len(loops)):
                loop = loops[loop_index]
                loop_inp = torch.cat([corner_features[loop], corners[loop]], dim=-1)
                loop_feature = loop_encoder(loop_inp)
                loop_features.append(loop_feature)
                loop_corner_pairs = torch.cat([torch.stack([loop, torch.cat([loop[-1:], loop[:-1]], dim=0)], dim=-1), torch.stack([loop, torch.cat([loop[1:], loop[:1]], dim=0)], dim=-1)], dim=0)
                loop_edge = (edge_corner.unsqueeze(1) == loop_corner_pairs).min(-1)[0].max(-1)[0]
                loop_edges.append(loop_edge)
                continue
            loop_features = torch.stack(loop_features, dim=0).transpose(1, 2)
            loop_x = self.loop_encoder(loop_features)
            loop_x = loop_x.max(2)[0]
            loop_pred = torch.sigmoid(self.loop_pred(loop_x).view(-1))
        
        
    def forward(self, image, corners, edges, corner_edge_pairs, edge_corner):
        num_corners = len(corners)
        
        image_x = self.image_layer_1(self.image_layer_0(image))
        coord_x = self.coord_layer_1(edges.unsqueeze(-1).unsqueeze(-1))

        self.aggregate_loop(image_x, coord_x, self.edge_pred_1)
        coord_x = self.aggregate(edge_x, corner_edge_pairs, edge_corner, num_corners, self.coord_aggregate_1)

        image_x = self.image_layer_2(image_x)        
        coord_x = self.layer_2(edge_x)
        
        image_x_3 = self.image_layer_3(image_x_2)
        image_x_4 = self.image_layer_4(image_x_3)

        num_corners = len(corners)

        x = self.aggregate(x, corner_edge_pairs, edge_corner, image_x_2, edges, num_corners)
        x = self.layer_3(x)
        x = self.aggregate(x, corner_edge_pairs, edge_corner, image_x_3, edges, num_corners)
        x = self.layer_4(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if aggregate:
            x = self.layer1(x)
            x = self.aggregate(x, corner_edge_pairs, edge_corner, num_corners)
            x = self.layer2(x)
            x = self.aggregate(x, corner_edge_pairs, edge_corner, num_corners)            
            x = self.layer3(x)
            x = self.aggregate(x, corner_edge_pairs, edge_corner, num_corners)            
            x = self.layer4(x)
            x = self.aggregate(x, corner_edge_pairs, edge_corner, num_corners)            
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            pass

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        pred = torch.sigmoid(x).view(-1)
        return pred


class GraphModelCustom(nn.Module):
    def __init__(self, options):
        super(GraphModelCustom, self).__init__()
        
        self.options = options
        
        if 'image_only' in self.options.suffix:
            self.inplanes = 64
        else:
            self.inplanes = 16
            pass
        self.num_channels = self.inplanes
        block = BasicBlock
        layers = [3, 4, 6, 3]
        self.image_layer_0 = nn.Sequential(nn.Conv2d(4, self.num_channels, kernel_size=7, stride=2, padding=3,
                                                     bias=False),
                                           nn.BatchNorm2d(self.num_channels),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        if 'image' in self.options.suffix:
            self.image_layer_1 = self._make_layer(block, self.num_channels, layers[0])
            self.image_layer_2 = self._make_layer(block, self.num_channels * 2, layers[1], stride=2)
            self.image_layer_3 = self._make_layer(block, self.num_channels * 4, layers[2], stride=2)
            self.image_layer_4 = self._make_layer(block, self.num_channels * 8, layers[3], stride=2)
            self.num_edge_points = 8            
            pass
        self.edge_pred = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 1))

        if 'loop' in self.options.suffix:
            loop_kernel_size = 7
            self.padding = nn.ReflectionPad1d((loop_kernel_size - 1) // 2)
            self.loop_encoder = nn.Sequential(self.padding, nn.Conv1d(128 + 3, 64, loop_kernel_size), nn.ReLU(),
                                           self.padding, nn.Conv1d(64, 64, loop_kernel_size), nn.ReLU(),
                                           self.padding, nn.Conv1d(64, 64, loop_kernel_size), nn.ReLU(),
            )
            self.loop_pred = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
            pass
        
        return

    def image_features(self, image_x, edges):
        width = image_x.shape[3]
        height = image_x.shape[2]
        x_1 = edges[:, 1:2] * width
        x_2 = edges[:, 3:4] * width
        y_1 = edges[:, 0:1] * height
        y_2 = edges[:, 2:3] * height
        alphas = (torch.arange(self.num_edge_points).float() / (self.num_edge_points - 1)).cuda()
        xs = torch.clamp(torch.round(x_1 + (x_2 - x_1) * alphas).long(), 0, width - 1)
        ys = torch.clamp(torch.round(y_1 + (y_2 - y_1) * alphas).long(), 0, height - 1)
        features = image_x[0, :, ys, xs]
        return features.mean(-1).transpose(1, 0)
    
    def aggregate(self, x, corner_edge_pairs, edge_corner, image_x, edges, num_corners, return_corner_features=False):
        if 'image_only' in self.options.suffix:
            x = self.image_features(image_x, edges)
        else:
            corner_features = torch.zeros((num_corners, x.shape[1])).cuda()
            corner_features.index_add_(0, corner_edge_pairs[:, 0], x[corner_edge_pairs[:, 1]])
            count = torch.zeros(num_corners).cuda()
            count.index_add_(0, corner_edge_pairs[:, 0], torch.ones(len(corner_edge_pairs)).cuda())
            corner_features = corner_features / torch.clamp(count.view((-1, 1)), min=1)
            left_x = corner_features[edge_corner[:, 0]]
            right_x = corner_features[edge_corner[:, 1]]            
            if 'image' in self.options.suffix:
                global_x = self.image_features(image_x, edges)
            else:
                global_x = x.mean(0, keepdim=True).repeat(len(x), 1)
                pass
            x = torch.cat([x, left_x, right_x, global_x], dim=1)
            pass
        if return_corner_features:
            return x, corner_features
        else:
            return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            continue

        return nn.Sequential(*layers)
    
    def forward(self, image, corners, edges, corner_edge_pairs, edge_corner):
        #cv2.imwrite('test/image.png', (image[0].detach().cpu().numpy().transpose((1, 2, 0))[:, :, :3] * 255).astype(np.uint8))
        if 'image' in self.options.suffix:
            image_x_1 = self.image_layer_1(self.image_layer_0(image))
            image_x_2 = self.image_layer_2(image_x_1)
            image_x_3 = self.image_layer_3(image_x_2)
            image_x_4 = self.image_layer_4(image_x_3)
        else:
            image_x_1 = None
            image_x_2 = None
            image_x_3 = None
            image_x_4 = None
            pass
        num_corners = len(corners)
        x = self.layer_1(edges)
        x = self.aggregate(x, corner_edge_pairs, edge_corner, image_x_1, edges, num_corners)
        x = self.layer_2(x)
        x = self.aggregate(x, corner_edge_pairs, edge_corner, image_x_2, edges, num_corners)
        x = self.layer_3(x)
        x = self.aggregate(x, corner_edge_pairs, edge_corner, image_x_3, edges, num_corners)
        x = self.layer_4(x)
        x, corner_features = self.aggregate(x, corner_edge_pairs, edge_corner, image_x_4, edges, num_corners, return_corner_features=True)            
        edge_pred = torch.sigmoid(self.edge_pred(x), ).view(-1)
        
        if 'loop' in self.options.suffix:
            all_loops = findLoopsModule(edge_pred, edge_corner, num_corners, self.options.max_num_loop_corners, corners=corners, disable_colinear=True)

            loop_features = []
            loop_corners = []
            for confidence, loops in all_loops:
                for loop_index in range(len(loops)):
                    loop = loops[loop_index]
                    feature = torch.cat([corner_features[loop], corners[loop], torch.ones((len(loop), 1)).cuda() * confidence[loop_index]], dim=-1)
                    #print(feature.shape)
                    if len(feature) < self.options.max_num_loop_corners:
                        feature = torch.cat([feature, torch.zeros((self.options.max_num_loop_corners - len(feature), feature.shape[1])).cuda()], dim=0)
                        pass
                    loop_features.append(feature)
                    loop_corners.append(loop)
                    continue
                continue
            loop_features = torch.stack(loop_features, dim=0).transpose(1, 2)
            loop_x = self.loop_encoder(loop_features)
            loop_x = loop_x.max(2)[0]
            loop_pred = torch.sigmoid(self.loop_pred(loop_x).view(-1))
        else:
            loop_pred = None
            loop_corners = []
            pass
        return edge_pred, loop_pred, loop_corners, loop_features
