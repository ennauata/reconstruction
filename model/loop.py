import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import cv2
import numpy as np
from model.modules import findLoopsModule
from model.resnet import BasicBlock, conv1x1

class LoopEncoderSparse(nn.Module):
    def __init__(self, num_input_channels):
        super(LoopEncoderSparse, self).__init__()
        kernel_size = 3
        self.padding = nn.ReflectionPad1d((kernel_size - 1) // 2)
        self.conv_0 = nn.Sequential(self.padding, nn.Conv1d(num_input_channels, 64, kernel_size=kernel_size), nn.ReLU(inplace=True))
        self.conv_1 = nn.Sequential(self.padding, nn.Conv1d(64, 64, kernel_size=kernel_size), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(self.padding, nn.Conv1d(64, 64, kernel_size=kernel_size), nn.ReLU(inplace=True))
        self.loop_pred = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        return
    
    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.max(-1)[0]
        pred = torch.sigmoid(self.loop_pred(x))
        return x, pred

class LoopEncoderDense(nn.Module):
    def __init__(self, num_input_channels):
        super(LoopEncoderDense, self).__init__()
        kernel_size = 5
        self.padding = nn.ReflectionPad1d((kernel_size - 1) // 2)
        self.conv_0 = nn.Sequential(self.padding, nn.Conv1d(num_input_channels, 64, kernel_size=kernel_size), nn.ReLU(inplace=True),
                                    self.padding, nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2), nn.ReLU(inplace=True))
        self.conv_1 = nn.Sequential(self.padding, nn.Conv1d(64, 64, kernel_size=kernel_size), nn.ReLU(inplace=True),
                                    self.padding, nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2), nn.ReLU(inplace=True))

        kernel_size = 3
        self.padding_small = nn.ReflectionPad1d((kernel_size - 1) // 2)                
        self.conv_2 = nn.Sequential(self.padding_small, nn.Conv1d(64, 64, kernel_size=kernel_size), nn.ReLU(inplace=True),
                                    self.padding_small, nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2), nn.ReLU(inplace=True))
        self.loop_feature = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.loop_pred = nn.Linear(64, 1)
        self.density = 20.0
        return
    
    def forward(self, image_x, corners):
        edges = torch.cat([corners, torch.cat([corners[1:], corners[:1]], dim=0)], dim=-1)        
        lengths = torch.norm(edges[:, :2] - edges[:, 2:4], dim=-1)
        length_sum = lengths.sum()
        density = max(self.density, 1.0 / lengths[lengths > 1e-4].min())
        num_points = ((length_sum * density).long() // 8 + 1) * 8
        offsets = torch.arange(num_points).float().cuda() * (length_sum / num_points)
        edge_lengths_2 = torch.cumsum(lengths, dim=0)
        edge_lengths_1 = torch.cat([torch.zeros(1).cuda(), edge_lengths_2[:-1]], dim=0)
        edge_offsets = (offsets.unsqueeze(-1) - edge_lengths_1) / torch.clamp(lengths, min=1e-4)
        points = edges[:, :2] + edge_offsets.unsqueeze(-1) * (edges[:, 2:4] - edges[:, :2])
        mask = ((offsets.unsqueeze(-1) >= edge_lengths_1) & (offsets.unsqueeze(-1) < edge_lengths_2)).float()
        points = (points * mask.unsqueeze(-1)).sum(1)
        points_round = torch.clamp(torch.round(points * image_x.shape[1]).long(), 0, image_x.shape[1] - 1)
        x = image_x[:, points_round[:, 0], points_round[:, 1]].clone()

        x = torch.cat([x, points.transpose(0, 1)], dim=0).unsqueeze(0)
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
              
        if False:
            points = points_round.detach().cpu().numpy()
            #image = np.zeros((64, 64, 3), dtype=np.uint8)
            #image = cv2.imread('test/image.png')
            image = image_x.detach().cpu().numpy().mean(0)
            image = (image / image.max() * 255).astype(np.uint8)
            image = np.stack([image, image, image], axis=-1)
            #image[points[:, 0], points[:, 1]] = np.array([255, 0, 0])
            corners = np.clip((corners.detach().cpu().numpy() * 64).round().astype(np.int32), 0, 255)
            for corner in corners:
                cv2.circle(image, (corner[1], corner[0]), radius=2, thickness=-1, color=(0, 0, 255))
                continue
            cv2.imwrite('test/mask.png', cv2.resize(image, (256, 256)))
            exit(1)
            pass
        x = x.max(-1)[0]
        x = self.loop_feature(x)
        pred = torch.sigmoid(self.loop_pred(x))
        return x, pred    
        
class LoopModel(nn.Module):
    def __init__(self, options, num_classes=1):
        super(LoopModel, self).__init__()
        self.options = options
        self.inplanes = 64

        block = BasicBlock
        layers = [3, 4, 6, 3]
        
        self.relu = nn.ReLU(inplace=True)            
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.image_layer_0 = nn.Sequential(nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,
                                                     bias=False),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))                
        self.image_layer_1 = self._make_layer(block, 64, layers[0])
        self.image_layer_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.image_layer_3 = self._make_layer(block, 256, layers[2], stride=2)
        self.image_layer_4 = self._make_layer(block, 512, layers[3], stride=2)

        self.coord_layer_1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU())
        self.coord_layer_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU())
        self.coord_layer_3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU())
        self.coord_layer_4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU())

        self.edge_pred_1 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.edge_pred_2 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.edge_pred_3 = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 1))
        self.edge_pred_4 = nn.Sequential(nn.Linear(1024, 64), nn.ReLU(), nn.Linear(64, 1))

        self.loop_pred_1 = LoopEncoderDense(64 + 2)
        self.loop_pred_2 = LoopEncoderDense(128 + 2)
        self.loop_pred_3 = LoopEncoderDense(256 + 2)
        self.loop_pred_4 = LoopEncoderDense(512 + 2)

        
        self.image_aggregate_1 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.image_aggregate_2 = nn.Sequential(nn.Conv2d(128 + 64, 128, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.image_aggregate_3 = nn.Sequential(nn.Conv2d(256 + 64, 256, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.image_aggregate_4 = nn.Sequential(nn.Conv2d(512 + 64, 512, kernel_size=1, bias=False), nn.ReLU(inplace=True))

        self.coord_aggregate_1 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.coord_aggregate_2 = nn.Sequential(nn.Conv2d(128 + 64, 128, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.coord_aggregate_3 = nn.Sequential(nn.Conv2d(256 + 64, 256, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.coord_aggregate_4 = nn.Sequential(nn.Conv2d(512 + 64, 512, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        
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

    def aggregate_loop(self, image_x, coord_x, edge_pred, loop_pred, image_aggregate, coord_aggregate, corners, corner_edge_pairs, edge_corner, num_corners):
        edge_x = torch.cat([image_x.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0], coord_x], dim=1).squeeze(-1).squeeze(-1)
        edge_confidence = torch.sigmoid(edge_pred(edge_x)).view(-1)
        
        all_loops = findLoopsModule(edge_confidence, edge_corner, num_corners, self.options.max_num_loop_corners, corners=corners, disable_colinear=True)

        corner_features = torch.zeros((num_corners, edge_x.shape[1])).cuda()
        corner_features.index_add_(0, corner_edge_pairs[:, 0], edge_x[corner_edge_pairs[:, 1]])
        count = torch.zeros(num_corners).cuda()
        count.index_add_(0, corner_edge_pairs[:, 0], torch.ones(len(corner_edge_pairs)).cuda())
        corner_features = corner_features / torch.clamp(count.view((-1, 1)), min=1)

        loop_features = []
        loop_confidences = []        
        loop_edges = []
        max_confidence = max([confidence.max() for confidence, loops in all_loops])
        for confidence, loops in all_loops:
            for loop_index in range(len(loops)):
                if confidence[loop_index] < min(0.5, max_confidence):
                    continue
                loop = loops[loop_index]
                #loop_corner_pairs = torch.cat([torch.stack([loop, torch.cat([loop[-1:], loop[:-1]], dim=0)], dim=-1), torch.stack([loop, torch.cat([loop[1:], loop[:1]], dim=0)], dim=-1)], dim=0)
                loop_corner_pairs = torch.cat([torch.stack([loop, torch.cat([loop[-1:], loop[:-1]], dim=0)], dim=-1), torch.stack([torch.cat([loop[-1:], loop[:-1]], dim=0), loop], dim=-1)], dim=0)
                loop_edge_mask = (edge_corner.unsqueeze(1) == loop_corner_pairs).min(-1)[0]
                loop_edges.append(loop_edge_mask.max(-1)[0].float())

                # print(loop)
                # print(edge_corner)
                # print(loop_corner_pairs)
                # print(loop_edge_mask)
                loop_edge_mask = torch.max(loop_edge_mask[:, :loop_edge_mask.shape[-1] // 2], loop_edge_mask[:, loop_edge_mask.shape[-1] // 2:])
                loop_edge = loop_edge_mask.max(0)[1]                
                #loop_edge = loop_edge.nonzero().view(-1)
                #loop_inp = torch.cat([corner_features[loop], corners[loop], edge_x[loop_edge], edge_confidence[loop_edge].view((-1, 1))], dim=-1).transpose(0, 1).unsqueeze(0)
                #edge_x = torch.cat([image_x.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0], coord_x], dim=1).squeeze(-1).squeeze(-1)

                ## Find loop corners along the loop
                # loop_corners = corners[edge_corner[loop_edge]]
                # if torch.norm(loop_corners[0, 0] - loop_corners[-1], dim=-1).min() > torch.norm(loop_corners[0, 1] - loop_corners[-1], dim=-1).min():
                #     loop_corners = torch.cat([torch.cat([loop_corners[:1, 1:2], loop_corners[:1, 0:1]], dim=1), loop_corners[1:]], dim=0)
                #     pass
                # loop_corners_prev = torch.cat([loop_corners[-1:], loop_corners[:-1]], dim=0)
                # distances = torch.stack([torch.norm(loop_corners[:, 0] - loop_corners_prev[:, 1], dim=-1), torch.norm(loop_corners[:, 1] - loop_corners_prev[:, 1], dim=-1)], dim=-1)
                # corner_indices = torch.cat([torch.zeros(1).long().cuda(), distances.min(-1)[1][1:]], dim=0)
                # corner_indices = torch.cumsum(corner_indices, dim=0) % 2
                # loop_corners = loop_corners[torch.arange(len(distances)).cuda().long(), corner_indices]
                
                #loop_inp = torch.cat([edge_x[loop_edge], corners[loop], edge_confidence[loop_edge].view((-1, 1))], dim=-1).transpose(0, 1).unsqueeze(0)
                #print(corners[loop])
                #print(corners[edge_corner[loop_edge]])
                
                loop_feature, loop_confidence = loop_pred(image_x[loop_edge].max(0)[0], corners[loop])
                loop_features.append(loop_feature.squeeze(0))
                loop_confidences.append(loop_confidence)
                continue
            continue
        loop_features = torch.stack(loop_features, dim=0)
        loop_confidences = torch.stack(loop_confidences, dim=0).view(-1)
        loop_edges = torch.stack(loop_edges, dim=0)

        loop_edge_weights = loop_confidences.view((-1, 1, 1)) * loop_edges.unsqueeze(-1)
        coord_features = (loop_features.unsqueeze(1) * loop_edge_weights).sum(0) / torch.clamp(loop_edge_weights.sum(0), min=1e-4)
        coord_features = coord_features.unsqueeze(-1).unsqueeze(-1)
        coord_x = coord_aggregate(torch.cat([coord_x, coord_features], dim=1))

        return image_x, coord_x, [edge_confidence, loop_confidences, loop_edges]
        
        #loop_edge_weights = loop_edge_weights.unsqueeze(-1).unsqueeze(-1)
        #print(image_x.shape, loop_edge_weights.shape, 
        #image_features = (image_x.unsqueeze(1) * loop_edge_weights).sum(0) / torch.clamp(loop_edge_weights.sum(0), min=1e-4)
        #print(image_features.shape)
        image_x = image_aggregate(torch.cat([image_x, coord_features.repeat((1, 1, image_x.shape[2], image_x.shape[3]))], dim=1))
        return image_x, coord_x, [edge_confidence, loop_confidences, loop_edges]
    
    def forward(self, image, corners, edges, corner_edge_pairs, edge_corner):
        #cv2.imwrite('test/image.png', (image[0, :3].detach().cpu().numpy() * 255).astype(np.uint8).transpose((1, 2, 0)))
        num_corners = len(corners)
        intermediate_results = []

        image_x = self.image_layer_0(image)
        coord_x = edges.unsqueeze(-1).unsqueeze(-1)
        
        image_x = self.image_layer_1(image_x)
        coord_x = self.coord_layer_1(coord_x)
        image_x, coord_x, intermediate_result = self.aggregate_loop(image_x, coord_x, self.edge_pred_1, self.loop_pred_1, self.image_aggregate_1, self.coord_aggregate_1, corners, corner_edge_pairs, edge_corner, num_corners)
        intermediate_results.append(intermediate_result)

        image_x = self.image_layer_2(image_x)
        coord_x = self.coord_layer_2(coord_x)
        image_x, coord_x, intermediate_result = self.aggregate_loop(image_x, coord_x, self.edge_pred_2, self.loop_pred_2, self.image_aggregate_2, self.coord_aggregate_2, corners, corner_edge_pairs, edge_corner, num_corners)
        intermediate_results.append(intermediate_result)

        image_x = self.image_layer_3(image_x)
        coord_x = self.coord_layer_3(coord_x)
        image_x, coord_x, intermediate_result = self.aggregate_loop(image_x, coord_x, self.edge_pred_3, self.loop_pred_3, self.image_aggregate_3, self.coord_aggregate_3, corners, corner_edge_pairs, edge_corner, num_corners)
        intermediate_results.append(intermediate_result)        
        
        image_x = self.image_layer_4(image_x)
        coord_x = self.coord_layer_4(coord_x)
        image_x, coord_x, intermediate_result = self.aggregate_loop(image_x, coord_x, self.edge_pred_4, self.loop_pred_4, self.image_aggregate_4, self.coord_aggregate_4, corners, corner_edge_pairs, edge_corner, num_corners)
        intermediate_results.append(intermediate_result)
        return intermediate_results
