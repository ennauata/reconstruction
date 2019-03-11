import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import cv2
import numpy as np
from model.modules import findLoopsModule, findMultiLoopsModule
from model.resnet import BasicBlock, conv1x1
from maskrcnn_benchmark.layers import roi_align
import sparseconvnet as scn

def sample_points(edges, density):
    num_points = torch.round(torch.norm(edges[:, :2] - edges[:, 2:4], dim=-1) * density).long()
    points = []
    for edge_index in range(len(num_points)):
        offsets = torch.rand(num_points[edge_index]).cuda().unsqueeze(-1)
        points.append(edges[edge_index, :2] + (edges[edge_index, 2:4] - edges[edge_index, :2]) * offsets)
        continue
    points = torch.cat(points, dim=0)
    return points

def edge_points(UVs, edges, size, distance_threshold=3):
    edges = edges * size
    directions = edges[:, 2:4] - edges[:, :2]
    lengths = torch.norm(directions, dim=-1)
    directions = directions / torch.clamp(lengths.unsqueeze(-1), min=1e-4)
    normals = torch.stack([directions[:, 1], -directions[:, 0]], dim=-1)
    normal_distances = torch.abs(((UVs - edges[:, :2]) * normals).sum(-1))
    direction_distances = ((UVs - edges[:, :2]) * directions).sum(-1)
    edge_mask = (normal_distances <= distance_threshold) & (direction_distances <= lengths) & (direction_distances >= 0) | (torch.norm(UVs - edges[:, :2], dim=-1) <= distance_threshold) | (torch.norm(UVs - edges[:, 2:4], dim=-1) <= distance_threshold)
    edge_mask = edge_mask.max(-1)[0]
    points = edge_mask.nonzero()
    return points

class LoopEncoder(nn.Module):
    def __init__(self, num_input_channels):
        super(LoopEncoder, self).__init__()
        kernel_size = 3
        self.padding = nn.ReflectionPad1d((kernel_size - 1) // 2)
        # self.conv_0 = nn.Sequential(self.padding, nn.Conv1d(num_input_channels, 64, kernel_size=kernel_size), nn.ReLU(inplace=True),
        #                             self.padding, nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2), nn.ReLU(inplace=True))
        # self.conv_1 = nn.Sequential(self.padding, nn.Conv1d(64, 64, kernel_size=kernel_size), nn.ReLU(inplace=True),
        #                             self.padding, nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2), nn.ReLU(inplace=True))
        self.conv_0 = nn.Sequential(self.padding, nn.Conv1d(num_input_channels, 64, kernel_size=kernel_size, stride=2), nn.ReLU(inplace=True))
        self.conv_1 = nn.Sequential(self.padding, nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2), nn.ReLU(inplace=True))

        kernel_size = 3
        self.padding_small = nn.ReflectionPad1d((kernel_size - 1) // 2)                
        # self.conv_2 = nn.Sequential(self.padding_small, nn.Conv1d(64, 64, kernel_size=kernel_size), nn.ReLU(inplace=True),
        #                             self.padding_small, nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(self.padding_small, nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2), nn.ReLU(inplace=True))        
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
        x = image_x[:, points_round[:, 0], points_round[:, 1]]

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
        pred = torch.sigmoid(self.loop_pred(x)).view(-1)
        return x, pred

class LoopEncoderMask(nn.Module):
    def __init__(self, num_input_channels):
        super(LoopEncoderMask, self).__init__()

        self.layer_0 = nn.Sequential(nn.Conv2d(65, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU())                
        self.loop_pred = nn.Sequential(nn.Linear(7 * 7 * 64, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1))

        self.mask_size = 28
        self.pool_size = 14
        self.max_pool = nn.MaxPool2d(kernel_size=self.mask_size // self.pool_size, stride=self.mask_size // self.pool_size)
        self.Us = torch.arange(self.mask_size).float().cuda().repeat((self.mask_size, 1)) + 0.5
        self.Vs = torch.arange(self.mask_size).float().cuda().unsqueeze(-1).repeat((1, self.mask_size)) + 0.5
        self.UVs = torch.stack([self.Vs, self.Us], dim=-1)        
        return
    
    def forward(self, image_x, loop_corners):
        loop_mins = []
        loop_maxs = []
        masks = []
        for corners in loop_corners:
            mins = corners.min(0)[0]
            maxs = corners.max(0)[0]
            loop_mins.append(mins)
            loop_maxs.append(maxs)

            edges = torch.stack([corners, torch.cat([corners[1:], corners[:1]], dim=0)], dim=-2)
            edges = (edges - mins) / (maxs - mins) * self.mask_size
            edge_normals = edges[:, 1] - edges[:, 0]
            edge_normals = torch.stack([edge_normals[:, 1], -edge_normals[:, 0]], dim=-1)
            edge_normals = edge_normals / torch.clamp(torch.norm(edge_normals, dim=-1, keepdim=True), min=1e-4)
            distances = ((self.UVs.unsqueeze(-2) - edges[:, 0]) * edge_normals).sum(-1)
            flags = ((distances > 0).long() * 2 - 1).sum(-1)
            if flags.max() < flags.min().abs():
                distances = -distances
                flags = -flags
                pass
            #print(distances)
            mask = (flags >= 2).float()
            masks.append(mask)
            continue
        loop_mins = torch.stack(loop_mins, dim=0)
        loop_maxs = torch.stack(loop_maxs, dim=0)
        masks = torch.stack(masks, dim=0)        
        
        # loop_mins = corners.min(0, keepdim=True)[0]
        # loop_maxs = corners.max(0, keepdim=True)[0]
        
        boxes = torch.stack([loop_mins[:, 1], loop_mins[:, 0], loop_maxs[:, 1], loop_maxs[:, 0]], dim=-1)
        pooled_x = roi_align(image_x.unsqueeze(0), torch.cat([torch.zeros(len(boxes), 1).cuda(), boxes], dim=-1), (self.pool_size, self.pool_size), float(image_x.shape[2]), 2)
        
        
        # print(edges)        
        # print(distances.shape)        
        # print(flags.min(), flags.max())        
        
        if False:
            #points = points_round.detach().cpu().numpy()
            #image = np.zeros((64, 64, 3), dtype=np.uint8)
            #image = cv2.imread('test/image.png')
            image = image_x.detach().cpu().numpy().mean(0)
            image = (image / image.max() * 255).astype(np.uint8)
            image = np.stack([image, image, image], axis=-1)
            #image[points[:, 0], points[:, 1]] = np.array([255, 0, 0])
            corners = np.clip((corners.detach().cpu().numpy() * image.shape[1]).round().astype(np.int32), 0, 255)
            mask = (mask.detach().cpu().numpy() * 255).astype(np.uint8)            
            for corner in corners:
                cv2.circle(image, (corner[1], corner[0]), radius=2, thickness=-1, color=(0, 0, 255))
                continue
            cv2.imwrite('test/image.png', cv2.resize(image, (256, 256)))
            cv2.imwrite('test/image_mask.png', cv2.resize(mask, (256, 256)))
            image = pooled_x.detach().cpu().numpy()[0].mean(0)
            image = (image / image.max() * 255).astype(np.uint8)
            cv2.imwrite('test/image_feature.png', cv2.resize(image, (256, 256)))            
            exit(1)
            pass
        masks = self.max_pool(masks.unsqueeze(1))        
        x = torch.cat([pooled_x, masks], dim=1)
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        pred = torch.sigmoid(self.loop_pred(x.view((len(x), -1)))).view(-1)
        
        return pred

class EdgeEncoder(nn.Module):
    def __init__(self, num_input_channels):
        super(EdgeEncoder, self).__init__()
        kernel_size = 5
        self.padding = nn.ReplicationPad1d((kernel_size - 1) // 2)
        self.conv_0 = nn.Sequential(self.padding, nn.Conv1d(num_input_channels, 64, kernel_size=kernel_size), nn.ReLU(inplace=True),
                                    self.padding, nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2), nn.ReLU(inplace=True))
        self.conv_1 = nn.Sequential(self.padding, nn.Conv1d(64, 64, kernel_size=kernel_size), nn.ReLU(inplace=True),
                                    self.padding, nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2), nn.ReLU(inplace=True))

        kernel_size = 3
        self.padding_small = nn.ReplicationPad1d((kernel_size - 1) // 2)                
        self.conv_2 = nn.Sequential(self.padding_small, nn.Conv1d(64, 64, kernel_size=kernel_size), nn.ReLU(inplace=True),
                                    self.padding_small, nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2), nn.ReLU(inplace=True))
        self.edge_feature = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.edge_pred = nn.Linear(64, 1)
        self.num_points = 32
        return
    
    def forward(self, image_x, edges):
        edge_offsets, _ = torch.sort(torch.rand(self.num_points).cuda())
        points = edges[:, :2].unsqueeze(-2) + edge_offsets.unsqueeze(-1) * (edges[:, 2:4] - edges[:, :2]).unsqueeze(-2)
        
        points_round = torch.clamp(torch.round(points * image_x.shape[1]).long(), 0, image_x.shape[1] - 1).view((-1, 2))
        x = image_x[:, points_round[:, 0], points_round[:, 1]]
        x = x.view((x.shape[0], -1, self.num_points))
        x = torch.cat([x.transpose(0, 1), points.transpose(1, 2)], dim=1)
        
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
              
        x = x.max(-1)[0]
        x = self.edge_feature(x)
        pred = torch.sigmoid(self.edge_pred(x)).view(-1)
        return pred

class SparseEncoder(nn.Module):
    def __init__(self, num_input_channels, full_scale=256):
        super(SparseEncoder, self).__init__()

        dimension = 2        
        m = 32
        residual_blocks = True
        block_reps = 2
        self.full_scale = full_scale
        self.distance_threshold = 3 if full_scale <= 64 else 5
        blocks = [['b', m * k, 2, 2] for k in [1, 2, 3, 4, 5]]
        num_final_channels = m * len(blocks)
        self.sparse_model = scn.Sequential().add(
            scn.InputLayer(dimension, full_scale, mode=4)).add(
            scn.SubmanifoldConvolution(dimension, num_input_channels, m, 3, False)).add(
            scn.MaxPooling(dimension, 3, 2)).add(
            scn.SparseResNet(dimension, m, blocks)).add(
            scn.BatchNormReLU(num_final_channels)).add(
            scn.SparseToDense(dimension, num_final_channels))

        self.pred = nn.Sequential(nn.Linear(num_final_channels, 64), nn.ReLU(), nn.Linear(64, 1))
        
        self.density = full_scale

        self.Us = torch.arange(full_scale).float().cuda().repeat((full_scale, 1)) + 0.5
        self.Vs = torch.arange(full_scale).float().cuda().unsqueeze(-1).repeat((1, full_scale)) + 0.5
        self.UVs = torch.stack([self.Vs, self.Us], dim=-1).unsqueeze(-2)
        return

    def forward(self, image_x, all_edges):
        all_points = []
        all_indices = []
        for edge_index, edges in enumerate(all_edges):
            points = edge_points(self.UVs, edges, self.full_scale, distance_threshold=self.distance_threshold)
            all_points.append(points)
            all_indices.append(torch.full((len(points), 1), edge_index).cuda().long())
            continue

        all_points = torch.cat(all_points, dim=0)        
        coords = torch.cat([all_points, torch.cat(all_indices, dim=0)], dim=-1)

        all_points = all_points.float() / self.full_scale * 2 - 1
        all_points = torch.stack([all_points[:, 1], all_points[:, 0]], dim=-1)        
        all_points = all_points.unsqueeze(0).unsqueeze(0)
        image_features = torch.nn.functional.grid_sample(image_x, all_points)
        image_features = image_features.view(image_x.shape[1], -1).transpose(0, 1)
        x = self.sparse_model((coords, image_features))
        pred = torch.sigmoid(self.pred(x.squeeze(-1).squeeze(-1))).view(-1)
        # cv2.imwrite('test/image.png', (image_x[0, :3].detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))
        # for mask_index, mask in enumerate((x[:, :3].detach().cpu().numpy().transpose((0, 2, 3, 1)) * 255).astype(np.uint8)):
        #     cv2.imwrite('test/mask_' + str(mask_index) + '.png', mask)
        #     continue
        # exit(1)
        return pred    
        
class LoopModel(nn.Module):
    def __init__(self, options, num_classes=1):
        super(LoopModel, self).__init__()
        self.options = options
        self.inplanes = 64

        block = BasicBlock
        layers = [3, 4, 6, 3]
        
        self.relu = nn.ReLU(inplace=True)            
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.image_layer_0 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                                                     bias=False),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))                
        self.image_layer_1 = self._make_layer(block, 64, layers[0])
        self.image_layer_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.image_layer_3 = self._make_layer(block, 256, layers[2], stride=2)
        self.image_layer_4 = self._make_layer(block, 512, layers[3], stride=2)

        self.decoder_4 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2), nn.BatchNorm2d(256), nn.ReLU())
        self.decoder_3 = nn.Sequential(nn.ConvTranspose2d(512, 128, kernel_size=4, padding=1, stride=2), nn.BatchNorm2d(128), nn.ReLU())
        self.decoder_2 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=4, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU())
        self.decoder_1 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU())
        self.decoder_0 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(64, 1, kernel_size=1))

        #self.loop_pred = LoopEncoderMask(64 + 2)        
        #self.edge_pred = EdgeEncoder(64 + 2)
        self.edge_pred = SparseEncoder(64, 127)
        self.loop_pred = SparseEncoder(64, 127)        
        self.multi_loop_pred = SparseEncoder(64, 127)
        #self.multi_loop_pred_mask = SparseEncoder(1, 127)        
        #self.multi_loop_pred = SparseEncoder(4, 256)
        
        # self.coord_layer_1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU())
        # self.coord_layer_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU())
        # self.coord_layer_3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU())
        # self.coord_layer_4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU())

        # self.edge_pred_1 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        # self.edge_pred_2 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        # self.edge_pred_3 = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 1))
        # self.edge_pred_4 = nn.Sequential(nn.Linear(1024, 64), nn.ReLU(), nn.Linear(64, 1))

        # self.loop_pred_1 = LoopEncoder(64 + 2)
        # self.loop_pred_2 = LoopEncoder(128 + 2)
        # self.loop_pred_3 = LoopEncoder(256 + 2)
        # self.loop_pred_4 = LoopEncoder(512 + 2)

        # self.image_aggregate_1 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        # self.image_aggregate_2 = nn.Sequential(nn.Conv2d(128 + 64, 128, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        # self.image_aggregate_3 = nn.Sequential(nn.Conv2d(256 + 64, 256, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        # self.image_aggregate_4 = nn.Sequential(nn.Conv2d(512 + 64, 512, kernel_size=1, bias=False), nn.ReLU(inplace=True))

        # self.coord_aggregate_1 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        # self.coord_aggregate_2 = nn.Sequential(nn.Conv2d(128 + 64, 128, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        # self.coord_aggregate_3 = nn.Sequential(nn.Conv2d(256 + 64, 256, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        # self.coord_aggregate_4 = nn.Sequential(nn.Conv2d(512 + 64, 512, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        
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

        image_x_0 = self.image_layer_0(image)
        image_x_1 = self.image_layer_1(image_x_0)
        image_x_2 = self.image_layer_2(image_x_1)
        image_x_3 = self.image_layer_3(image_x_2)
        image_x_4 = self.image_layer_4(image_x_3)
        image_x_4_up = self.decoder_4(image_x_4)
        image_x_3_up = self.decoder_3(torch.cat([image_x_4_up, image_x_3], dim=1))
        image_x_2_up = self.decoder_2(torch.cat([image_x_3_up, image_x_2], dim=1))        
        image_x_1_up = self.decoder_1(torch.cat([image_x_2_up, image_x_1], dim=1))                                           
        edge_image_pred = torch.sigmoid(self.decoder_0(image_x_1_up))

        #edge_pred = self.edge_pred(image_x_1_up.squeeze(0), edges)
        edge_pred = self.edge_pred(image_x_1_up, edges.unsqueeze(1))
        num_corners = len(corners)
        all_loops = findLoopsModule(edge_pred, edge_corner, num_corners, max_num_loop_corners=len(corners), corners=corners, disable_colinear=True)

        #loop_pred = []
        loop_corner_indices = []
        loop_corners = []        
        loop_edge_masks = []
        loop_confidence = []
        max_confidence = max([confidence.max() for confidence, loops in all_loops])
        for confidence, loops in all_loops:
            for loop_index in range(len(loops)):
                if confidence[loop_index] < min(0.5, max_confidence):
                    continue
                loop = loops[loop_index]
                loop_corner_indices.append(loop)
                                
                loop_corners.append(corners[loop])
                #loop_pred.append(loop_confidence)

                loop_corner_pairs = torch.cat([torch.stack([loop, torch.cat([loop[-1:], loop[:-1]], dim=0)], dim=-1), torch.stack([torch.cat([loop[-1:], loop[:-1]], dim=0), loop], dim=-1)], dim=0)
                loop_edge_mask = (edge_corner.unsqueeze(1) == loop_corner_pairs).min(-1)[0]
                loop_edge_masks.append(loop_edge_mask.max(-1)[0].float())
                
                loop_confidence.append(confidence[loop_index])
                # loop_edge_mask = torch.max(loop_edge_mask[:, :loop_edge_mask.shape[-1] // 2], loop_edge_mask[:, loop_edge_mask.shape[-1] // 2:])
                # loop_edge = loop_edge_mask.max(0)[1]                                                
                continue
            continue
        loop_info = list(zip(loop_corner_indices, loop_edge_masks, loop_corners))
        loop_confidence = torch.stack(loop_confidence)
        
        loop_edge_masks = torch.stack(loop_edge_masks, dim=0)
        #loop_pred = self.loop_pred(image_x_1_up[0], loop_corners)
        loop_edges = [edges[loop_edge_masks[index].nonzero()[:, 0]] for index in range(len(loop_edge_masks))]        
        loop_pred = self.loop_pred(image_x_1_up, loop_edges)
        
        #print(edge_pred)
        #multi_loop_edge_masks = findMultiLoopsModule(loop_confidence, loop_info, edge_corner, num_corners, self.options.max_num_loop_corners, corners=corners, disable_colinear=True, edge_pred=edge_pred)
        multi_loop_edge_masks = findMultiLoopsModule(loop_pred, loop_info, edge_corner, num_corners, self.options.max_num_loop_corners, corners=corners, disable_colinear=True, edge_pred=edge_pred)
        multi_loop_edges = [edges[multi_loop_edge_masks[index].nonzero()[:, 0]] for index in range(len(multi_loop_edge_masks))]
        #print(multi_loop_edge_masks)
        multi_loop_pred = self.multi_loop_pred(image_x_1_up, multi_loop_edges)
        #multi_loop_pred_mask = self.multi_loop_pred_mask(torch.ones((1, 1, 128, 128)).cuda(), multi_loop_edges)        
        #max_mask = (torch.arange(len(multi_loop_pred)).cuda().long() == multi_loop_pred.max(0)[1]).float()
        #multi_loop_pred = multi_loop_pred * (1 - max_mask) + max_mask
        #multi_loop_pred = max_mask
        #multi_loop_pred = torch.cat([torch.ones(1), torch.zeros(len(multi_loop_edges) - 1)], dim=0).cuda()
        multi_loop_predictions = torch.LongTensor([multi_loop_pred.max(0)[1], 0, (multi_loop_edge_masks * (edge_pred - 0.5)).sum(-1).max(0)[1]]).cuda()
        return edge_image_pred, [[edge_pred, loop_pred, loop_edge_masks, multi_loop_pred, multi_loop_edge_masks, multi_loop_predictions]]
