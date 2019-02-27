import torch
import numpy as np

def findLoopsModule(edge_confidence, edge_corner, num_corners, max_num_loop_corners=10, confidence_threshold=0, corners=None, disable_colinear=True):
    ## The confidence of connecting two corners
    corner_confidence = torch.zeros(num_corners, num_corners).cuda()
    corner_confidence[edge_corner[:, 0], edge_corner[:, 1]] = edge_confidence
    corner_confidence[edge_corner[:, 1], edge_corner[:, 0]] = edge_confidence
    corner_confidence = corner_confidence - torch.diag(torch.ones(num_corners).cuda()) * max_num_loop_corners
    
    corner_range = torch.arange(num_corners).cuda().long()
    corner_range_map = corner_range.view((-1, 1)).repeat((1, num_corners))

    ## Paths = [(path_confidence, path_corners)] where the ith path has a length of (i + 1), path_confidence is the summation of confidence along the path between two corners, and path_corners is visited corners along the path
    paths = [(corner_confidence, corner_range_map.unsqueeze(-1))]
    dot_product_threshold = np.cos(np.deg2rad(20))
    while len(paths) < max_num_loop_corners - 1:
        path_confidence, path_corners = paths[-1]
        total_confidence = path_confidence.unsqueeze(-1) + corner_confidence
        visited_mask = (path_corners.unsqueeze(-1) == corner_range).max(-2)[0]
        if disable_colinear and path_corners.shape[-1] > 1:
            prev_edge = corners[path_corners[:, :, -1]] - corners[path_corners[:, :, -2]]
            prev_edge = prev_edge / torch.norm(prev_edge, dim=-1, keepdim=True)
            current_edge = corners - corners[path_corners[:, :, -1]].unsqueeze(-2)
            current_edge = current_edge / torch.norm(current_edge, dim=-1, keepdim=True)
            dot_product = (prev_edge.unsqueeze(-2) * current_edge).sum(-1)
            colinear_mask = torch.abs(dot_product) > dot_product_threshold
            visited_mask = visited_mask | colinear_mask
            pass
        visited_mask = visited_mask.float()
        #visited_mask = torch.max(visited_mask, (prev_corner.unsqueeze(1) == corner_range.unsqueeze(-1)).float())
        total_confidence = total_confidence * (1 - visited_mask) - (max_num_loop_corners) * visited_mask
        #print(path_confidence, total_confidence, visited_mask)        
        #print(total_confidence, visited_mask)
        new_path_confidence, last_corner = total_confidence.max(1)
        existing_path = path_corners[corner_range_map.view(-1), last_corner.view(-1)].view((num_corners, num_corners, -1))
        new_path_corners = torch.cat([existing_path, last_corner.unsqueeze(-1)], dim=-1)
        paths.append((new_path_confidence, new_path_corners))
        continue
    #print(paths)
    ## Find closed loops by adding the starting point to the path
    paths = paths[1:]
    loops = []
    for index, (path_confidence, path_corners) in enumerate(paths):
        total_confidence = path_confidence.unsqueeze(-1) + corner_confidence
        visited_mask = (path_corners[:, :, 1:].unsqueeze(-1) == corner_range).float().max(-2)[0]
        total_confidence = total_confidence * (1 - visited_mask) - (max_num_loop_corners) * visited_mask
        #print(total_confidence, visited_mask)
        loop_confidence, last_corner = total_confidence.max(1)
        last_corner = last_corner.diagonal()
        loop = path_corners[corner_range, last_corner].view((num_corners, -1))
        loop = torch.cat([loop, last_corner.unsqueeze(-1)], dim=-1)
        loop_confidence = loop_confidence.diagonal() / (index + 3)
        mask = loop_confidence > confidence_threshold
        if mask.sum().item() == 0:
            loop_confidence, index = loop_confidence.max(0, keepdim=True)
            index = index.squeeze().item()
            loops.append((loop_confidence, loop[index:index + 1]))
            continue
        loop_confidence = loop_confidence[mask]
        loop = loop[mask]
        if len(loop) > 0:
            same_mask = torch.abs(loop.unsqueeze(1).unsqueeze(0) - loop.unsqueeze(-1).unsqueeze(1)).min(-1)[0].max(-1)[0] == 0
            loop_range = torch.arange(len(loop)).long().cuda()
            same_mask = same_mask & (loop_range.unsqueeze(-1) > loop_range.unsqueeze(0))
            same_mask = same_mask.max(-1)[0]^1
            loops.append((loop_confidence[same_mask], loop[same_mask]))
        else:
            loops.append((loop_confidence, loop))
            pass
        continue
    return loops

    # #paths = [(path[0], torch.cat([path[1], corner_range_map.transpose(0, 1).unsqueeze(-1)], dim=-1)) for path in paths]
    # loops = []
    # print(paths)
    # for path in paths:
    #     confidence = path[0].diagonal()
    #     path = path[1].diagonal().transpose(0, 1)
    #     mask = confidence > 0
    #     loops.append((confidence[mask], path[mask]))
    #     continue
    # print(loops)
    # exit(1)
    # return loops


## The pyramid module from pyramid scene parsing
class PyramidModule(torch.nn.Module):
    def __init__(self, options, in_planes, middle_planes, scales=[32, 16, 8, 4]):
        super(PyramidModule, self).__init__()
        
        self.pool_1 = torch.nn.AvgPool2d((scales[0], scales[0]))
        self.pool_2 = torch.nn.AvgPool2d((scales[1], scales[1]))        
        self.pool_3 = torch.nn.AvgPool2d((scales[2], scales[2]))
        self.pool_4 = torch.nn.AvgPool2d((scales[3], scales[3]))        
        self.conv_1 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_2 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_3 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_4 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(scales[0], scales[0]), mode='bilinear')
        return
    
    def forward(self, inp):
        x_1 = self.upsample(self.conv_1(self.pool_1(inp)))
        x_2 = self.upsample(self.conv_2(self.pool_2(inp)))
        x_3 = self.upsample(self.conv_3(self.pool_3(inp)))
        x_4 = self.upsample(self.conv_4(self.pool_4(inp)))
        out = torch.cat([inp, x_1, x_2, x_3, x_4], dim=1)
        return out

## Conv + bn + relu
class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, mode='conv'):
        super(ConvBlock, self).__init__()
       
        if padding == None:
            padding = (kernel_size - 1) // 2
            pass
        if mode == 'conv':
            self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        elif mode == 'deconv':
            self.conv = torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        elif mode == 'conv_3d':
            self.conv = torch.nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        elif mode == 'deconv_3d':
            self.conv = torch.nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        else:
            print('conv mode not supported', mode)
            exit(1)
            pass
        if '3d' not in mode:
            self.bn = torch.nn.BatchNorm2d(out_planes)
        else:
            self.bn = torch.nn.BatchNorm3d(out_planes)
            pass
        self.relu = torch.nn.ReLU(inplace=True)
        return
   
    def forward(self, inp):
        #return self.relu(self.conv(inp))       
        return self.relu(self.bn(self.conv(inp)))    
