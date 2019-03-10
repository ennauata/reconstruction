import torch
import numpy as np
import cv2

def findLoopsModule(edge_confidence, edge_corner, num_corners, max_num_loop_corners=12, confidence_threshold=0, corners=None, disable_colinear=True, disable_intersection=True):
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

        if disable_colinear and path_corners.shape[-1] > 0:
            #prev_edge = corners[path_corners[:, :, -1]] - corners[path_corners[:, :, -2]]
            #prev_edge = corners.unsqueeze(-2) - corners[path_corners[:, :, -1]].squeeze(-3)
            prev_edge = corners - corners[path_corners[:, :, -1]]
            prev_edge = prev_edge / torch.clamp(torch.norm(prev_edge, dim=-1, keepdim=True), min=1e-4)
            #current_edge = corners - corners[path_corners[:, :, -1]].unsqueeze(-2)
            current_edge = corners.unsqueeze(-2) - corners.unsqueeze(-3)
            current_edge = current_edge / torch.clamp(torch.norm(current_edge, dim=-1, keepdim=True), min=1e-4)
            dot_product = (prev_edge.unsqueeze(-2) * current_edge).sum(-1)
            #print(prev_edge, current_edge)
            #print(dot_product)
            colinear_mask = torch.abs(dot_product) > dot_product_threshold
            #print(path_corners.shape, visited_mask.shape, colinear_mask.shape)
            visited_mask_ori = visited_mask            
            visited_mask = visited_mask | colinear_mask
            pass
        
        if disable_intersection and path_corners.shape[-1] > 1:
            current_edge = torch.cat([corners.unsqueeze(-2).repeat((1, len(corners), 1)), corners.unsqueeze(-3).repeat((len(corners), 1, 1))], dim=-1).unsqueeze(0)
            current_edge_normal = torch.stack([current_edge[:, :, :, 3] - current_edge[:, :, :, 1], current_edge[:, :, :, 0] - current_edge[:, :, :, 2]], dim=-1)

            # final_edge = current_edge.transpose(0, 1)
            # final_edge_normal = current_edge_normal.transpose(0, 1)
            # prev_edge = torch.cat([corners.repeat((len(corners), 1, 1)), corners[path_corners[:, :, -1]]], dim=-1).unsqueeze(-2)
            # prev_edge_normal = torch.stack([prev_edge[:, :, :, 3] - prev_edge[:, :, :, 1], prev_edge[:, :, :, 0] - prev_edge[:, :, :, 2]], dim=-1)
            # invalid_mask_1 = ((prev_edge[:, :, :, :2] - final_edge[:, :, :, :2]) * final_edge_normal).sum(-1) * ((prev_edge[:, :, :, 2:4] - final_edge[:, :, :, :2]) * final_edge_normal).sum(-1) < 0
            # invalid_mask_2 = ((final_edge[:, :, :, :2] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) * ((final_edge[:, :, :, 2:4] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) < 0
            # intersection_mask = invalid_mask_1 & invalid_mask_2
            
            for index in range(path_corners.shape[-1] - 1):
                prev_edge = torch.cat([corners[path_corners[:, :, index + 1]], corners[path_corners[:, :, index]]], dim=-1).unsqueeze(-2)
                prev_edge_normal = torch.stack([prev_edge[:, :, :, 3] - prev_edge[:, :, :, 1], prev_edge[:, :, :, 0] - prev_edge[:, :, :, 2]], dim=-1)                
                invalid_mask_1 = ((prev_edge[:, :, :, :2] - current_edge[:, :, :, :2]) * current_edge_normal).sum(-1) * ((prev_edge[:, :, :, 2:4] - current_edge[:, :, :, :2]) * current_edge_normal).sum(-1) < 0
                invalid_mask_2 = ((current_edge[:, :, :, :2] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) * ((current_edge[:, :, :, 2:4] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) < 0
                if index == 0:
                    intersection_mask = invalid_mask_1 & invalid_mask_2
                else:
                    intersection_mask = intersection_mask | (invalid_mask_1 & invalid_mask_2)
                    pass
                continue
            visited_mask = visited_mask | intersection_mask
            pass

        if False and disable_colinear and path_corners.shape[-1] > 0:
            visited_mask_debug = visited_mask.float()
            #visited_mask = torch.max(visited_mask, (prev_corner.unsqueeze(1) == corner_range.unsqueeze(-1)).float())
            total_confidence_debug = total_confidence * (1 - visited_mask_debug) - (max_num_loop_corners) * visited_mask_debug
            #print(path_confidence, total_confidence, visited_mask)        
            #print(total_confidence, visited_mask)
            _, last_corner = total_confidence_debug.max(1)
            
            visited_mask_debug = visited_mask_ori.float()
            total_confidence_debug = total_confidence * (1 - visited_mask_debug) - (max_num_loop_corners) * visited_mask_debug
            _, last_corner_ori = total_confidence_debug.max(1)
        
            mask = last_corner != last_corner_ori
            #mask, index = mask.max(-1)
            if mask.sum() > 0:
                #print(path_corners[mask])                                
                #print(dot_product[mask])                
                print('colinear', torch.cat([path_corners[mask][:, -2:], last_corner_ori[mask].view((-1, 1)), last_corner[mask].view((-1, 1))], dim=1))
                pass
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
    
    ## Find closed loops by adding the starting point to the path    
    paths = paths[1:]
    loops = []
    for index, (path_confidence, path_corners) in enumerate(paths):
        total_confidence = path_confidence.unsqueeze(-1) + corner_confidence
        visited_mask = (path_corners[:, :, 1:].unsqueeze(-1) == corner_range).max(-2)[0]

        if disable_colinear and path_corners.shape[-1] > 0:
            prev_edge = corners - corners[path_corners[:, :, -1]]
            prev_edge = prev_edge / torch.clamp(torch.norm(prev_edge, dim=-1, keepdim=True), min=1e-4)
            current_edge = corners.unsqueeze(-2) - corners.unsqueeze(-3)
            current_edge = current_edge / torch.clamp(torch.norm(current_edge, dim=-1, keepdim=True), min=1e-4)
            dot_product = (prev_edge.unsqueeze(-2) * current_edge).sum(-1)
            colinear_mask = torch.abs(dot_product) > dot_product_threshold
            visited_mask_ori = visited_mask            
            visited_mask = visited_mask | colinear_mask
            pass
        
        if disable_intersection and path_corners.shape[-1] > 1:
            current_edge = torch.cat([corners.unsqueeze(-2).repeat((1, len(corners), 1)), corners.unsqueeze(-3).repeat((len(corners), 1, 1))], dim=-1).unsqueeze(0)
            current_edge_normal = torch.stack([current_edge[:, :, :, 3] - current_edge[:, :, :, 1], current_edge[:, :, :, 0] - current_edge[:, :, :, 2]], dim=-1)

            # final_edge = current_edge.transpose(0, 1)
            # final_edge_normal = current_edge_normal.transpose(0, 1)
            # prev_edge = torch.cat([corners.repeat((len(corners), 1, 1)), corners[path_corners[:, :, -1]]], dim=-1).unsqueeze(-2)
            # prev_edge_normal = torch.stack([prev_edge[:, :, :, 3] - prev_edge[:, :, :, 1], prev_edge[:, :, :, 0] - prev_edge[:, :, :, 2]], dim=-1)
            # invalid_mask_1 = ((prev_edge[:, :, :, :2] - final_edge[:, :, :, :2]) * final_edge_normal).sum(-1) * ((prev_edge[:, :, :, 2:4] - final_edge[:, :, :, :2]) * final_edge_normal).sum(-1) < 0
            # invalid_mask_2 = ((final_edge[:, :, :, :2] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) * ((final_edge[:, :, :, 2:4] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) < 0
            # intersection_mask = invalid_mask_1 & invalid_mask_2

            for index in range(path_corners.shape[-1] - 1):
                prev_edge = torch.cat([corners[path_corners[:, :, index + 1]], corners[path_corners[:, :, index]]], dim=-1).unsqueeze(-2)
                prev_edge_normal = torch.stack([prev_edge[:, :, :, 3] - prev_edge[:, :, :, 1], prev_edge[:, :, :, 0] - prev_edge[:, :, :, 2]], dim=-1)                
                invalid_mask_1 = ((prev_edge[:, :, :, :2] - current_edge[:, :, :, :2]) * current_edge_normal).sum(-1) * ((prev_edge[:, :, :, 2:4] - current_edge[:, :, :, :2]) * current_edge_normal).sum(-1) < 0
                invalid_mask_2 = ((current_edge[:, :, :, :2] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) * ((current_edge[:, :, :, 2:4] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) < 0
                if index == 0:
                    intersection_mask = invalid_mask_1 & invalid_mask_2
                else:
                    intersection_mask = intersection_mask | (invalid_mask_1 & invalid_mask_2)
                    pass
                continue
            
            visited_mask = visited_mask | intersection_mask
            pass

        visited_mask = visited_mask.float()
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

        #print(index)
        #print(path_corners)
        #print(index, loop)
        
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

    #torch.save(paths, 'test/paths.pth')
    #torch.save(corner_confidence, 'test/confidence.pth')
    #torch.save((edge_pred, edge_corner, num_corners, corners), 'test/debug.pth')
    #print(corner_confidence)
    #print(corner_confidence[4, 5])    
    return loops

def findMultiLoopsModule(loop_confidence, loop_info, edge_corner, num_corners, max_num_loop_corners=10, confidence_threshold=0, corners=None, disable_colinear=True, edge_pred=None):

    loop_edges = []
    loop_masks = []

    mask_size = 64
    Us = torch.arange(mask_size).float().cuda().repeat((mask_size, 1))
    Vs = torch.arange(mask_size).float().cuda().unsqueeze(-1).repeat((1, mask_size))
    UVs = torch.stack([Vs, Us], dim=-1)
    
    for loop in loop_info:
        loop_corners = loop[2]

        edges = torch.stack([loop_corners, torch.cat([loop_corners[1:], loop_corners[:1]], dim=0)], dim=-2)
        edges = edges * mask_size
        edge_normals = edges[:, 1] - edges[:, 0]
        edge_normals = torch.stack([edge_normals[:, 1], -edge_normals[:, 0]], dim=-1)
        edge_normals = edge_normals / torch.clamp(torch.norm(edge_normals, dim=-1, keepdim=True), min=1e-4)
        edge_normals = edge_normals * ((edge_normals[:, 1:2] > 0).float() * 2 - 1)
        direction_mask = ((UVs.unsqueeze(-2) - edges[:, 0]) * edge_normals).sum(-1) > 0
        range_mask = (UVs[:, :, 0].unsqueeze(-1) > torch.min(edges[:, 0, 0], edges[:, 1, 0])) & (UVs[:, :, 0].unsqueeze(-1) <= torch.max(edges[:, 0, 0], edges[:, 1, 0]))
        flags = direction_mask & range_mask
        mask = flags.sum(-1) % 2 == 1
        loop_masks.append(mask)            
        continue

    loop_edges = torch.stack([loop[1] for loop in loop_info])
    loop_masks = torch.stack(loop_masks)

    num_loops = len(loop_confidence)
    _, order = torch.sort(loop_confidence, descending=True)
    loop_confidence = loop_confidence[order]
    loop_edges = loop_edges[order]
    loop_masks = loop_masks[order]

    same_edge_mask = ((loop_edges.unsqueeze(1) == loop_edges) & (loop_edges > 0.5)).max(-1)[0]
    min_area = loop_masks.sum(-1).sum(-1).min()
    area_threshold = min(min_area // 2, 50)
    overlap_mask = (loop_masks.unsqueeze(1) * loop_masks).sum(-1).sum(-1) > area_threshold

    conflict_mask = same_edge_mask & overlap_mask

    best_multi_loop = (loop_confidence[0], torch.LongTensor([0]).cuda(), conflict_mask[0])
    new_multi_loops = [best_multi_loop]
    ori_multi_loops = [[]]
    for loop_index in range(1, num_loops):
        ori_multi_loops.append(best_multi_loop)
        
        best_prev_multi_loop = (-1, None, None)
        for prev_multi_loop in new_multi_loops:
            #print(prev_multi_loop[2])
            if prev_multi_loop[2][loop_index]^1 and prev_multi_loop[0] > best_prev_multi_loop[0]:
                best_prev_multi_loop = prev_multi_loop
                pass
            continue
        if best_prev_multi_loop[0] < 0:
            new_multi_loops.append((loop_confidence[loop_index], torch.LongTensor([loop_index]).cuda(), conflict_mask[loop_index]))
        else:
            new_multi_loop = (best_prev_multi_loop[0] + loop_confidence[loop_index], torch.cat([best_prev_multi_loop[1], torch.LongTensor([loop_index]).cuda()], dim=0), best_prev_multi_loop[2] | conflict_mask[loop_index])
            new_multi_loops.append(new_multi_loop)
            if new_multi_loop[0] > best_multi_loop[0]:
                best_multi_loop = new_multi_loop
                pass
            pass
        continue
    #multi_loop_confidence = torch.stack([new_multi_loop[0] for new_multi_loop in new_multi_loops])
    multi_loop_edge_mask = torch.stack([loop_edges[new_multi_loop[1]].max(0)[0] for new_multi_loop in new_multi_loops])
    multi_loop_confidence = (multi_loop_edge_mask * (edge_pred - 0.5)).sum(-1)
    _, multi_loop_order = torch.sort(multi_loop_confidence, descending=True)

    if len(conflict_mask) > 1 and False:
        print(corners)
        print(loop_confidence)
        print(edge_corner)
        print([(index, loop[2]) for index, loop in enumerate(loop_info)])
        print(same_edge_mask)        
        print(overlap_mask)
        loop_masks = loop_masks.detach().cpu().numpy()
        for index, mask in enumerate(loop_masks):
            mask_image = (mask * 255).astype(np.uint8)
            mask_image = np.stack([mask_image, mask_image, mask_image], axis=-1)
            mask_image = cv2.resize(mask_image, (256, 256))
            corners = np.clip((loop_info[order[index]][2].detach().cpu().numpy() * mask_image.shape[1]).round().astype(np.int32), 0, mask_image.shape[1] - 1)
            for corner_index, corner in enumerate(corners):
                cv2.line(mask_image, (corner[1], corner[0]), (corners[(corner_index + 1) % len(corners)][1], corners[(corner_index + 1) % len(corners)][0]), thickness=1, color=(255, 0, 0))
                continue
            cv2.imwrite('test/mask_' + str(index) + '.png', mask_image)
            continue
        print(conflict_mask)        
        print(new_multi_loops)
        print(multi_loop_confidence)
        print(multi_loop_order, new_multi_loops[multi_loop_order[0]][1])
        #print(multi_loop_edge_mask[order[:10]])        
        #exit(1)
        pass
    
    return multi_loop_edge_mask[multi_loop_order[:10]]


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
