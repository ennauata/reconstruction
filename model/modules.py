import torch
import numpy as np
import cv2

dot_product_threshold = np.cos(np.deg2rad(20))

def compute_edge_conflict_mask(edges):
    edge_normals = edges[:, 1] - edges[:, 0]
    edge_lengths = torch.norm(edge_normals, dim=-1)
    edge_normals = edge_normals / edge_lengths.unsqueeze(-1)
    edge_normals = torch.stack([edge_normals[:, 1], -edge_normals[:, 0]], dim=-1)
    edges_1 = edges.unsqueeze(1)
    edges_2 = edges.unsqueeze(0)
    edge_normals_1 = edge_normals.unsqueeze(1)      
    edge_normals_2 = edge_normals.unsqueeze(0)
    distance_1_1 = ((edges_1[:, :, 0] - edges_2[:, :, 0]) * edge_normals_2).sum(-1)
    distance_1_2 = ((edges_1[:, :, 1] - edges_2[:, :, 0]) * edge_normals_2).sum(-1)
    invalid_mask_1 = (distance_1_1 * distance_1_2 < 0) & (torch.min(torch.abs(distance_1_1), torch.abs(distance_1_2)) > 0.02)
    distance_2_1 = ((edges_2[:, :, 0] - edges_1[:, :, 0]) * edge_normals_1).sum(-1)
    distance_2_2 = ((edges_2[:, :, 1] - edges_1[:, :, 0]) * edge_normals_1).sum(-1)
    invalid_mask_2 = (distance_2_1 * distance_2_2 < 0) & (torch.min(torch.abs(distance_2_1), torch.abs(distance_2_2)) > 0.02)
    intersection_mask = invalid_mask_1 & invalid_mask_2
    return intersection_mask

def findLoopsModule(edge_confidence, edge_corner, num_corners, max_num_loop_corners=12, confidence_threshold=0.5, corners=None, disable_colinear=True, disable_intersection=True):

    ## The confidence of connecting two corners
    corner_confidence = torch.zeros(num_corners, num_corners).cuda()
    corner_confidence[edge_corner[:, 0], edge_corner[:, 1]] = edge_confidence
    corner_confidence[edge_corner[:, 1], edge_corner[:, 0]] = edge_confidence
    corner_confidence = corner_confidence - torch.diag(torch.ones(num_corners).cuda()) * max_num_loop_corners

    corner_range = torch.arange(num_corners).cuda().long()
    corner_range_map = corner_range.view((-1, 1)).repeat((1, num_corners))

    ## Paths = [(path_confidence, path_corners)] where the ith path has a length of (i + 1), path_confidence is the summation of confidence along the path between two corners, and path_corners is visited corners along the path
    paths = [(corner_confidence, corner_range_map.unsqueeze(-1))]

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
    for path_index, (path_confidence, path_corners) in enumerate(paths):
        total_confidence = path_confidence + corner_confidence
        visited_mask = (path_corners[:, :, 1:] == corner_range.unsqueeze(-1).unsqueeze(-1)).max(-1)[0]

        if disable_colinear:
            prev_edge = corners - corners[path_corners[:, :, -1]]
            prev_edge = prev_edge / torch.clamp(torch.norm(prev_edge, dim=-1, keepdim=True), min=1e-4)
            current_edge = corners.unsqueeze(-2) - corners.unsqueeze(-3)
            current_edge = current_edge / torch.clamp(torch.norm(current_edge, dim=-1, keepdim=True), min=1e-4)
            dot_product = (prev_edge * current_edge).sum(-1)
            colinear_mask = torch.abs(dot_product) > dot_product_threshold

            prev_edge = corners[path_corners[:, :, 1]] - corners[path_corners[:, :, 0]]
            prev_edge = prev_edge / torch.clamp(torch.norm(prev_edge, dim=-1, keepdim=True), min=1e-4)
            dot_product = (prev_edge * current_edge).sum(-1)
            colinear_mask = colinear_mask | (torch.abs(dot_product) > dot_product_threshold)
            
            visited_mask = visited_mask | colinear_mask
            pass
        
        if disable_intersection:
            current_edge = torch.cat([corners.unsqueeze(-2).repeat((1, len(corners), 1)), corners.unsqueeze(-3).repeat((len(corners), 1, 1))], dim=-1)
            current_edge_normal = torch.stack([current_edge[:, :, 3] - current_edge[:, :, 1], current_edge[:, :, 0] - current_edge[:, :, 2]], dim=-1)

            # final_edge = current_edge.transpose(0, 1)
            # final_edge_normal = current_edge_normal.transpose(0, 1)
            # prev_edge = torch.cat([corners.repeat((len(corners), 1, 1)), corners[path_corners[:, :, -1]]], dim=-1).unsqueeze(-2)
            # prev_edge_normal = torch.stack([prev_edge[:, :, :, 3] - prev_edge[:, :, :, 1], prev_edge[:, :, :, 0] - prev_edge[:, :, :, 2]], dim=-1)
            # invalid_mask_1 = ((prev_edge[:, :, :, :2] - final_edge[:, :, :, :2]) * final_edge_normal).sum(-1) * ((prev_edge[:, :, :, 2:4] - final_edge[:, :, :, :2]) * final_edge_normal).sum(-1) < 0
            # invalid_mask_2 = ((final_edge[:, :, :, :2] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) * ((final_edge[:, :, :, 2:4] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) < 0
            # intersection_mask = invalid_mask_1 & invalid_mask_2

            for index in range(path_corners.shape[-1] - 1):
                prev_edge = torch.cat([corners[path_corners[:, :, index + 1]], corners[path_corners[:, :, index]]], dim=-1)
                prev_edge_normal = torch.stack([prev_edge[:, :, 3] - prev_edge[:, :, 1], prev_edge[:, :, 0] - prev_edge[:, :, 2]], dim=-1)                
                invalid_mask_1 = ((prev_edge[:, :, :2] - current_edge[:, :, :2]) * current_edge_normal).sum(-1) * ((prev_edge[:, :, 2:4] - current_edge[:, :, :2]) * current_edge_normal).sum(-1) < 0
                invalid_mask_2 = ((current_edge[:, :, :2] - prev_edge[:, :, :2]) * prev_edge_normal).sum(-1) * ((current_edge[:, :, 2:4] - prev_edge[:, :, :2]) * prev_edge_normal).sum(-1) < 0
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

        loop = torch.cat([path_corners, corner_range.unsqueeze(-1).repeat((len(corner_range), 1, 1))], dim=-1)
        loop_confidence = total_confidence / (path_index + 3)
        #mask = (loop_confidence > confidence_threshold) & (corner_confidence > 0.5)
        mask = loop_confidence > confidence_threshold

        if loop.shape[-1] == 4 and False:
            print(loop)
            print(loop_confidence)
            print(corner_confidence)
            print(corner_confidence[0, 3], loop_confidence[0, 3])
            pass
        
        if mask.sum().item() == 0:
            # loop_confidence, index = loop_confidence.max(0, keepdim=True)
            # index = index.squeeze().item()
            # loops.append((loop_confidence, loop[index:index + 1]))
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
            if path_index == 1 and False:
                print(total_confidence[0, 2], visited_mask[0, 2])
                print(paths[0][0][0, 1], paths[0][1][0, 1])
                print(paths[0][0][0, 5], paths[0][1][0, 5])                        
                print(path_corners[0, 2])
                print(loop)
                print(loop[same_mask])
                exit(1)
                pass            
            loops.append((loop_confidence[same_mask], loop[same_mask]))
        else:
            #loops.append((loop_confidence, loop))
            pass
        continue
    if len(loops) == 0:
        _, loop_edge = torch.sort(edge_confidence, descending=True)
        loop_edge = loop_edge[:num_corners]

        ## Find loop corners along the loop
        loop_corners = edge_corner[loop_edge]
        if torch.norm(corners[loop_corners[0, 0]] - corners[loop_corners[-1]], dim=-1).min() > torch.norm(corners[loop_corners[0, 1]] - corners[loop_corners[-1]], dim=-1).min():
            loop_corners = torch.cat([torch.cat([loop_corners[:1, 1:2], loop_corners[:1, 0:1]], dim=1), loop_corners[1:]], dim=0)
            pass
        loop_corners_prev = torch.cat([loop_corners[-1:], loop_corners[:-1]], dim=0)
        distances = torch.stack([torch.norm(corners[loop_corners[:, 0]] - corners[loop_corners_prev[:, 1]], dim=-1), torch.norm(corners[loop_corners[:, 1]] - corners[loop_corners_prev[:, 1]], dim=-1)], dim=-1)
        corner_indices = torch.cat([torch.zeros(1).long().cuda(), distances.min(-1)[1][1:]], dim=0)
        corner_indices = torch.cumsum(corner_indices, dim=0) % 2
        loop_corners = loop_corners[torch.arange(len(distances)).cuda().long(), corner_indices]
        loops.append((edge_confidence[loop_edge].mean(0, keepdim=True), loop_corners.unsqueeze(0)))
        pass
    
    # paths = paths[1:]
    # loops = []
    # for path_index, (path_confidence, path_corners) in enumerate(paths):
    #     total_confidence = path_confidence.unsqueeze(-1) + corner_confidence
    #     visited_mask = (path_corners[:, :, 1:].unsqueeze(-1) == corner_range).max(-2)[0]

    #     if disable_colinear and path_corners.shape[-1] > 0:
    #         prev_edge = corners - corners[path_corners[:, :, -1]]
    #         prev_edge = prev_edge / torch.clamp(torch.norm(prev_edge, dim=-1, keepdim=True), min=1e-4)
    #         current_edge = corners.unsqueeze(-2) - corners.unsqueeze(-3)
    #         current_edge = current_edge / torch.clamp(torch.norm(current_edge, dim=-1, keepdim=True), min=1e-4)
    #         dot_product = (prev_edge.unsqueeze(-2) * current_edge).sum(-1)
    #         colinear_mask = torch.abs(dot_product) > dot_product_threshold
    #         visited_mask_ori = visited_mask            
    #         visited_mask = visited_mask | colinear_mask
    #         pass
        
    #     if disable_intersection and path_corners.shape[-1] > 1:
    #         current_edge = torch.cat([corners.unsqueeze(-2).repeat((1, len(corners), 1)), corners.unsqueeze(-3).repeat((len(corners), 1, 1))], dim=-1).unsqueeze(0)
    #         current_edge_normal = torch.stack([current_edge[:, :, :, 3] - current_edge[:, :, :, 1], current_edge[:, :, :, 0] - current_edge[:, :, :, 2]], dim=-1)

    #         # final_edge = current_edge.transpose(0, 1)
    #         # final_edge_normal = current_edge_normal.transpose(0, 1)
    #         # prev_edge = torch.cat([corners.repeat((len(corners), 1, 1)), corners[path_corners[:, :, -1]]], dim=-1).unsqueeze(-2)
    #         # prev_edge_normal = torch.stack([prev_edge[:, :, :, 3] - prev_edge[:, :, :, 1], prev_edge[:, :, :, 0] - prev_edge[:, :, :, 2]], dim=-1)
    #         # invalid_mask_1 = ((prev_edge[:, :, :, :2] - final_edge[:, :, :, :2]) * final_edge_normal).sum(-1) * ((prev_edge[:, :, :, 2:4] - final_edge[:, :, :, :2]) * final_edge_normal).sum(-1) < 0
    #         # invalid_mask_2 = ((final_edge[:, :, :, :2] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) * ((final_edge[:, :, :, 2:4] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) < 0
    #         # intersection_mask = invalid_mask_1 & invalid_mask_2

    #         for index in range(path_corners.shape[-1] - 1):
    #             prev_edge = torch.cat([corners[path_corners[:, :, index + 1]], corners[path_corners[:, :, index]]], dim=-1).unsqueeze(-2)
    #             prev_edge_normal = torch.stack([prev_edge[:, :, :, 3] - prev_edge[:, :, :, 1], prev_edge[:, :, :, 0] - prev_edge[:, :, :, 2]], dim=-1)                
    #             invalid_mask_1 = ((prev_edge[:, :, :, :2] - current_edge[:, :, :, :2]) * current_edge_normal).sum(-1) * ((prev_edge[:, :, :, 2:4] - current_edge[:, :, :, :2]) * current_edge_normal).sum(-1) < 0
    #             invalid_mask_2 = ((current_edge[:, :, :, :2] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) * ((current_edge[:, :, :, 2:4] - prev_edge[:, :, :, :2]) * prev_edge_normal).sum(-1) < 0
    #             if index == 0:
    #                 intersection_mask = invalid_mask_1 & invalid_mask_2
    #             else:
    #                 intersection_mask = intersection_mask | (invalid_mask_1 & invalid_mask_2)
    #                 pass
    #             continue
            
    #         visited_mask = visited_mask | intersection_mask
    #         pass

    #     visited_mask = visited_mask.float()
    #     total_confidence = total_confidence * (1 - visited_mask) - (max_num_loop_corners) * visited_mask
    #     #print(total_confidence, visited_mask)
        
    #     # loop_confidence, last_corner = total_confidence.max(1)
    #     # last_corner = last_corner.diagonal()
    #     # loop = path_corners[corner_range, last_corner].view((num_corners, -1))
    #     # loop = torch.cat([loop, last_corner.unsqueeze(-1)], dim=-1)
    #     # loop_confidence = loop_confidence.diagonal() / (path_index + 3)

    #     loop = torch.cat([path_corners, corner_range.unsqueeze(-1).repeat((len(corner_range), 1, 1))], dim=-1)
    #     loop_confidence = total_confidence[corner_range, :, corner_range] / (path_index + 3)
        
    #     mask = loop_confidence > confidence_threshold
    #     if mask.sum().item() == 0:
    #         # loop_confidence, index = loop_confidence.max(0, keepdim=True)
    #         # index = index.squeeze().item()
    #         # loops.append((loop_confidence, loop[index:index + 1]))
    #         continue
        
    #     loop_confidence = loop_confidence[mask]
    #     loop = loop[mask]
    #     print(loop)

    #     if path_index == 1:
    #         print(loop)
    #         print(loop_confidence)
    #         print(loop[loop_confidence > 0.5])
    #         exit(1)
    #         pass
        
    #     #print(index)
    #     #print(path_corners)
    #     #print(index, loop)
        
    #     if len(loop) > 0:
    #         same_mask = torch.abs(loop.unsqueeze(1).unsqueeze(0) - loop.unsqueeze(-1).unsqueeze(1)).min(-1)[0].max(-1)[0] == 0
    #         loop_range = torch.arange(len(loop)).long().cuda()
    #         same_mask = same_mask & (loop_range.unsqueeze(-1) > loop_range.unsqueeze(0))
    #         same_mask = same_mask.max(-1)[0]^1
    #         loops.append((loop_confidence[same_mask], loop[same_mask]))
    #     else:
    #         loops.append((loop_confidence, loop))
    #         pass
    #     continue



    #print(paths)
    #torch.save(paths, 'test/paths.pth')
    #torch.save(corner_confidence, 'test/confidence.pth')
    #torch.save((edge_confidence, edge_corner, num_corners, corners), 'test/debug.pth')
    #print(corner_confidence)
    # print(corner_confidence[8, 11], corner_confidence[3, 11])
    # print(paths[0][0][6, 8], paths[0][1][6, 8])        
    # print(paths[0][0][6, 3], paths[0][1][6, 3])    
    # print(paths[1][1][3, 4], paths[1][0][3, 4], corner_confidence[3, 4])
    # exit(1)
    #print(corner_confidence[4, 5])    
    return loops

def compute_loop_conflict_mask(loop_edge_masks, loop_masks, corners, edge_corner, edge_confidence, loop_corner_indices):
    num_loops = len(loop_edge_masks)
    loop_edge_flags = loop_edge_masks > 0
    
    if True:
        #same_edge_mask = ((loop_edge_masks.unsqueeze(1) == loop_edge_masks) & (loop_edge_masks > 0)).max(-1)[0]
        #same_corner_mask = ((loop_edge_masks.unsqueeze(1) == loop_edge_masks) & (loop_edge_masks > 0)).max(-1)[0]
        #min_area = loop_masks.sum(-1).sum(-1).min()
        #area_threshold = min(min_area // 2, 50)
        same_corner_mask = torch.stack([torch.stack([(torch.norm(corners[corner_indices_1.unsqueeze(-1)] - corners[corner_indices_2], dim=-1) < 0.02).max(-1)[0].max(-1)[0] for corner_indices_2 in loop_corner_indices]) for corner_indices_1 in loop_corner_indices])
        loop_areas = loop_masks.sum(-1).sum(-1).float()        
        overlap_mask = (loop_masks.unsqueeze(1) & loop_masks).sum(-1).sum(-1).float() / (loop_masks.unsqueeze(1) | loop_masks).sum(-1).sum(-1).float() > 0.7
        overlap_mask = (loop_masks.unsqueeze(1) & loop_masks).sum(-1).sum(-1).float() / torch.clamp(torch.min(loop_areas.unsqueeze(-1), loop_areas), min=1e-4) > 0.1
        #print(overlap_mask, same_corner_mask)
        overlap_mask = overlap_mask & same_corner_mask
        
        # overlap_mask = torch.zeros((num_loops, num_loops)).cuda().byte()    
        # overlap_indices = same_edge_mask.nonzero()
        # if len(overlap_indices) > 0:
        #     loop_areas = loop_masks.sum(-1).sum(-1).float()        
        #     overlap_loop_areas = loop_areas[overlap_indices]
        #     similar_area_mask = overlap_loop_areas.min(-1)[0] > overlap_loop_areas.max(-1)[0] * 0.7
        #     overlap_indices = overlap_indices[similar_area_mask]
        #     overlap_loop_areas = overlap_loop_areas[similar_area_mask]        
        #     if len(overlap_indices) > 0:
        #         overlap_loop_masks = loop_masks[overlap_indices]
        #         intersection = (overlap_loop_masks[:, 0] & overlap_loop_masks[:, 1]).sum(-1).sum(-1).float()
        #         union = (overlap_loop_masks[:, 0] | overlap_loop_masks[:, 1]).sum(-1).sum(-1).float()
        #         IOU_mask = intersection / union > 0.7
        #         #IOU_mask = IOU_mask | (intersection < 0.9 * overlap_loop_areas.min(-1)[0])
        #         overlap_mask = overlap_mask.index_put_((overlap_indices[:, 0], overlap_indices[:, 1]), IOU_mask)
        #         pass
        #     pass
        edge_flags = edge_confidence > 0.5
        num_good_edges = (loop_edge_flags * edge_flags).sum(-1)
        containing_mask = ((loop_edge_flags.unsqueeze(1) | loop_edge_flags) * edge_flags).sum(-1) == torch.max(num_good_edges.unsqueeze(-1), num_good_edges)
        overlap_mask = overlap_mask | containing_mask
        loop_subset_mask = torch.stack([torch.stack([(torch.norm(corners[corner_indices_1.unsqueeze(-1)] - corners[corner_indices_2], dim=-1) < 0.02).max(-1)[0].min(-1)[0] for corner_indices_2 in loop_corner_indices]) for corner_indices_1 in loop_corner_indices])        
        loop_subset_mask = torch.max(loop_subset_mask, loop_subset_mask.transpose(0, 1))
        overlap_mask = overlap_mask | loop_subset_mask
    else:
        loop_subset_mask = torch.stack([torch.stack([(torch.norm(corners[corner_indices_1.unsqueeze(-1)] - corners[corner_indices_2], dim=-1) < 0.02).max(-1)[0].min(-1)[0] for corner_indices_2 in loop_corner_indices]) for corner_indices_1 in loop_corner_indices])
        overlap_mask = torch.max(loop_subset_mask, loop_subset_mask.transpose(0, 1))        
        pass
    # edges = corners[edge_corner]
    # edge_directions = edges[:, 1] - edges[:, 0]
    # edge_directions = edge_directions / torch.norm(edge_directions, dim=-1, keepdim=True)
    # same_edge_mask = (edge_corner.unsqueeze(-2) == edge_corner[:, :1]).max(-1)[0] | (edge_corner.unsqueeze(-2) == edge_corner[:, 1:2]).max(-1)[0]
    # colinear_mask = (torch.abs((edge_directions.unsqueeze(-2) * edge_directions).sum(-1)) > dot_product_threshold) & same_edge_mask
    
    edges = corners[edge_corner]
    edge_intersection_mask = compute_edge_conflict_mask(edges)
    # print(intersection_mask)
    # print(loop_edge_flags[0])
    # print(loop_edge_flags[3])    
    # print(edge_corner[loop_edge_flags[0]], edge_corner[loop_edge_flags[3]])
    # exit(1)
    intersection_mask = (((loop_edge_flags.unsqueeze(-1) * edge_intersection_mask).max(1, keepdim=True)[0] == loop_edge_flags) & loop_edge_flags).max(-1)[0]
    conflict_mask = overlap_mask | intersection_mask

    #print(same_corner_mask, overlap_mask, intersection_mask)
    return conflict_mask

def findMultiLoopsModule(loop_confidence, loop_info, edge_corner, num_corners, max_num_loop_corners=10, confidence_threshold=0, corners=None, disable_colinear=True, edge_pred=None):

    loop_masks = []

    loop_edges = torch.stack([loop[1] for loop in loop_info])
    loop_masks = torch.stack([loop[2] for loop in loop_info])

    num_loops = len(loop_confidence)
    _, order = torch.sort(loop_confidence, descending=True)
    loop_confidence = loop_confidence[order]
    loop_edges = loop_edges[order]
    loop_masks = loop_masks[order]

    conflict_mask = compute_loop_conflict_mask(loop_edges, loop_masks, corners, edge_corner, edge_pred)

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
    #multi_loop_confidence = (multi_loop_edge_mask * (edge_pred - 0.5)).sum(-1)
    multi_loop_confidence = torch.stack([(loop_confidence[new_multi_loop[1]] - 0.5).sum(0) for new_multi_loop in new_multi_loops])
    _, multi_loop_order = torch.sort(multi_loop_confidence, descending=True)

    # print(edge_corner[loop_edges.long()].view((len(loop_edges), -1)))

    #print(overlap_mask, containing_mask, intersection_mask)    
    #print(conflict_mask)    
    if False:
        print(overlap_mask, containing_mask, intersection_mask)    
        print(conflict_mask)
        print([(new_multi_loop[0].item(), new_multi_loop[1]) for new_multi_loop in new_multi_loops])

        # print(corners)
        # print(loop_confidence)
        # print(edge_corner)
        # print([(index, loop[2]) for index, loop in enumerate(loop_info)])
        # print(same_edge_mask)        
        # print(overlap_mask)
        # loop_masks = loop_masks.detach().cpu().numpy()
        # for index, mask in enumerate(loop_masks):
        #     mask_image = (mask * 255).astype(np.uint8)
        #     mask_image = np.stack([mask_image, mask_image, mask_image], axis=-1)
        #     mask_image = cv2.resize(mask_image, (256, 256))
        #     corners = np.clip((loop_info[order[index]][2].detach().cpu().numpy() * mask_image.shape[1]).round().astype(np.int32), 0, mask_image.shape[1] - 1)
        #     for corner_index, corner in enumerate(corners):
        #         cv2.line(mask_image, (corner[1], corner[0]), (corners[(corner_index + 1) % len(corners)][1], corners[(corner_index + 1) % len(corners)][0]), thickness=1, color=(255, 0, 0))
        #         continue
        #     cv2.imwrite('test/mask_' + str(index) + '.png', mask_image)
        #     continue
        # print(conflict_mask)        
        # print(new_multi_loops)
        # print(multi_loop_confidence)
        # print(multi_loop_order, new_multi_loops[multi_loop_order[0]][1])
        # #print(multi_loop_edge_mask[order[:10]])        
        # #exit(1)
        pass
    #print('num', len(loop_confidence), len(multi_loop_edge_mask))
    return multi_loop_edge_mask[multi_loop_order[:10]]

def findBestMultiLoop(loop_confidence, edge_confidence, loop_edges, conflict_mask, edge_corner, corners):

    # max_num_loops = 20
    # if len(loop_confidence) > max_num_loops:
    #     order = torch.sort(loop_confidence, descending=True)[1]
    #     order = order[:max_num_loops]
    #     loop_confidence = loop_confidence[order]
    #     loop_edges = loop_edges[order]
    #     loop_masks = loop_masks[order]
    #     pass

    # max_num_edges = 50
    # if len(edge_confidence) > max_num_edges:
    #     order = torch.sort(edge_confidence, descending=True)[1]
    #     order = order[:max_num_edges]
    #     edge_confidence = edge_confidence[order]
    #     edge_corner = edge_corner[order]
    #     loop_edges = loop_edges[:, order]
    #     pass

    
    num_loops = len(loop_confidence)
    #conflict_mask = compute_loop_conflict_mask(loop_edges, loop_masks, corners, edge_corner, edge_confidence)
    loop_num_edges = loop_edges.sum(-1)

    #multi_loop_masks = [torch.LongTensor([loop_index]).cuda() for loop_index in range(num_loops)]
    loop_range = torch.arange(num_loops).cuda().long()
    multi_loop_masks = loop_range.unsqueeze(-1) == loop_range
    
    for _ in range(1, num_loops):
        prev_num_multi_loops = len(multi_loop_masks)
        multi_loop_conflict_mask = (multi_loop_masks.unsqueeze(-1) * conflict_mask).max(1)[0]
        new_multi_loop_masks = []
        for loop_index in range(num_loops):
            loop_mask = loop_range == loop_index
            valid_multi_loop_masks = multi_loop_masks[multi_loop_conflict_mask[:, loop_index] < 1]
            if len(valid_multi_loop_masks) > 0:
                new_multi_loop_masks.append(torch.max(valid_multi_loop_masks, loop_mask))
                pass
            continue
        if len(new_multi_loop_masks) == 0:
            break
        multi_loop_masks = torch.cat([multi_loop_masks] + new_multi_loop_masks, dim=0)
        #print(multi_loop_masks.shape, prev_num_multi_loops)
        
        if len(multi_loop_masks) < 5000:
            same_mask = (multi_loop_masks.unsqueeze(1) == multi_loop_masks).min(-1)[0]
            multi_loop_range = torch.arange(len(multi_loop_masks)).cuda().long()
            same_mask = same_mask & (multi_loop_range.unsqueeze(-1) > multi_loop_range.unsqueeze(0))
            same_mask = same_mask.max(-1)[0]
            multi_loop_masks = multi_loop_masks[same_mask^1]
        else:
            multi_loop_confidence = (multi_loop_masks.float() * ((loop_confidence - 0.5) * loop_num_edges)).sum(-1)
            #multi_loop_edge_masks = (multi_loop_masks.long().unsqueeze(-1) * loop_edges.long()).max(1)[0]            
            #multi_loop_edge_confidence = (multi_loop_edge_masks.float() * (edge_confidence - 0.5)).sum(-1)
            order = torch.sort(multi_loop_confidence, descending=True)[1]
            multi_loop_masks = multi_loop_masks[order[:5000]]
            pass
        
        if len(multi_loop_masks) == prev_num_multi_loops:
            break
        continue

    multi_loop_edge_masks = (multi_loop_masks.long().unsqueeze(-1) * loop_edges.long()).max(1)[0]

    #multi_loop_confidence = (multi_loop_masks.float() * ((loop_confidence - 0.5) * loop_num_edges)).sum(-1)
    multi_loop_confidence = (multi_loop_masks.float() * (loop_confidence - 0.5)).sum(-1)
    multi_loop_edge_confidence = (multi_loop_edge_masks.float() * (edge_confidence - 0.5)).sum(-1)    

    #best_multi_loop = multi_loop_edge_masks[(multi_loop_confidence + multi_loop_edge_confidence).max(0)[1]]
    best_multi_loop = multi_loop_edge_masks[(multi_loop_edge_confidence).max(0)[1]]

    debug_info = multi_loop_edge_masks, multi_loop_confidence, multi_loop_edge_confidence
    return best_multi_loop, debug_info


def findBestMultiLoopEdge(edge_confidence, edge_corner, all_corners):
    num_corners = len(all_corners)
    all_loops = findLoopsModule(edge_confidence, edge_corner, num_corners, max_num_loop_corners=num_corners, corners=all_corners, disable_colinear=True)

    #loop_pred = []
    loop_corner_indices = []
    loop_confidence = []
    max_confidence = max([confidence.max() for confidence, loops in all_loops])
    for confidence, loops in all_loops:
        for loop_index in range(len(loops)):
            if confidence[loop_index] < min(0.5, max_confidence):
                continue
            loop_corner_indices.append(loops[loop_index])                                
            loop_confidence.append(confidence[loop_index])
            continue
        continue
    loop_confidence = torch.stack(loop_confidence, dim=0)                        
    if len(loop_corner_indices) > 1:
        loop_subset_mask = torch.stack([torch.stack([(corner_indices_1.unsqueeze(-1) == corner_indices_2).max(-1)[0].min(0)[0] for corner_indices_2 in loop_corner_indices]) for corner_indices_1 in loop_corner_indices])
        loop_subset_mask = torch.max(loop_subset_mask, loop_subset_mask.transpose(0, 1))
        if False:
            order = torch.sort(loop_confidence, descending=True)[1]
            conflict_mask = loop_subset_mask[order[0]]
            valid_indices = [order[0]]
            for index in range(1, len(order)):
                index = order[index]
                if conflict_mask[index]:
                    continue
                conflict_mask = torch.max(conflict_mask, loop_subset_mask[index])
                valid_indices.append(index)
                continue
            loop_confidence = loop_confidence[torch.stack(valid_indices)]
            loop_corner_indices = [loop_corner_indices[index] for index in valid_indices]
        else:
            confidence_mask = loop_confidence.unsqueeze(-1) < loop_confidence
            valid_indices = ((loop_subset_mask & confidence_mask).max(-1)[0] == 0).nonzero()[:, 0]
            loop_confidence = loop_confidence[valid_indices]
            loop_corner_indices = [loop_corner_indices[index] for index in valid_indices]
            pass
        pass
    #print(loop_corner_indices)
    
    if len(loop_confidence) > 200:
        #order = np.argsort([confidence.item() for confidence in loop_confidence])[::-1][:200]
        _, order = torch.sort(loop_confidence, descending=True)[:200]
        order = order[torch.randperm(len(order))]
        #np.random.shuffle(order)
        loop_confidence = loop_confidence[order]
        loop_corner_indices = [loop_corner_indices[index] for index in order]

        # order = np.argsort(loop_confidence)[::-1][:200]
        # np.random.shuffle(order)
        # loop_confidence = loop_confidence[order]
        # loop_corner_indices = [loop_corner_indices[index] for index in order]
        pass


    mask_size = 64
    Us = torch.arange(mask_size).float().cuda().repeat((mask_size, 1))
    Vs = torch.arange(mask_size).float().cuda().unsqueeze(-1).repeat((1, mask_size))
    UVs = torch.stack([Vs, Us], dim=-1)
    
    loop_edge_masks = []
    loop_masks = []
    loop_edge_indices = []
    all_loop_edges = []
    for loop in loop_corner_indices:

        loop_corners = all_corners[loop]
        loop_edges = torch.cat([loop_corners, torch.cat([loop_corners[-1:], loop_corners[:-1]], dim=0)], dim=-1)
        clockwise = ((loop_edges[:, 0] - loop_edges[:, 2]) * (loop_edges[:, 1] + loop_edges[:, 3])).sum() > 0
        if not clockwise.item():
            reverse = torch.arange(start=len(loop) - 1, end=-1, step=-1).long()
            loop_edges = loop_edges[reverse]
            loop = loop[reverse]
            pass
        all_loop_edges.append(loop_edges)

        loop_corner_pairs = torch.cat([torch.stack([loop, torch.cat([loop[-1:], loop[:-1]], dim=0)], dim=-1), torch.stack([torch.cat([loop[-1:], loop[:-1]], dim=0), loop], dim=-1)], dim=0)
        loop_edge_mask = (edge_corner.unsqueeze(1) == loop_corner_pairs).min(-1)[0]
        loop_edge_masks.append(loop_edge_mask.max(-1)[0].float())

        loop_edge_mask = loop_edge_mask[:, :len(loop)] + loop_edge_mask[:, len(loop):]
        loop_edge_indices.append(loop_edge_mask.max(0)[1])

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
    loop_edge_masks = torch.stack(loop_edge_masks, dim=0)
    loop_masks = torch.stack(loop_masks, dim=0)
        
    all_edges = all_corners[edge_corner]    
    #edge_conflict_mask = compute_edge_conflict_mask(all_edges)
    conflict_mask = compute_loop_conflict_mask(loop_edge_masks, loop_masks, all_corners, edge_corner, edge_confidence, loop_corner_indices)

    
    num_loops = len(loop_confidence)
    #conflict_mask = compute_loop_conflict_mask(loop_edges, loop_masks, corners, edge_corner, edge_confidence)
    loop_num_edges = loop_edge_masks.sum(-1)

    #multi_loop_masks = [torch.LongTensor([loop_index]).cuda() for loop_index in range(num_loops)]
    loop_range = torch.arange(num_loops).cuda().long()
    multi_loop_masks = loop_range.unsqueeze(-1) == loop_range
    
    for _ in range(1, num_loops):
        prev_num_multi_loops = len(multi_loop_masks)
        multi_loop_conflict_mask = (multi_loop_masks.unsqueeze(-1) * conflict_mask).max(1)[0]
        new_multi_loop_masks = []
        for loop_index in range(num_loops):
            loop_mask = loop_range == loop_index
            valid_multi_loop_masks = multi_loop_masks[multi_loop_conflict_mask[:, loop_index] < 1]
            if len(valid_multi_loop_masks) > 0:
                new_multi_loop_masks.append(torch.max(valid_multi_loop_masks, loop_mask))
                pass
            continue
        if len(new_multi_loop_masks) == 0:
            break
        multi_loop_masks = torch.cat([multi_loop_masks] + new_multi_loop_masks, dim=0)
        #print(multi_loop_masks.shape, prev_num_multi_loops)
        
        if len(multi_loop_masks) < 5000:
            same_mask = (multi_loop_masks.unsqueeze(1) == multi_loop_masks).min(-1)[0]
            multi_loop_range = torch.arange(len(multi_loop_masks)).cuda().long()
            same_mask = same_mask & (multi_loop_range.unsqueeze(-1) > multi_loop_range.unsqueeze(0))
            same_mask = same_mask.max(-1)[0]
            multi_loop_masks = multi_loop_masks[same_mask^1]
        else:
            multi_loop_confidence = (multi_loop_masks.float() * ((loop_confidence - 0.5) * loop_num_edges)).sum(-1)
            #multi_loop_edge_masks = (multi_loop_masks.long().unsqueeze(-1) * loop_edge_masks.long()).max(1)[0]            
            #multi_loop_edge_confidence = (multi_loop_edge_masks.float() * (edge_confidence - 0.5)).sum(-1)
            order = torch.sort(multi_loop_confidence, descending=True)[1]
            multi_loop_masks = multi_loop_masks[order[:5000]]
            pass
        
        if len(multi_loop_masks) == prev_num_multi_loops:
            break
        continue

    multi_loop_edge_masks = (multi_loop_masks.long().unsqueeze(-1) * loop_edge_masks.long()).max(1)[0]

    #multi_loop_confidence = (multi_loop_masks.float() * ((loop_confidence - 0.5) * loop_num_edges)).sum(-1)
    multi_loop_confidence = (multi_loop_masks.float() * (loop_confidence - 0.5)).sum(-1)
    if False:
        multi_loop_edge_confidence = (multi_loop_edge_masks.float() * (edge_confidence - 0.5)).sum(-1)
    else:
        edge_normals = all_edges[:, 1] - all_edges[:, 0]
        edge_normals = edge_normals / torch.clamp(torch.norm(edge_normals, dim=-1, keepdim=True), 1e-4)
        dot_product = (edge_normals.unsqueeze(-2) * edge_normals).sum(-1)
        dot_product_mask = torch.abs(dot_product) > dot_product_threshold
        same_corner_mask = (edge_corner.unsqueeze(0).unsqueeze(-2) == edge_corner.unsqueeze(1).unsqueeze(-1)).max(-1)[0].max(-1)[0]
        active_mask = torch.min(edge_confidence.unsqueeze(-1), edge_confidence) > 0.5
        colinear_mask = dot_product_mask & same_corner_mask & active_mask
        colinear_mask = torch.max(colinear_mask, torch.eye(len(colinear_mask)).cuda().byte())
        while True:
            new_colinear_mask = (colinear_mask.unsqueeze(-1) * colinear_mask).max(1)[0]
            if (new_colinear_mask == colinear_mask).min():
                break
            colinear_mask = new_colinear_mask
            continue
        edge_group_mapping = torch.full((len(colinear_mask), ), fill_value=-1).cuda().long()
        new_index = 0
        for index in range(len(colinear_mask)):
            if edge_group_mapping[index] >= 0:
                continue
            edge_group_mapping[colinear_mask[index]] = new_index
            new_index += 1
            continue
        
        # edge_values = (colinear_mask.long() * torch.arange(len(colinear_mask)).cuda()).sum(-1)
        # print(edge_values)
        # _, edge_group_mapping = torch.unique(edge_values, return_inverse=True)
        edge_group_masks = edge_group_mapping.unsqueeze(-1) == torch.arange(new_index).cuda()
        edge_group_masks = edge_group_masks.float()
        group_confidence = (edge_group_masks * edge_confidence.unsqueeze(-1)).sum(0) / torch.clamp(edge_group_masks.sum(0), 1e-4)
        multi_loop_group_masks = (multi_loop_edge_masks.float().unsqueeze(-1) * edge_group_masks).max(1)[0]
        multi_loop_edge_confidence = (multi_loop_group_masks.float() * (group_confidence - 0.5)).sum(-1)        
        pass
    
    #best_multi_loop = multi_loop_edge_masks[(multi_loop_confidence + multi_loop_edge_confidence).max(0)[1]]
    best_multi_loop = multi_loop_edge_masks[(multi_loop_edge_confidence).max(0)[1]]

    top_indices = torch.sort(multi_loop_edge_confidence, descending=True)[1][:10]
    debug_info = [loop_edge_masks, multi_loop_edge_masks[top_indices], multi_loop_masks[top_indices], multi_loop_group_masks[top_indices], multi_loop_confidence[top_indices], multi_loop_edge_confidence[top_indices]]
    return best_multi_loop, debug_info


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

class LinearBlock(torch.nn.Module):
    """The block consists of a linear layer and a ReLU layer"""    
    def __init__(self, in_planes, out_planes, use_bn=False):
        super(LinearBlock, self).__init__()
        self.use_bn = use_bn
        self.linear = torch.nn.Linear(in_planes, out_planes)
        self.relu = torch.nn.ReLU(inplace=True)
        if self.use_bn:
            self.bn = torch.nn.BatchNorm(out_planes)
            pass
        return

    def forward(self, inp):
        if self.use_bn:
            return self.relu(self.bn(self.linear(inp)))
        else:
            return self.relu(self.linear(inp))
