import torch

def findLoopsModule(edge_confidence, edge_corner, num_corners, max_num_loop_corners=10):
    corner_confidence = torch.zeros(num_corners, num_corners).cuda()
    corner_confidence[edge_corner[:, 0], edge_corner[:, 1]] = edge_confidence
    corner_confidence[edge_corner[:, 1], edge_corner[:, 0]] = edge_confidence
    corner_confidence = corner_confidence - torch.diag(torch.ones(num_corners).cuda()) * max_num_loop_corners
    corner_range = torch.arange(num_corners).cuda().long()
    corner_range_map = corner_range.view((-1, 1)).repeat((1, num_corners))
    paths = [(corner_confidence, corner_range_map.unsqueeze(-1))]
    while len(paths) < max_num_loop_corners - 1:
        path_confidence, path_corners = paths[-1]
        total_confidence = path_confidence.unsqueeze(-1) + corner_confidence
        prev_corner = path_corners[:, :, -1]
        visited_mask = (path_corners.unsqueeze(-1) == corner_range).max(-2)[0].float()
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
        mask = loop_confidence > 0
        loops.append((loop_confidence[mask], loop[mask]))
        continue
    return loops

    #paths = [(path[0], torch.cat([path[1], corner_range_map.transpose(0, 1).unsqueeze(-1)], dim=-1)) for path in paths]
    loops = []
    print(paths)
    for path in paths:
        confidence = path[0].diagonal()
        path = path[1].diagonal().transpose(0, 1)
        mask = confidence > 0
        loops.append((confidence[mask], path[mask]))
        continue
    print(loops)
    exit(1)
    return loops
