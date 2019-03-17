dot_product_threshold = np.cos(np.deg2rad(20))

def evaluate_result(corner_pred, edge_pred, edge_corner, corner_gt, edge_gt):
    edge_corner = edge_corner[edge_pred > 0.5]
    
    edges = corner_pred[edge_corner]
    edge_directions = edges[:, 1] - edges[:, 0]
    edge_lengths = np.maximum(np.linalg.norm(edge_directions, axis=-1), 1e-4)
    edge_directions = edge_directions / np.expand_dims(edge_lengths, -1)
    edge_normals = np.stack([edge_directions[:, 1], -edge_directions[:, 0]], axis=-1)
    
    # edge_centers = edges[:, 0] + edges[:, 1]) / 2
    
    # normal_distance = np.abs(((np.expand_dims(edge_centers, 1) - edges[:, 0]) * edge_normals).sum(-1))
    # normal_distance = np.minimum(normal_distance, normal_distance.transpose())
    # colinear_mask = normal_distance > dot_product_threshold
    
    # tangent_distance_1 = ((expand_dims(edges[:, 0], 1) - edges[:, 0]) * edge_directions).sum(-1)
    # tangent_distance_2 = ((expand_dims(edges[:, 1], 1) - edges[:, 0]) * edge_directions).sum(-1)    
    # non_overlap_mask = (tangent_distance_1 < -1e-4 & tangent_distance_2 < -1e-4) | (tangent_distance_1 > 1 - 1e-4 & tangent_distance_2 > 1 - 1e-4)

    # num_edges = len(edge_pred)
    # edge_mapping = np.arange(num_edges, dtype=np.int32)
    # for edge_index_1 in range(num_edges):
    #     for edge_index_2 in range(num_edges):        
    #         if edge_index_2 <= edge_index_1:
    #             continue

    connected_mask = (np.expand_dims(np.expand_dims(edge_corner, -1), 1) == np.expand_dims(np.expand_dims(edge_corner, -2), 0)).any(-1).any(-1)
    
    normal_distance = np.abs(((np.expand_dims(corners, 1) - edges[:, 0]) * edge_normals).sum(-1))
    tangent_distance = ((expand_dims(corners, 1) - edges[:, 0]) * edge_directions).sum(-1)
    independent_mask = np.ones((len(corners), len(edge_pred)))
    independent_mask[edge_corner[:, 0], np.arange(len(edge_corner), dtype=np.int32)] = 0
    independent_mask[edge_corner[:, 1], np.arange(len(edge_corner), dtype=np.int32)] = 0
    
    colinear_mask = (tangent_distance > 0 & tangent_distance < 1) & normal_distance < 0.02 & independent_mask
    for edge_index, corner_indices in enumerate(edge_pred):
        for corner_index in corner_indices:
            if 
    for corner in corner_pred
            
