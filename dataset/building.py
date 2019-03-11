import numpy as np
import os
from skimage.draw import line_aa
from PIL import Image, ImageDraw
import torch
import cv2
import itertools
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def draw_edges(edges_on, edges):
    im = np.zeros((256, 256))
    for edge in edges[edges_on > 0.5]:
        cv2.line(im, (edge[1], edge[0]), (edge[3], edge[2]), thickness=3, color=1)
        continue
    return im

def draw_edge(edge_index, edges):
    im = np.zeros((256, 256))
    edge = edges[edge_index]
    cv2.line(im, (edge[1], edge[0]), (edge[3], edge[2]), thickness=3, color=1)
    return im


def findLoopsModuleCPU2(edge_confidence, edge_corner, num_corners, max_num_loop_corners=10, confidence_threshold=0, corners=None, disable_colinear=True):
    ## The confidence of connecting two corners
    corner_confidence = torch.zeros(num_corners, num_corners)
    corner_confidence[edge_corner[:, 0], edge_corner[:, 1]] = edge_confidence
    corner_confidence[edge_corner[:, 1], edge_corner[:, 0]] = edge_confidence
    corner_confidence = corner_confidence - torch.diag(torch.ones(num_corners)) * max_num_loop_corners
    
    corner_range = torch.arange(num_corners).long()
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
            loop_range = torch.arange(len(loop)).long()
            same_mask = same_mask & (loop_range.unsqueeze(-1) > loop_range.unsqueeze(0))
            same_mask = same_mask.max(-1)[0]^1
            loops.append((loop_confidence[same_mask], loop[same_mask]))
        else:
            loops.append((loop_confidence, loop))
            pass
        continue
    return loops

def findLoopsModuleCPU(edge_confidence, edge_corner, num_corners, max_num_loop_corners=10, confidence_threshold=0):
    ## The confidence of connecting two corners
    corner_confidence = torch.zeros(num_corners, num_corners)
    corner_confidence[edge_corner[:, 0], edge_corner[:, 1]] = edge_confidence
    corner_confidence[edge_corner[:, 1], edge_corner[:, 0]] = edge_confidence
    corner_confidence = corner_confidence - torch.diag(torch.ones(num_corners)) * max_num_loop_corners
    
    corner_range = torch.arange(num_corners).long()
    corner_range_map = corner_range.view((-1, 1)).repeat((1, num_corners))

    ## Paths = [(path_confidence, path_corners)] where the ith path has a length of (i + 1), path_confidence is the summation of confidence along the path between two corners, and path_corners is visited corners along the path
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

    ## Find closed loops by adding the starting point to the path
    paths = paths[1:]
    loops = []
    max_confidence_loop = (0, None)
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
            if loop_confidence.item() > max_confidence_loop[0]:
                max_confidence_loop = (loop_confidence.item(), loop[index:index + 1])
                pass
        else:
            loop_confidence = loop_confidence[mask]
            loop = loop[mask]
            if len(loop) > 0:
                same_mask = torch.abs(loop.unsqueeze(1).unsqueeze(0) - loop.unsqueeze(-1).unsqueeze(1)).min(-1)[0].max(-1)[0] == 0
                loop_range = torch.arange(len(loop)).long()
                same_mask = same_mask & (loop_range.unsqueeze(-1) > loop_range.unsqueeze(0))
                same_mask = same_mask.max(-1)[0]^1                
                loops.append(loop[same_mask])
                pass
            pass
        continue
    if len(loops) == 0:
        loops.append(max_confidence_loop[1])
        pass

    return loops

class Building():
    """Maintain a building to create data examples in the same format"""

    def __init__(self, options, _id, with_augmentation=True, corner_type='annots_only'):
        self.options = options
        self.with_augmentation = with_augmentation

        PREFIX = options.data_path
        LADATA_FOLDER = '{}/dataset_atlanta/processed/'.format(PREFIX)
        ANNOTS_FOLDER = '{}/dataset_atlanta/processed/annots'.format(PREFIX)
        EDGES_FOLDER = '{}/dataset_atlanta/processed/{}/edges'.format(PREFIX, corner_type)
        CORNERS_FOLDER = '{}/dataset_atlanta/processed/{}/corners'.format(PREFIX, corner_type)
        
        self.annots_folder = ANNOTS_FOLDER
        self.edges_folder = EDGES_FOLDER
        self.corners_folder = CORNERS_FOLDER
        self.dataset_folder = LADATA_FOLDER
        self._id = _id

        # annots
        annot_path = os.path.join(self.annots_folder, _id +'.npy')
        annot = np.load(open(annot_path, 'rb'), encoding='bytes')
        graph = dict(annot[()])

        # # augment data
        # if with_augmentation:
        #     rot = np.random.choice([0, 90, 180, 270])
        #     flip = np.random.choice([True, False])
        #     use_gt = np.random.choice([False])
        # else:
        #     rot = 0
        #     flip = False
        #     use_gt = False
        
        rot = 0
        flip = False
        use_gt = False
        
        #print(rot, flip)
        # load annots
        corners_annot, edges_annot = self.load_annot(graph, rot, flip)
        edge_corner_annots = self.compute_edge_corner_from_annots(corners_annot, edges_annot)

        # retrieve corners and edges detections
        corners_path = os.path.join(self.corners_folder, "{}_{}_{}.npy".format(_id, rot, flip))
        edges_path = os.path.join(self.edges_folder, "{}_{}_{}.npy".format(_id, rot, flip))
        corners_embs_path = os.path.join(self.corners_folder.replace('corners', 'corners_feats'), "{}_{}_{}.npy".format(_id, rot, flip))
        edges_embs_path = os.path.join(self.edges_folder.replace('edges', 'edges_feats'), "{}_{}_{}.npy".format(_id, rot, flip))

        # load primitives and features
        #print(corners_path)
        corners_det = np.load(open(corners_path, 'rb'))
        corners_embs = np.load(open(corners_embs_path, 'rb'))
        edges_embs = np.load(open(edges_embs_path, 'rb'))

        # extract edges from corners
        edges_det_from_corners, ce_assignment, self.graph_edge_index, self.graph_edge_attr, self.left_edges, self.right_edges = self.extract_edges_from_corners(corners_det, None)
        e_xys = self.compute_edges_map(edges_det_from_corners)
        # print(edges_det_from_corners)
        # print(e_xys)
        # print(corners_det)
        # match compute true/false samples for detections
        corners_gt, edges_gt = self.compute_gt(corners_det, corners_annot, edges_det_from_corners, edges_annot)
        #exit(1)

        #inds = np.load(edges_embs_path.replace('edges_feats', 'filters'))[1, :]
        edges_gt = edges_gt.astype(np.int32)
        ce_t0 = np.zeros_like(edges_gt)
        #ce_t0[inds] = 1.0
        corner_edge, corner_edge_pairs, edge_corner = self.compute_dists(corners_det, edges_det_from_corners, ce_assignment)

        #ce_angles_bins, ca_gt = self.compute_corner_angle_bins(corner_edge, edges_det_from_corners, edges_gt, corners_det)
        ce_angles_bins = None

        # read images
        self.rgb = self.read_rgb(_id, self.dataset_folder)
        self.imgs = self.read_input_images(_id, self.dataset_folder)
        self.edge_corner_annots = edge_corner_annots
        self.corners_gt = corners_gt                
        self.corners_det = np.round(corners_det[:, :2]).astype(np.int32)
        self.edges_gt = edges_gt
        self.edges_det = np.round(edges_det_from_corners).astype(np.int32)
        self.e_xys = e_xys
        self.ce_angles_bins = ce_angles_bins
        self.corner_edge = corner_edge
        self.corner_edge_pairs = corner_edge_pairs
        self.edge_corner = edge_corner
        self.generate_bins = False
        self.num_edges = len(self.edges_gt)
        self.num_edges_gt = self.edges_gt.sum()
        self.corners_annot = corners_annot
        self.edges_annot = edges_annot

        # include colinear edges to gt
        self.edges_gt_no_colinear = np.array(edges_gt)
        self.edges_gt = self.include_colinear_edges(edges_gt)

        if options.suffix != '':
            suffix = '_' + corner_type + '_' + options.suffix
        else:
            suffix = '_' + corner_type
            pass
        self.prediction_path = './cache/predicted_edges/{}/{}.npy'.format(corner_type, self._id)
        if os.path.exists(self.prediction_path):
            self.predicted_edges = np.load(self.prediction_path)
        else:
            self.predicted_edges = np.zeros((1, self.num_edges), dtype=np.int32)
            pass
        
        #if _id == '1525563157.13' and True:
        if _id == '1525562852.02' and False:
            print('test')
            print(self.predicted_edges.shape)
            print(self.predicted_edges[-1], self.edges_gt)
            mask = draw_edges(self.predicted_edges[-1], self.edges_det)
            image = (self.imgs[:, :, :3]).astype(np.uint8)
            mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
            image[mask > 128] = np.array([0, 0, 255])
            cv2.imwrite('test/image.png', image)
            #cv2.imwrite('test/mask.png', mask)
            exit(1)
            pass
        return

    def include_colinear_edges(self, edges_gt):

        colinear_edges = self.colinear_edges()
        for (e1, e2) in colinear_edges:
            if (edges_gt[e1] == 1) or (edges_gt[e2] == 1):
                edges_gt[e1] = 1
                edges_gt[e2] = 1
        return edges_gt

    def compute_edge_corner_from_annots(self, corners_annot, edges_annot):
        corners_annot = np.array(corners_annot).astype('float32')
        edges_annot = np.array(edges_annot).astype('float32')
        edge_corner_annots = []

        for e in edges_annot:
            c1, c2 = e[:2], e[2:]
            e_inds = []
            for k, c3 in enumerate(corners_annot):
                c3 = c3[:2]
                if np.array_equal(c3, c1) or np.array_equal(c3, c2):
                    e_inds.append(k)
            edge_corner_annots.append(e_inds)

        return np.array(edge_corner_annots)

    def create_samples(self, num_edges_source=-1, mode='inference'):
        """Create all data examples:
        num_edges_source: source graph size, -1 for using the latest
        """
        assert(num_edges_source < len(self.predicted_edges))
        samples = []
        labels = []
        for edge_index in range(self.num_edges):
            sample, label, _ = self.create_sample(edge_index, num_edges_source, mode=mode)
            samples.append(sample)
            labels.append(label)
            continue
        samples = torch.stack(samples)        
        labels = torch.stack(labels).view(-1)
        return samples, labels

    def create_sample(self, edge_index=-1, num_edges_source=-1, mode='training'):
        """Create one data example:
        edge_index: edge index to flip, -1 for random sampling
        num_edges_source: source graph size, -1 for using the latest
        """
        #assert(num_edges_source < len(self.predicted_edges))

        if self.with_augmentation:
            imgs, corners_det, edges_det = self.augment(self.imgs.copy(), self.corners_det, self.edges_det)
        else:
            imgs, corners_det, edges_det = self.imgs, self.corners_det, self.edges_det
            pass
        imgs = imgs.transpose((2, 0, 1)).astype(np.float32) / 255
        
        img_c, corner_masks = self.compute_corner_image(corners_det)
        imgs = np.concatenate([imgs, np.array(img_c)[np.newaxis, :, :]], axis=0)

        if edge_index < 0:
            edge_index = np.random.randint(self.num_edges)
            pass        
        if ('uniform' in self.options.suffix or 'single' in self.options.suffix) and mode == 'training':
            indices = np.nonzero(self.edges_gt)[0]
            indices = np.random.choice(indices, np.random.randint(len(indices) + 1), replace=False)
            current_edges = np.zeros(self.edges_gt.shape, dtype=np.int32)
            current_edges[indices] = 1
            if 'single' in self.options.suffix:
                current_edges[edge_index] = 0
                pass            
        else:
            current_edges = self.predicted_edges[num_edges_source].copy()
            pass
        # draw current state
        im_s0 = draw_edges(current_edges, edges_det)
        # draw new state
        new_state = np.array(current_edges)
        if 'single' in self.options.suffix:
            new_state.fill(0)
            pass
        new_state[edge_index] = 1 - new_state[edge_index]
        im_s1 = draw_edges(new_state, edges_det)
        sample = np.concatenate([imgs, im_s0[np.newaxis, :, :], im_s1[np.newaxis, :, :]])
        label = 1 if new_state[edge_index] == self.edges_gt[edge_index] else 0
        #sample_rev = np.concatenate([imgs, im_s1[np.newaxis, :, :], im_s0[np.newaxis, :, :]])    
        #label_rev = 0.0 if new_state[k] == edges_gt[k] else 1.0

        # generate bins
        
        # if self.generate_bins:
        #     curr_bins = generate_bins(current_edges, ce_angles_bins)
        #     new_bins = generate_bins(new_state, ce_angles_bins)
        #     c1, c2 = None, None
        #     try:
        #         c1, c2 = np.where(corner_edge[:, edge_index] == 1)[0]
        #     except:
        #         print('Edge has more than two corners', np.where(corner_edge[:, edge_index] == 1)[0])

        #     curr_bin_state = np.concatenate([curr_bins[c1], curr_bins[c2]])
        #     new_bin_state = np.concatenate([new_bins[c1], new_bins[c2]])

        #     if np.random.choice([True, False]):
        #         curr_bin_state = np.zeros_like(curr_bin_state)
        #         new_bin_state = np.zeros_like(new_bin_state)

        #     bin_sample = np.concatenate([curr_bin_state, new_bin_state])
        #     sample = np.concatenate([sample, bin_sample], axis=0)
        #     pass
        return torch.from_numpy(sample.astype(np.float32)), torch.Tensor([label]).long(), torch.Tensor([edges_det[edge_index].astype(np.float32)])

    def create_sample_graph(self, load_heatmaps=False):
        """Create one data example:
        edge_index: edge index to flip, -1 for random sampling
        num_edges_source: source graph size, -1 for using the latest
        """
        #assert(num_edges_source < len(self.predicted_edges))

        if self.with_augmentation:
            imgs, corners_det, edges_det = self.augment(self.imgs.copy(), self.corners_det, self.edges_det)
        else:
            imgs, corners_det, edges_det = self.imgs, self.corners_det, self.edges_det
            pass

        imgs = imgs.transpose((2, 0, 1)).astype(np.float32) / 255
        if load_heatmaps:
            img_c, corner_masks = self.compute_corner_image(corners_det)
            imgs = np.concatenate([imgs, np.array(img_c)[np.newaxis, :, :]], axis=0)

            edge_images = []
            for edge_index in range(len(edges_det)):
                edge_images.append(draw_edge(edge_index, edges_det))
                continue
            edge_images = np.stack(edge_images, axis=0)
            #return [imgs.astype(np.float32), edge_images.astype(np.float32), corners_det.astype(np.float32) / 256, edges_det.astype(np.float32) / 256, self.corners_gt.astype(np.float32), self.edges_gt.astype(np.float32), self.graph_edge_index, self.graph_edge_attr, self.left_edges.astype(np.int64), self.right_edges.astype(np.int64)]
            return [imgs.astype(np.float32), corner_masks.astype(np.float32), edge_images.astype(np.float32), corners_det.astype(np.float32) / 256, edges_det.astype(np.float32) / 256, self.corners_gt.astype(np.float32), self.edges_gt.astype(np.float32), self.corner_edge_pairs, self.edge_corner, self.left_edges.astype(np.int64), self.right_edges.astype(np.int64)]
        else:
            return [imgs.astype(np.float32), corners_det.astype(np.float32) / 256, edges_det.astype(np.float32) / 256, self.corners_gt.astype(np.float32), self.edges_gt.astype(np.float32), self.graph_edge_index, self.graph_edge_attr]

    def create_sample_edge(self, edge_index, load_heatmaps=False):
        """Create one data example:
        edge_index: edge index to flip, -1 for random sampling
        num_edges_source: source graph size, -1 for using the latest
        """
        #assert(num_edges_source < len(self.predicted_edges))

        if self.with_augmentation:
            imgs, corners_det, edges_det = self.augment(self.imgs.copy(), self.corners_det, self.edges_det)
        else:
            imgs, corners_det, edges_det = self.imgs, self.corners_det, self.edges_det
            pass

        imgs = imgs.transpose((2, 0, 1)).astype(np.float32) / 255
        img_c, corner_masks = self.compute_corner_image(corners_det)
        imgs = np.concatenate([imgs, np.array(img_c)[np.newaxis, :, :]], axis=0)

        edge_image = draw_edge(edge_index, edges_det)
        return [imgs.astype(np.float32), edge_image.astype(np.float32), self.edges_gt[edge_index]]
    
    def update_edge(self, edge_index):
        """Add new prediction"""
        current_edges = self.predicted_edges[-1].copy()
        current_edges[edge_index] = 1 - current_edges[edge_index]
        num_edges = current_edges.sum()
        
        if num_edges >= len(self.predicted_edges): ## Add new state
            self.predicted_edges = np.concatenate([self.predicted_edges, np.expand_dims(current_edges, 0)], axis=0)
        else: ## Removed an edge
            self.predicted_edges[num_edges] = current_edges
            self.predicted_edges = self.predicted_edges[:num_edges + 1]
            pass
        return
    
    def update_edges(self, edges):
        """Reset predicted edges"""
        self.predicted_edges = np.concatenate([self.predicted_edges, np.expand_dims(edges, 0)], axis=0)
        return    

    def set_edges(self, edges):
        self.predicted_edges = edges
        return 

    def current_num_edges(self):
        return len(self.predicted_edges) - 1

    def current_edges(self):
        return self.predicted_edges[-1]
    
    def reset(self, num_edges=0):
        """Reset predicted edges"""
        self.predicted_edges = self.predicted_edges[:num_edges + 1]
        return

    def save(self):
        """Save predicted edges"""
        np.save(self.prediction_path, self.predicted_edges)
        return

    def check_result(self):
        """Save predicted edges"""
        return np.all(self.predicted_edges[-1] == self.edges_gt)

    def compute_corner_image(self, corners):
        im_c = np.zeros((len(corners), 256, 256))
        for corner_index, c in enumerate(corners):
            cv2.circle(im_c[corner_index], (c[1], c[0]), color=1, radius=5, thickness=-1)
            # x, y, _, _ = np.array(c)
            # x, y = int(x), int(y)
            # im_c[x, y] = 1
        return im_c.max(0), im_c

    def read_rgb(self, _id, path):
        im = np.array(Image.open("{}/rgb/{}.jpg".format(path, _id)))#.resize((128, 128))
        return im

    def read_input_images(self, _id, path):

        #im = np.array(Image.open("{}/rgb/{}.jpg".format(path, _id)))#.resize((128, 128))
        im = np.array(Image.open("{}/outlines/{}.jpg".format(path, _id)).convert('L'))#.resize((128, 128))
        #im = cv2.imread("{}/rgb/{}.jpg".format(path, _id))#.resize((128, 128))
        # dp_im = Image.open(info['path'].replace('rgb', 'depth')).convert('L')
        # surf_im = Image.open(info['path'].replace('rgb', 'surf'))
        # gray_im = Image.open(info['path'].replace('rgb', 'gray')).convert('L')
        #out_im = Image.open("{}/outlines/{}.jpg".format(path, _id)).convert('L')#.resize((128, 128))

        #im = im.rotate(rot)
        # dp_im = dp_im.rotate(rot)
        # surf_im = surf_im.rotate(rot)
        # gray_im = gray_im.rotate(rot)
        #out_im = out_im.rotate(rot)
        # if flip == True:
        #     im = im.transpose(Image.FLIP_LEFT_RIGHT)
        #     out_im = out_im.transpose(Image.FLIP_LEFT_RIGHT)

        # print(rot, flip)
        # dp_im = dp_im.transpose(Image.FLIP_LEFT_RIGHT)
        # surf_im = surf_im.transpose(Image.FLIP_LEFT_RIGHT)
        # gray_im = gray_im.transpose(Image.FLIP_LEFT_RIGHT)
        #out_im = out_im.transpose(Image.FLIP_LEFT_RIGHT)

        ## Add normalization here for consistency
        #imgs = im.transpose((2, 0, 1)) / 255.0 # - 0.5
        #return np.array(out_im)[:, :, np.newaxis]
        #return np.concatenate([np.array(im), np.array(dp_im)[:, :, np.newaxis], np.array(gray_im)[:, :, np.newaxis], np.array(surf_im)], axis=-1)
        #return np.array(im)
        return im

    def rotate_flip(self, edges_coords, rot, flip):
        new_edges_coords = []
        for e in edges_coords:
            y1, x1, y2, x2 = e
            x1, y1  = self.rotate_coords(np.array([256, 256]), np.array([x1, y1]), rot)
            x2, y2  = self.rotate_coords(np.array([256, 256]), np.array([x2, y2]), rot)
            if flip:
                x1, y1 = (128-abs(128-x1), y1) if x1 > 128 else (128+abs(128-x1), y1)
                x2, y2 = (128-abs(128-x2), y2) if x2 > 128 else (128+abs(128-x2), y2)
            e_aug = (y1, x1, y2, x2)
            new_edges_coords.append(e_aug)
        return new_edges_coords

    def compute_angles(self, edges_coords, delta_degree=5.0, n_bins=36):

        edges_coords = np.array(edges_coords)
        inds = []
        angles = []
        for i, e1 in enumerate(edges_coords):
            y2, x2, y1, x1 = e1

            # compute angle
            pc = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
            pp = np.array([0, 1])
            pr = np.array([x1, y1]) if x1 >= x2 else np.array([x2, y2])
            pr -= pc
            cosine_angle = np.dot(pp, pr) / (np.linalg.norm(pp) * np.linalg.norm(pr) + 1e-8)
            angle = np.arccos(cosine_angle)
            angle = 180.0 - np.degrees(angle)

            bin_num = (int(angle/delta_degree)%n_bins)
            inds.append(bin_num)
            angles.append(angle)

        one_hot = np.zeros((edges_coords.shape[0], n_bins))
        one_hot[np.arange(edges_coords.shape[0]), np.array(inds)] = 1.0
        angles = np.array(angles)

        return one_hot, angles

    def rotate_coords(self, image_shape, xy, angle):
        org_center = (image_shape-1)/2.
        rot_center = (image_shape-1)/2.
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a), -org[0]*np.sin(a) + org[1]*np.cos(a)])
        return new+rot_center

    def compute_edges_map(self, edges_det, grid_size=128, scale=2.0, rot=0, flip=False):

        xys_unpadded = []
        mlen = 0
        for e in edges_det:
            y1, x1, y2, x2 = (e/scale).astype(np.int32)
            edges_map = Image.new("RGB", (grid_size, grid_size), (0, 0, 0)) #Image.fromarray(np.zeros((grid_size, grid_size)))
            draw = ImageDraw.Draw(edges_map)
            draw.line((x1, y1, x2, y2), width=2, fill='white')

            # apply augmentation
            edges_map = edges_map.rotate(rot)
            if flip == True:
                edges_map = edges_map.transpose(Image.FLIP_LEFT_RIGHT)

            edges_map = np.array(edges_map)
            inds = np.array(np.where(edges_map > 0))[:2, :]
            if inds.shape[-1] > mlen:
                mlen = inds.shape[-1]
            xys_unpadded.append(inds)

        xys = []
        for inds in xys_unpadded:
            padded_inds = np.pad(inds, ((0, 0), (0, mlen-inds.shape[-1])), 'constant', constant_values=-1)
            xys.append(padded_inds)
        xys = np.stack(xys, 0)

        return xys

    def extract_edges_from_corners(self, corners_det, inds):
        c_e_assignment = []

        graph_edge_index = []
        if 'corner' not in self.options.suffix:
            edge_index_offset = 0
        else:
            edge_index_offset = len(corners_det)
            pass        
        corner_edge_map = {}
        if inds is None:
            for i1, c1 in enumerate(corners_det):
                for i2, c2 in enumerate(corners_det):
                    if i2 <= i1:
                        continue
                    graph_edge_index.append([i1, edge_index_offset + len(c_e_assignment)])
                    graph_edge_index.append([i2, edge_index_offset + len(c_e_assignment)])
                    if i1 not in corner_edge_map:
                        corner_edge_map[i1] = []
                        pass
                    corner_edge_map[i1].append(len(c_e_assignment))
                    if i2 not in corner_edge_map:
                        corner_edge_map[i2] = []
                        pass
                    corner_edge_map[i2].append(len(c_e_assignment))
                    c_e_assignment.append((c1, c2))
        else:
            k = 0
            for i, c1 in enumerate(corners_det):
                for i2, c2 in enumerate(corners_det):
                    if i2 <= i1:
                        continue
                    if (k in list(inds)):                    
                        graph_edge_index.append([i1, edge_index_offset + len(c_e_assignment)])
                        graph_edge_index.append([i2, edge_index_offset + len(c_e_assignment)])
                        if i1 not in corner_edge_map:
                            corner_edge_map[i1] = []
                            pass
                        corner_edge_map[i1].append(len(c_e_assignment))
                        if i2 not in corner_edge_map:
                            corner_edge_map[i2] = []
                            pass
                        corner_edge_map[i2].append(len(c_e_assignment))                        
                        c_e_assignment.append((c1, c2))
                        pass
                    k+=1

        e_from_corners = []
        for c1, c2 in c_e_assignment:
            y1, x1 = c1[:2]
            y2, x2 = c2[:2]
            e_from_corners.append([y1, x1, y2, x2])
        e_from_corners = np.array(e_from_corners)

        graph_edge_attr = np.ones((len(graph_edge_index), 1), dtype=np.float32)

        if 'corner' not in self.options.suffix:
            graph_edge_attr = np.zeros((0, 1), dtype=np.float32)
            graph_edge_index = []
            pass
        
        edge_neighbors = [[] for edge_index in range(len(c_e_assignment))]
        if 'sparse' not in self.options.suffix:
            for _, edge_indices in corner_edge_map.items():
                for edge_index_1, edge_index_2 in itertools.combinations(edge_indices, 2):
                    graph_edge_index.append([edge_index_offset + edge_index_1, edge_index_offset + edge_index_2])
                    graph_edge_index.append([edge_index_offset + edge_index_2, edge_index_offset + edge_index_1])                    
                    continue
                for edge_index in edge_indices:
                    edge_neighbors[edge_index].append(edge_indices)
                    continue
                continue
            graph_edge_attr = np.concatenate([graph_edge_attr, np.ones((len(graph_edge_index) - len(graph_edge_attr), 1), dtype=np.float32)], axis=0)
            pass
        left_edges = [np.array([[edge_index, neighbor] for neighbor in neighbors[0]]) for edge_index, neighbors in enumerate(edge_neighbors)]
        left_edges = np.concatenate(left_edges, axis=0)
        left_edges = left_edges[left_edges[:, 0] != left_edges[:, 1]]
        right_edges = [np.array([[edge_index, neighbor] for neighbor in neighbors[1]]) for edge_index, neighbors in enumerate(edge_neighbors)]
        right_edges = np.concatenate(right_edges, axis=0)
        right_edges = right_edges[right_edges[:, 0] != right_edges[:, 1]]
        graph_edge_index = np.array(graph_edge_index)
        
        return e_from_corners, c_e_assignment, graph_edge_index.transpose(), graph_edge_attr, left_edges, right_edges

    def compute_dists(self, corners_det, edges_det, c_e_assignment, thresh=2.0):

        # # compute corner dist
        # y, x, _, _ = np.split(corners_det, 4, axis=-1)
        # c_dist = np.sqrt((x - x.transpose(1, 0))**2 + (y - y.transpose(1, 0))**2)
        # ind_pos = c_dist<thresh
        # ind_neg = c_dist>=thresh
        # c_dist[ind_pos] = 1.0
        # c_dist[ind_neg] = 0.0
        # np.fill_diagonal(c_dist, 0.0)

        # # compute edge dist
        # y1, x1, y2, x2 = np.split(edges_det, 4, axis=-1)
        # y3, x3, y4, x4 = np.split(edges_det, 4, axis=-1)

        # dist13 = np.sqrt((x1 - x3.transpose(1, 0))**2 + (y1 - y3.transpose(1, 0))**2)
        # dist14 = np.sqrt((x1 - x4.transpose(1, 0))**2 + (y1 - y4.transpose(1, 0))**2)
        # dist23 = np.sqrt((x2 - x3.transpose(1, 0))**2 + (y2 - y3.transpose(1, 0))**2)
        # dist24 = np.sqrt((x2 - x4.transpose(1, 0))**2 + (y2 - y4.transpose(1, 0))**2)

        # d1 = dist13 + dist24
        # d2 = dist14 + dist23

        # e_dist = np.stack([d1, d2], axis=-1)
        # e_dist = np.min(e_dist, axis=-1)
        # ind_pos = e_dist<thresh*2
        # ind_neg = e_dist>=thresh*2
        # e_dist[ind_pos] = 1.0
        # e_dist[ind_neg] = 0.0
        # np.fill_diagonal(e_dist, 0.0)

        # compute corner-edge dist
        r_dist = np.zeros((corners_det.shape[0], edges_det.shape[0]))
        corner_edge_pairs = []
        edge_corner = {}

        for i, c in enumerate(corners_det):
            for j in range(edges_det.shape[0]):
                c1 = c_e_assignment[j][0]
                c2 = c_e_assignment[j][1]
                if np.array_equal(c1, c) or np.array_equal(c2, c):
                    r_dist[i, j] = 1.0
                    corner_edge_pairs.append([i, j])
                    if j not in edge_corner:
                        edge_corner[j] = []
                        pass
                    edge_corner[j].append(i)
        corner_edge_pairs = np.array(corner_edge_pairs)
        edge_corner = np.array(list(edge_corner.values()))

        return r_dist, corner_edge_pairs, edge_corner

    def compute_corner_angle_bins(self, ce_mat, edges_det, edges_gt, corners_det, n_bins=72, delta_degree=5.0):

        ce_angles_bins = np.zeros_like(ce_mat)-1
        c_angles_gt = np.zeros((ce_mat.shape[0], n_bins))
        for i in range(ce_mat.shape[0]):
            y, x, _, _ = corners_det[i]

            for j, e in enumerate(edges_det):
                
                if ce_mat[i, j]>0:

                    # compute angle
                    y1, x1, y2, x2 = e
                    pc = np.array([x, y])
                    pp = np.array([0, 1])
                    pr = np.array([x2, y2]) if np.array_equal(np.array([x1, y1]), pc) else np.array([x1, y1])
                    pr -= pc
                    cosine_angle = np.dot(pp, pr) / (np.linalg.norm(pp) * np.linalg.norm(pr) + 1e-8)
                    angle = np.arccos(cosine_angle)
                    #print(angle, cosine_angle)
                    angle = 180.0 - np.degrees(angle) 
                    if x != min(x1, x2):
                        angle = 360.0-angle 

                    bin_num = (int(angle/delta_degree)%n_bins)
                    ce_angles_bins[i, j] = bin_num
                    if edges_gt[j] == 1:
                        c_angles_gt[i, bin_num] = 1.0

                # im = np.zeros((256, 256, 3)).astype('uint8')
                # im = Image.fromarray(im)
                # draw = ImageDraw.Draw(im)
                # draw.line((x1, y1, x2, y2), fill='blue')
                # draw.ellipse((x-2, y-2, x+2, y+2), fill='red')
                # print(edges_gt[j])
                # print(angle)
                # print(bin_num)
                # plt.imshow(im)
                # plt.show()

        return ce_angles_bins, c_angles_gt    
        
    def load_annot(self, graph, rot, flip):

        # prepare edge instances for this image
        edge_set = set()
        for v1 in graph:
            for v2 in graph[v1]:
                x1, y1 = v1
                x2, y2 = v2
                # make an order
                if x1 > x2:
                    x1, x2, y1, y2 = x2, x1, y2, y1  # swap
                elif x1 == x2 and y1 > y2:
                    x1, x2, y1, y2 = x2, x1, y2, y1  # swap
                else:
                    pass

                x1, y1  = self.rotate_coords(np.array([256, 256]), np.array([x1, y1]), rot)
                x2, y2  = self.rotate_coords(np.array([256, 256]), np.array([x2, y2]), rot)
                if flip:
                    x1, y1 = (128-abs(128-x1), y1) if x1 > 128 else (128+abs(128-x1), y1)
                    x2, y2 = (128-abs(128-x2), y2) if x2 > 128 else (128+abs(128-x2), y2)
                edge = (y1, x1, y2, x2)
                edge_set.add(edge)
        edge_set = np.array([list(e) for e in edge_set])

        corner_set = []
        for v in graph:
            x, y = v
            x, y  = self.rotate_coords(np.array([256, 256]), np.array([x, y]), rot)
            if flip:
                x, y = (128-abs(128-x), y) if x > 128 else (128+abs(128-x), y)
            corner = [y, x, -1, -1]
            corner_set.append(corner)

        return corner_set, edge_set

    def compute_gt(self, corners_det, corners_annot, edges_det_from_corners, edges_annot, max_dist=8.0, shape=256):

        # init
        corners_det = np.array(corners_det)
        corners_annot = np.array(corners_annot)

        gt_c = np.zeros(corners_det.shape[0])
        gt_e = np.zeros(edges_det_from_corners.shape[0])

        # load ground-truth
        c_annots = corners_annot[:, :2]

        # load detections
        c_dets = corners_det[:, :2]

        # assign corner detection with annotation minimize linear sum assignment
        assigned_corners_ = []
        dist = cdist(c_annots, c_dets)

        annots_idx, dets_idx = np.arange(dist.shape[0]), np.argmin(dist, -1)
        #annots_idx, dets_idx = np.argmin(dist, 0), np.arange(dist.shape[1])

        # apply threshold
        assignment_filtered = []
        for i, j in zip(annots_idx, dets_idx):
            if dist[i, j] < max_dist:
                assignment_filtered.append((i, j))

        # get edges enpoints to corner dets mapping
        edges_inds_dets = []
        for e in edges_det_from_corners:
            y1, x1, y2, x2 = e
            inds = []
            for k, c in enumerate(corners_det):
                y3, x3, _, _ = c
                if (x1 == x3) and (y1 == y3):
                    inds.append(k)
                elif (x2 == x3) and (y2 == y3):
                    inds.append(k)
            edges_inds_dets.append(inds)
        edges_inds_dets = np.array(edges_inds_dets)

        # get edges enpoints to corner dets mapping
        edges_inds_annots = []
        for e in edges_annot:
            y1, x1, y2, x2 = e
            inds = []
            for k, c in enumerate(corners_annot):
                y3, x3, _, _ = c
                if (x1 == x3) and (y1 == y3):
                    inds.append(k)
                elif (x2 == x3) and (y2 == y3):
                    inds.append(k)
            edges_inds_annots.append(inds)
        edges_inds_annots = np.array(edges_inds_annots)

        # get corners gt
        corner_map_det_to_annot = {}
        for (i, j) in assignment_filtered:
            if j in corner_map_det_to_annot:
                #corner_map_det_to_annot[j].append(i)
                if dist[i, j] < dist[corner_map_det_to_annot[j][0], j]:
                    corner_map_det_to_annot[j] = [i]
                    pass
            else:
                corner_map_det_to_annot[j] = [i]
            gt_c[j] = 1

        # get edges gt
        for k, e_det in enumerate(edges_inds_dets):
            det_i, det_j = e_det
            if (gt_c[det_i] == 1) and (gt_c[det_j] == 1):
                for annot_i in corner_map_det_to_annot[det_i]:
                    for annot_j in corner_map_det_to_annot[det_j]: 
                        for e_annot in edges_inds_annots:
                            m, n = e_annot
                            if ((m == annot_i) and (n == annot_j)) or((n == annot_i) and (m == annot_j)):
                                gt_e[k] = 1
        return gt_c, gt_e
 

    def rotate_coords(self, image_shape, xy, angle):
        org_center = (image_shape-1)/2.
        rot_center = (image_shape-1)/2.
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a), -org[0]*np.sin(a) + org[1]*np.cos(a)])
        return new+rot_center

    def augment(self, imgs, corners, edges):
        size = imgs.shape[1]
        ys, xs = np.nonzero(imgs.min(-1) < 250)
        vertices = np.array([[xs.min(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.min()], [xs.max(), ys.max()]])
        #center = vertices[0] + np.random.random(2) * (vertices[-1] - vertices[0])
        angle = np.random.random() * 360
        #rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        #print('vertices', tuple(((vertices[0] + vertices[-1]) / 2).tolist()))
        #rotation_matrix = cv2.getRotationMatrix2D(tuple(((vertices[0] + vertices[-1]) / 2).tolist()), angle, 1)
        rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1)
        transformed_vertices = np.matmul(rotation_matrix, np.concatenate([vertices, np.ones((len(vertices), 1))], axis=-1).transpose()).transpose()
        mins = transformed_vertices.min(0)
        maxs = transformed_vertices.max(0)
        max_range = (maxs - mins).max()
        new_size = min(max_range, size) + max(size - max_range, 0) * np.random.random()
        scale = float(new_size) / max_range
        offset = (np.random.random(2) * 2 - 1) * (size - (maxs - mins) * scale) / 2 + (size / 2 - (maxs + mins) / 2 * scale)
        #offset = 0 * (size - (maxs - mins) * scale) + (size / 2 - (maxs + mins) / 2 * scale)
        translation_matrix = np.array([[scale, 0, offset[0]], [0, scale, offset[1]]])
        transformation_matrix = np.matmul(translation_matrix, np.concatenate([rotation_matrix, np.array([[0, 0, 1]])], axis=0))
        #transformation_matrix_shuffled = np.stack([transformation_matrix[1], transformation_matrix[0]], axis=0)
        # transformation_matrix_shuffled = transformation_matrix.copy()
        # transformation_matrix_shuffled[1, 2] = transformation_matrix[0, 2]
        # transformation_matrix_shuffled[0, 2] = transformation_matrix[1, 2]        
        imgs = cv2.warpAffine(imgs, transformation_matrix, (size, size), borderValue=(255, 255, 255))
        #print(corners)
        corners_ori = corners
        corners = np.matmul(transformation_matrix, np.concatenate([corners[:, [1, 0]], np.ones((len(corners), 1))], axis=-1).transpose()).transpose()
        if (corners.min() < 0 or corners.max() > 256) and False:
            cv2.imwrite('test/image.png', imgs.astype(np.uint8))
            print(vertices)
            print(corners_ori)
            print(np.matmul(rotation_matrix, np.concatenate([corners_ori[:, [1, 0]], np.ones((len(corners), 1))], axis=-1).transpose()).transpose() * scale)
            print(transformed_vertices * scale)
            print(scale, size, new_size, mins, maxs, (maxs - mins) * scale, (maxs + mins) / 2 * scale, offset, corners.min(0), corners.max(0))
            print((size - (maxs - mins) * scale) / 2, size / 2 - (maxs + mins) / 2 * scale, (maxs + mins) / 2 * scale)
            exit(1)
            pass
        corners = corners[:, [1, 0]]            
        corners = np.clip(np.round(corners).astype(np.int32), 0, size - 1)
        edge_points = edges.reshape((-1, 2))[:, [1, 0]]
        edge_points = np.matmul(transformation_matrix, np.concatenate([edge_points, np.ones((len(edge_points), 1))], axis=-1).transpose()).transpose()
        edges = edge_points[:, [1, 0]].reshape((-1, 4))
        edges = np.clip(np.round(edges).astype(np.int32), 0, size - 1)
        #print('rotation', rotation_matrix, translation_matrix)
        #print(corners)
        return imgs, corners, edges

    def visualize(self, mode='last_mistake', edge_state=None, building_idx=None, post_processing=False):
        image = self.rgb.copy()        
        corner_image, corner_masks = self.compute_corner_image(self.corners_det)
        image[corner_image > 0.5] = np.array([255, 0, 0], dtype=np.uint8)
        if debug:
            for corner_index, corner in enumerate(self.corners_det):
                cv2.putText(image, str(corner_index), (corner[1], corner[0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255))
                continue
            pass
        
        edge_image = image.copy()
        if 'last' in mode:
            edge_mask = draw_edges(self.predicted_edges[-1], self.edges_det)
        else:
            if post_processing:
                edge_state = self.post_processing(edge_state)
            else:
                edge_state = edge_state > 0.5
                pass
            edge_mask = draw_edges(edge_state, self.edges_det)
            pass
        edge_image[edge_mask > 0.5] = np.array(color, dtype=np.uint8)
        images = [edge_image]
        if 'mistake' in mode:
            if (self.predicted_edges[-1] - self.edges_gt).max() > 0:
                for edges in self.predicted_edges:
                    if (edges - self.edges_gt).max() > 0:
                        edge_image = image.copy()
                        edge_mask = draw_edges(edges, self.edges_det)
                        edge_image[edge_mask > 0.5] = np.array(color, dtype=np.uint8)
                        images.append(edge_image)
                        break
                    continue
            else:            
            #if 'mistake' in mode and len(images) == 1:
                images.append(np.zeros(edge_image.shape, dtype=np.uint8))
                pass
            pass

        if 'draw_annot' in mode:    
            corner_annot, corner_masks = self.compute_corner_image(np.array(self.corners_annot).astype('int'))
            corner_image_annot = self.rgb.copy()
            corner_image_annot[corner_annot > 0.5] = np.array([255, 0, 0], dtype=np.uint8)
            images.append(corner_image_annot)

            edge_image_annot = corner_image_annot.copy()
            edge_mask = draw_edges(np.ones(self.edges_annot.shape[0]), np.array(self.edges_annot.astype('int')))
            edge_image_annot[edge_mask > 0.5] = np.array([0, 255, 0], dtype=np.uint8)
            images.append(edge_image_annot)

        return images, np.array([(np.logical_and(self.predicted_edges[-1] == self.edges_gt, self.edges_gt == 1)).sum(), self.predicted_edges[-1].sum(), self.edges_gt.sum(), int(np.all(self.predicted_edges[-1] == self.edges_gt))])

    def visualize_multiple_loops(self, loop_edges):
        image = self.rgb.copy()        
        corner_image, corner_masks = self.compute_corner_image(self.corners_det)
        image[corner_image > 0.5] = np.array([255, 0, 0], dtype=np.uint8)
        loop_image = image.copy()

        for loop in loop_edges:
            e_inds = np.where(loop>0)[0]
            c = np.uint8(np.random.uniform(0, 255, 3))
            c = tuple(map(int, c))
            for e in e_inds:
                y1, x1, y2, x2 = self.edges_det[e]
                cv2.line(loop_image, (x1, y1), (x2, y2), thickness=3, color=c)

        return loop_image


    def visualizeLoops(self, loop_corners, loop_state):
        image = self.rgb.copy()        
        corner_image, corner_masks = self.compute_corner_image(self.corners_det)
        image[corner_image > 0.5] = np.array([255, 0, 0], dtype=np.uint8)
        loop_image = image.copy()
        for loop, state in zip(loop_corners, loop_state.tolist()):
            if state < 0.5:
                continue
            loop = loop.tolist()
            for corner_index_1, corner_index_2 in zip(loop, loop[1:] + loop[:1]):
                corner_1 = self.corners_det[corner_index_1]
                corner_2 = self.corners_det[corner_index_2]
                cv2.line(loop_image, (corner_1[1], corner_1[0]), (corner_2[1], corner_2[0]), thickness=3, color=(0, 255, 0))
                continue
        return loop_image

    def colinear_edges(self):
        colinear_edges = []
        dot_threshold = np.cos(np.deg2rad(2))
        for corner_index in range(len(self.corner_edge)):
            corner = self.corners_det[corner_index]
            edge_indices = self.corner_edge[corner_index].nonzero()[0].tolist()
            for edge_index_1, edge_index_2 in itertools.combinations(edge_indices, 2):
                direction_1 = self.edges_det[edge_index_1].reshape((2, 2)).mean(0) - corner
                direction_2 = self.edges_det[edge_index_2].reshape((2, 2)).mean(0) - corner
                if np.dot(direction_1, direction_2) / (np.linalg.norm(direction_1) * np.linalg.norm(direction_2)) > dot_threshold:
                    colinear_edges.append([edge_index_1, edge_index_2])

                    # import matplotlib.pyplot as plt
                    # im = Image.new('RGB', (256, 256))
                    # draw = ImageDraw.Draw(im)
                    # y1, x1, y2, x2 = self.edges_det[edge_index_1]
                    # y3, x3, y4, x4 = self.edges_det[edge_index_2]
                    # draw.line((x1, y1, x2, y2), width=1, fill='blue')
                    # draw.line((x3, y3, x4, y4), width=1, fill='red')
                    # plt.imshow(im)
                    # plt.show()

                    pass
                continue
            continue
        colinear_edges = np.array(colinear_edges)
        return colinear_edges

    def post_processing(self, edge_confidence, debug=False):
        corner_edge = self.corner_edge > 0.5
        num_corners = len(corner_edge)
        num_edges = len(edge_confidence)

        edge_corner = self.edge_corner
        
        edge_state = edge_confidence > 0.5
        colinear_edges = self.colinear_edges()
        for edge_pair in colinear_edges:
            for c in range(2):
                if edge_state[edge_pair[c]] and not edge_state[edge_pair[1 - c]]:
                    edge_confidence[edge_pair[1 - c]] = min(edge_confidence[edge_pair[1 - c]], 1 - edge_confidence[edge_pair[c]])
                    pass
                continue
            continue
        colinear_edges = {(edge_pair.min(), edge_pair.max()): True for edge_pair in colinear_edges}
        valid_corner_neighbors = [{} for _ in range(num_corners)]
        invalid_corner_neighbors = [{} for _ in range(num_corners)]        
        for edge_index, (corner_pair, confidence) in enumerate(zip(edge_corner, edge_confidence)):
            if confidence > 0.5:
                valid_corner_neighbors[corner_pair[0]][corner_pair[1]] = (0.5 - confidence, edge_index)
                valid_corner_neighbors[corner_pair[1]][corner_pair[0]] = (0.5 - confidence, edge_index)
            else:
                invalid_corner_neighbors[corner_pair[0]][corner_pair[1]] = (0.5 - confidence, edge_index)
                invalid_corner_neighbors[corner_pair[1]][corner_pair[0]] = (0.5 - confidence, edge_index)
                pass
            continue
        
        ## Compute two paths between any pair of corners, one passing through chosen edges and one doesn't
        path_distances = []
        path_lengths = []
        ## Compute the path through inactive edges        
        for _ in range(1):
            corner_path_distances = np.full((num_corners, num_corners), fill_value=100.0)
            corner_path_lengths = np.zeros((num_corners, num_corners))
            for edge_index in range(num_edges):
                if not edge_state[edge_index]:
                    distance = 0.5 - edge_confidence[edge_index]
                    #print(edge_index, edge_corner[edge_index], distance)
                    corner_path_distances[edge_corner[edge_index][0], edge_corner[edge_index][1]] = distance
                    corner_path_distances[edge_corner[edge_index][1], edge_corner[edge_index][0]] = distance
                    corner_path_lengths[edge_corner[edge_index][0], edge_corner[edge_index][1]] = 1
                    corner_path_lengths[edge_corner[edge_index][1], edge_corner[edge_index][0]] = 1
                    pass
                continue
            while True:
                corner_neighbor = corner_path_distances.argmin(-1)
                new_corner_path_distances = np.minimum(corner_path_distances, corner_path_distances.min(-1, keepdims=True) + corner_path_distances[corner_neighbor])
                #corner_nearest_neighbors.append(corner_nearest_neighbor)

                changed_mask = new_corner_path_distances < corner_path_distances
                changed_mask[np.diag(np.ones(len(changed_mask), dtype=np.bool))] = False
                if not np.any(changed_mask):
                    break
                corner_path_lengths[changed_mask] = (corner_path_lengths[np.arange(num_corners, dtype=np.int32), corner_neighbor] + np.ones(corner_path_lengths.shape))[changed_mask]
                #print(corner_path_distances[4], new_corner_path_distances[4], corner_neighbor[4], corner_path_distances[corner_neighbor[4]])
                #exit(1)
                corner_path_distances = new_corner_path_distances
                continue
            path_distances.append(corner_path_distances)
            path_lengths.append(corner_path_lengths)            
            continue

        ## Compute the path through active edges
        for _ in range(1):
            corner_path_distances = np.full((num_corners, num_corners), fill_value=100.0)
            corner_path_lengths = np.zeros((num_corners, num_corners))
            for edge_index in range(num_edges):
                if edge_state[edge_index]:
                    distance = 0.5 - edge_confidence[edge_index]
                    corner_path_distances[edge_corner[edge_index][0], edge_corner[edge_index][1]] = distance
                    corner_path_distances[edge_corner[edge_index][1], edge_corner[edge_index][0]] = distance
                    corner_path_lengths[edge_corner[edge_index][0], edge_corner[edge_index][1]] = 1
                    corner_path_lengths[edge_corner[edge_index][1], edge_corner[edge_index][0]] = 1
                    pass
                continue

            #visited_corner_mask = np.zeros(num_corners, np.bool)
            for corner_index in range(num_corners):
                corner_paths = [[[corner_index, neighbor], distance, [edge_index]] for neighbor, (distance, edge_index) in valid_corner_neighbors[corner_index].items() if distance <= corner_path_distances[corner_index, neighbor]]
                while len(corner_paths) > 0:
                    corner_path = corner_paths[0]
                    corner_paths = corner_paths[1:]
                    #print(corner_path)
                    for neighbor, (distance, edge_index) in valid_corner_neighbors[corner_path[0][-1]].items():
                        if neighbor in corner_path[0]:
                            continue
                        if (min(edge_index, corner_path[2][-1]), max(edge_index, corner_path[2][-1])) in colinear_edges:
                            continue
                        new_distance = corner_path[1] + distance
                        #print(corner_index, neighbor, new_distance, corner_path_distances[corner_index, neighbor])
                        if new_distance / (len(corner_path[0]) + 1) <= corner_path_distances[corner_index, neighbor] / len(corner_path[0]):
                            corner_path_distances[corner_index, neighbor] = new_distance
                            corner_path_lengths[corner_index, neighbor] = len(corner_path[0])
                            new_corner_path = [corner_path[0] + [neighbor], new_distance, corner_path[2] + [edge_index]]
                            corner_paths.append(new_corner_path)
                            pass
                        continue
                    continue
                continue                
            path_distances.append(corner_path_distances)
            path_lengths.append(corner_path_lengths)
            continue

        ## The optimal corner to connect
        #mean_path_distances = (path_distances[0] * path_lengths[0] + path_distances[1] * path_lengths[1]) / np.maximum(path_distances[0] + path_lengths[1], 1)

        mean_path_distances = path_distances[0] + path_distances[1] / path_lengths[1]
        
        mean_path_distances[np.diag(np.ones(len(mean_path_distances), dtype=np.bool))] = 100

        corner_neighbor = mean_path_distances.argmin(-1)
        corner_neighbor[mean_path_distances.min(-1) > 0] = -1

        if debug:
            print(edge_corner[edge_state])
            print(path_distances)
            print(path_lengths)            
            print(mean_path_distances)                
            print('neighbor', corner_neighbor)
            pass
        
        #valid_corners = corner_degrees > 0
        #valid_corner_mask = corner_degrees >= 2
        #valid_edge_mask = np.logical_and(np.logical_and(valid_corner_mask[edge_corner[:, 0]], valid_corner_mask[edge_corner[:, 1]]), edge_state)

        corner_pairs = []
        invalid_corner_indices = []
        while True:
            corner_edge_state = corner_edge * edge_state        
            corner_degrees = corner_edge_state.sum(-1)        
            singular_corners = (corner_degrees == 1).nonzero()[0].tolist()
            if len(singular_corners) == 0:
                break
            for corner_index in singular_corners:
                target = corner_neighbor[corner_index]
                if target < 0 or target == corner_index:
                    invalid_corner_indices.append(corner_index)
                    continue
                if debug:
                    print('target', corner_index, target)
                    pass
                corner_paths = [[[corner_index, neighbor], distance] for neighbor, (distance, _) in invalid_corner_neighbors[corner_index].items()]
                corner_paths = sorted(corner_paths, key=lambda x: x[1])
                while len(corner_paths) > 0:
                    corner_path = corner_paths[0]
                    corner_paths = corner_paths[1:]
                    if corner_path[0][-1] == target:
                        if debug:
                            print('path', corner_path)
                            pass
                        for index in range(len(corner_path[0]) - 1):
                            corner_pairs.append((corner_path[0][index], corner_path[0][index + 1]))
                            continue
                        break
                    for neighbor, (distance, _) in invalid_corner_neighbors[corner_path[0][-1]].items():
                        if neighbor in corner_path[0]:
                            continue
                        new_corner_path = [corner_path[0] + [neighbor], corner_path[1] + distance]
                        inserted = False
                        for index in range(len(corner_paths)):
                            if corner_paths[index][1] > new_corner_path[1]:
                                corner_paths = corner_paths[:index] + [new_corner_path] + corner_paths[index:]
                                inserted = True
                                break
                            continue
                        if not inserted:
                            corner_paths += [new_corner_path]
                            pass
                        continue
                    continue
                continue
            if debug:
                print('results', corner_pairs, invalid_corner_indices)
                pass
            new_edge_state = edge_state.copy()
            for corner_pair in corner_pairs:
                edge_index = invalid_corner_neighbors[corner_pair[0]][corner_pair[1]][1]
                new_edge_state[edge_index] = 1
                continue
            for invalid_corner_index in invalid_corner_indices:
                #for _, (_, edge_index) in corner_neighbors[invalid_corner_index].items():
                new_edge_state[corner_edge[invalid_corner_index]] = 0
                continue
            if np.all(edge_state == new_edge_state):
                break            
            edge_state = new_edge_state
            continue
        return edge_state
    
    ## Find closed loop inside the graph
    def compute_edge_groups(self, corner_pair_edge_map, edges_gt):
        corner_neighbors_map = {}
        for corner_pair, edge_index in corner_pair_edge_map.items():
            if edges_gt[edge_index] < 0.5:
                continue
            for c in range(2):
                if corner_pair[c] not in corner_neighbors_map:
                    corner_neighbors_map[corner_pair[c]] = []
                    pass
                corner_neighbors_map[corner_pair[c]].append(corner_pair[1 - c])
                continue
            continue
        
        corner_groups = []
        visited_corners = {}
        for corner_index in corner_neighbors_map.keys():
            if corner_index in visited_corners:
                continue
            visited_corners[corner_index] = True
            queue = [(corner_index, [corner_index])]
            while len(queue) > 0:
                element = queue[0]
                queue = queue[1:]
                neighbors = corner_neighbors_map[element[0]]
                for neighbor in neighbors:
                    visited_corners[neighbor] = True
                    if neighbor in element[1]:
                        index = element[1].index(neighbor)
                        if len(element[1]) - index >= 3:
                            corner_groups.append(element[1][index:])
                            pass
                    else:
                        queue.append((neighbor, element[1] + [neighbor]))
                        pass
                    continue
                continue
            continue
        
        edge_groups = []
        valid_corner_groups = []
        for corner_group in corner_groups:
            edge_group = []
            for corner_index_1 in corner_group:
                for corner_index_2 in corner_group:
                    if corner_index_2 <= corner_index_1:
                        continue
                    edge_group.append(corner_pair_edge_map[(corner_index_1, corner_index_2)])
                    continue
                continue
            edge_group = np.array(edge_group)
            num_edges_gt = edges_gt[edge_group].sum()
            ## The number of edges equals to the number of corners for a closed loop
            if num_edges_gt == len(corner_group):
                corner_group = np.array(corner_group)
                reverse_corner_group = corner_group.copy()
                reverse_corner_group[1:] = reverse_corner_group[1:][::-1]
                exists = len(valid_corner_groups) > 0 and max([np.all(group == reverse_corner_group) for group in valid_corner_groups])
                if not exists:
                    edge_groups.append(edge_group)
                    valid_corner_groups.append(corner_group)
                    pass
                pass
            continue

        return edge_groups
    
    def create_loops_sample_edge_features(self, phase=None):

        with np.load('{}/{}_0_False.npz'.format(self.options.predicted_edges_path, self._id)) as data:
            edges_confidence = data['arr_0']
            
        all_loops, loop_labels, edges_loops, loop_acc = self.find_loops(edges_confidence, self.edge_corner, self.corners_det, self.edges_gt, self.corners_det.shape[0])
        
        if phase == "train":
            loop_labels, to_keep = self.filter_loops(edges_loops, loop_labels)
            all_loops = [x for k, x in enumerate(all_loops) if k in to_keep]
            edges_loops = edges_loops[to_keep]

        loop_feats = []
        for k, loop in enumerate(all_loops):
            rot = 0
            flip = False
            # if phase == "train":
            #     rot = np.random.choice([0, 90, 180, 270])
            #     flip = np.random.choice([False, True])
            with np.load('{}/{}_{}_{}.npz'.format(self.options.predicted_edges_path, self._id, rot, flip)) as data:
                edge_feats = data['arr_1']

            inds = np.where(edges_loops[k]==1)[0]
            y1, x1, y2, x2 = np.split(self.edges_det[inds, :]/255.0, 4, -1)
            alg_feats = np.concatenate([y1, x1, y2, x2, y1**2, x1**2, y2**2, x2**2, y1*x1, y1*x2, y2*x1, y2*x2], -1)
            loop_feat = np.concatenate([edge_feats[inds, :], alg_feats], -1)

            loop_feat = np.pad(loop_feat, ((0, 20-loop_feat.shape[0]), (0, 0)), 'constant', constant_values=0)
            loop_feats.append(loop_feat)
        loop_feats = np.stack(loop_feats)

        if phase == "train":

            # balance samples
            pos_inds = np.where(loop_labels==1)[0]
            neg_inds = np.where(loop_labels==0)[0]
            num_pos = pos_inds.shape[0]
            num_neg = neg_inds.shape[0]
            np.random.shuffle(neg_inds)
            np.random.shuffle(pos_inds)

            if num_pos < num_neg:
                neg_inds = neg_inds[:max(1, 3*num_pos)]  
            else:
                pos_inds = pos_inds[:max(1, num_neg//3)]  

            loop_feats = np.concatenate([loop_feats[pos_inds, :], loop_feats[neg_inds, :]])
            loop_labels = np.concatenate([loop_labels[pos_inds], loop_labels[neg_inds]])

        # print("neg", np.where(loop_labels==0)[0].shape)
        # print("pos", np.where(loop_labels==1)[0].shape)

        return loop_feats, loop_labels, edges_loops, all_loops, loop_acc

    def create_multi_loops_sample_cmp(self, phase=None):

        with np.load('{}/{}_0_False.npz'.format(self.options.predicted_edges_path, self._id)) as data:
            edges_confidence = data['arr_0']
            
        all_loops, loop_labels, edges_loops, loop_acc = self.find_loops(edges_confidence, self.edge_corner, self.corners_det, self.edges_gt, self.corners_det.shape[0])
        
        if phase == "train":
            loop_labels, to_keep = self.filter_loops(edges_loops, loop_labels)
            all_loops = [x for k, x in enumerate(all_loops) if k in to_keep]
            edges_loops = edges_loops[to_keep]

        e_xys_dict = {}
        for rot in [0, 90, 180, 270]:
            for flip in [False, True]:
                e_xys_dict["{}_{}".format(rot, flip)] = self.compute_edges_map(self.edges_det, rot=rot, flip=flip)

        loop_feats = []
        for k, loop in enumerate(all_loops):
            rot = 0
            flip = False
            if phase == "train":
                rot = np.random.choice([0, 90, 180, 270])
                flip = np.random.choice([False, True])

            with np.load('{}/{}_{}_{}.npz'.format(self.options.predicted_edges_path, self._id, rot, flip)) as data:
                edge_feats = data['arr_1']
                edge_confs = data['arr_0']

            e_inds = np.where(edges_loops[k]==1)[0]
            e_xys = e_xys_dict["{}_{}".format(rot, flip)]

            # place features in grid
            rot_im = Image.fromarray(self.imgs.copy()).resize((128, 128)).rotate(rot)
            if flip == True:
                rot_im = rot_im.transpose(Image.FLIP_LEFT_RIGHT)
            #rot_im = np.transpose(np.array(rot_im)/255.0, (2, 0, 1))
            print(rot_im.shape)
            asd
            rot_im = np.array(rot_im)/255.0

            grid = np.zeros((128, 128, 128))
            for e in e_inds:
                xs_inds = np.array(np.where(e_xys[e, 0, :]>=0)[0])
                ys_inds = np.array(np.where(e_xys[e, 1, :]>=0)[0])

                if xs_inds.shape[0] > 0:
                    xs = e_xys[e, 0, xs_inds]
                    ys = e_xys[e, 1, ys_inds]

                    feat_in = edge_feats[e, :]
                    feat_in = feat_in[:, np.newaxis]
                    feat_in = np.repeat(feat_in, xs_inds.shape[0], axis=1)

                    #feat_in = edge_confs[e]
                    grid[:, xs, ys] += feat_in

            # debug_arr = np.sum(grid, 0)
            # inds = np.where(debug_arr!=0)
            # debug_arr[inds] = 255.0
            # print(loop_labels[k])
            # debug_im = Image.fromarray(debug_arr)
            # import matplotlib.pyplot as plt
            # plt.imshow(debug_im)
            # plt.show()
            grid = np.concatenate([grid, rot_im], 0)
            

            loop_feats.append(grid)
        loop_feats = np.stack(loop_feats)

        if phase == "train":

            # balance samples
            pos_inds = np.where(loop_labels==1)[0]
            neg_inds = np.where(loop_labels==0)[0]
            num_pos = pos_inds.shape[0]
            num_neg = neg_inds.shape[0]
            np.random.shuffle(neg_inds)
            np.random.shuffle(pos_inds)

            if num_pos < num_neg:
                neg_inds = neg_inds[:max(1, 5*num_pos)]  
            else:
                pos_inds = pos_inds[:max(1, num_neg//5)]  

            loop_feats = np.concatenate([loop_feats[pos_inds, :, :, :], loop_feats[neg_inds, :, :, :]])
            loop_labels = np.concatenate([loop_labels[pos_inds], loop_labels[neg_inds]])

        # print("neg", np.where(loop_labels==0)[0].shape)
        # print("pos", np.where(loop_labels==1)[0].shape)

        return loop_feats, loop_labels, edges_loops, all_loops, loop_acc

    def create_loops_sample_cmp(self, phase=None, n_bins=36):

        with np.load('{}/{}_0_False.npz'.format(self.options.predicted_edges_path, self._id)) as data:
            edges_confidence = data['arr_0']
            
        all_loops, loop_labels, edges_loops, loop_acc = self.find_loops(edges_confidence, self.edge_corner, self.corners_det, self.edges_gt, self.corners_det.shape[0])
        
        if phase == "train":
            loop_labels, to_keep = self.filter_loops(edges_loops, loop_labels)
            all_loops = [x for k, x in enumerate(all_loops) if k in to_keep]
            edges_loops = edges_loops[to_keep]

        e_xys_dict = {}
        for rot in [0, 90, 180, 270]:
            for flip in [False, True]:
                e_xys_dict["{}_{}".format(rot, flip)] = self.compute_edges_map(self.edges_det, rot=rot, flip=flip)

        loop_feats = []
        for k, loop in enumerate(all_loops):
            rot = 0
            flip = False
            if phase == "train":
                rot = np.random.choice([0, 90, 180, 270])
                flip = np.random.choice([False, True])

            with np.load('{}/{}_{}_{}.npz'.format(self.options.predicted_edges_path, self._id, rot, flip)) as data:
                edge_feats = data['arr_1']
                edge_confs = data['arr_0']

            e_inds = np.where(edges_loops[k]==1)[0]
            e_xys = e_xys_dict["{}_{}".format(rot, flip)]


            # compute bins
            edges_coords_aug = self.rotate_flip(self.edges_det, rot, flip)
            one_hot, angles = self.compute_angles(edges_coords_aug)

            # place features in grid
            rot_im = Image.fromarray(self.imgs.copy()).resize((128, 128)).rotate(rot)
            rot_rgb = Image.fromarray(self.rgb.copy()).resize((128, 128)).rotate(rot)
            if flip == True:
                rot_im = rot_im.transpose(Image.FLIP_LEFT_RIGHT)
                rot_rgb = rot_rgb.transpose(Image.FLIP_LEFT_RIGHT)
            rot_im = np.array(rot_im)[np.newaxis, :, :]/255.0
            rot_rgb = np.array(rot_rgb)/255.0
            rot_rgb = np.transpose(rot_rgb, (2, 0, 1))
            grid = np.zeros((128+n_bins, 128, 128))

            for e in e_inds:
                xs_inds = np.array(np.where(e_xys[e, 0, :]>=0)[0])
                ys_inds = np.array(np.where(e_xys[e, 1, :]>=0)[0])

                if xs_inds.shape[0] > 0:
                    xs = e_xys[e, 0, xs_inds]
                    ys = e_xys[e, 1, ys_inds]

                    feat_in = np.concatenate([edge_feats[e, :], one_hot[e, :]])
                    feat_in = feat_in[:, np.newaxis]
                    feat_in = np.repeat(feat_in, xs_inds.shape[0], axis=1)

                    #feat_in = edge_confs[e]
                    grid[:, xs, ys] += feat_in

            # debug_arr = np.sum(grid, 0)
            # inds = np.where(debug_arr!=0)
            # debug_arr[inds] = 255.0
            # print(loop_labels[k])
            # debug_im = Image.fromarray(debug_arr)
            # debug_im = Image.fromarray((rot_im[0, :, :]*255).astype("int32"))
            # import matplotlib.pyplot as plt
            # plt.imshow(debug_im)
            # plt.show()
            grid = np.concatenate([grid, rot_rgb, rot_im], 0)
            loop_feats.append(grid)
        loop_feats = np.stack(loop_feats)

        if phase == "train":

            # balance samples
            pos_inds = np.where(loop_labels==1)[0]
            neg_inds = np.where(loop_labels==0)[0]
            num_pos = pos_inds.shape[0]
            num_neg = neg_inds.shape[0]
            np.random.shuffle(neg_inds)
            np.random.shuffle(pos_inds)

            if num_pos < num_neg:
                neg_inds = neg_inds[:max(1, 5*num_pos)]  
            else:
                pos_inds = pos_inds[:max(1, num_neg//5)]  

            loop_feats = np.concatenate([loop_feats[pos_inds, :, :, :], loop_feats[neg_inds, :, :, :]])
            loop_labels = np.concatenate([loop_labels[pos_inds], loop_labels[neg_inds]])

        # print("neg", np.where(loop_labels==0)[0].shape)
        # print("pos", np.where(loop_labels==1)[0].shape)

        return loop_feats, loop_labels, edges_loops, all_loops, loop_acc

    def create_loops_sample(self, phase=None):

        # imgs, corners_det, edges_det = self.augment(self.imgs.copy(), self.corners_det, self.edges_det)

        edges_confidence = np.load('{}/{}.npy'.format(self.options.predicted_edges_path, self._id))


        all_loops, loop_labels, edges_loops, loop_acc = self.find_loops(edges_confidence, self.edge_corner, self.corners_det, self.edges_gt, self.corners_det.shape[0])
        if phase == "train":
            #print("WARNING: FILTERING LOOPS!")
            loop_labels, to_keep = self.filter_loops(edges_loops, loop_labels)
            all_loops = [x for k, x in enumerate(all_loops) if k in to_keep]
            edges_loops = edges_loops[to_keep]

        loop_ims = []
        for k, loop in enumerate(all_loops):
            im = draw_edges(edges_loops[k], self.edges_det)

            # augmentation
            flip = np.random.choice([False, True])
            rot = np.random.choice([0, 90, 180, 270])
            rot_loop_im = Image.fromarray(im).rotate(rot)
            rot_rgb_im = Image.fromarray(self.imgs.copy()).rotate(rot)

            if flip:
                rot_loop_im = rot_loop_im.transpose(Image.FLIP_LEFT_RIGHT)
                rot_rgb_im = rot_rgb_im.transpose(Image.FLIP_LEFT_RIGHT)

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(np.array(rot_loop_im) * 255)
            # plt.figure()
            # plt.imshow(rot_rgb_im)
            # plt.show()

            rot_loop_im = np.array(rot_loop_im)
            rot_rgb_im = np.array(rot_rgb_im)

            im_in = np.concatenate([rot_loop_im[np.newaxis, :, :], np.transpose(rot_rgb_im/255.0, (2, 0, 1))], axis=0)
            loop_ims.append(im_in)
        loop_ims = np.stack(loop_ims)

        return loop_ims, loop_labels, edges_loops, all_loops, loop_acc

    def filter_loops(self, loop_edges, loop_labels):

        colinear_edges = self.colinear_edges()
        new_loop_labels = np.array(loop_labels)
        loop_gt_im = draw_edges(self.edges_gt_no_colinear, self.edges_det)
        im_colinear = {}
        
        for e1, e2 in colinear_edges:
            if e1 not in im_colinear:
                edges_on = np.zeros(self.edges_det.shape[0])
                edges_on[e1] = 1
                edges_im = draw_edges(edges_on, self.edges_det)
                im_colinear[e1] = edges_im

            if e2 not in im_colinear:
                edges_on = np.zeros(self.edges_det.shape[0])
                edges_on[e2] = 1
                edges_im = draw_edges(edges_on, self.edges_det)
                im_colinear[e2] = edges_im              
        
        to_keep = []
        for k, loop in enumerate(loop_edges):
            for e1, e2 in colinear_edges:
                iou = np.logical_and(im_colinear[e1], im_colinear[e2]).sum()/np.logical_or(im_colinear[e1], im_colinear[e2]).sum()
                if loop[e1] == 1 and loop[e2] == 1 and iou > .1:
                    new_loop_labels[k] = 0
                    break
            loop_im = draw_edges(loop, self.edges_det)
            valid_check = np.logical_and(loop_im, loop_gt_im).sum()/loop_im.sum()
            to_keep.append(k)
            if (new_loop_labels[k] == 1) and (valid_check < .95):
                new_loop_labels[k] = 0
            
            # loop_im = draw_edges(loop, self.edges_det)
            # print(valid_check)
            # print("label: ", new_loop_labels[k])
            # cv2.imshow('loop_gt', loop_gt_im)
            # cv2.imshow('loop', loop_im)
            # cv2.waitKey(0)

        new_loop_labels = new_loop_labels[to_keep]
        return new_loop_labels, np.array(to_keep)

    def find_loops(self, current_edges, edge_corner, corners, edges_gt, num_corners):
        all_loops = findLoopsModuleCPU2(torch.from_numpy(current_edges.astype(np.float32)), torch.from_numpy(edge_corner), num_corners, max_num_loop_corners=15, confidence_threshold=0.5, corners=torch.from_numpy(self.corners_det).float(), disable_colinear=False)        

        #all_loops = findLoopsModuleCPU(torch.from_numpy(current_edges.astype(np.float32)), torch.from_numpy(edge_corner), num_corners, max_num_loop_corners=20, confidence_threshold=0.7)        
        loop_corners = []
        loop_labels = []
        edges_loops = []
        loop_acc = []
        for accs, loops in all_loops:
            loops = loops.detach().cpu().numpy()

            # create corner sequence
            for loop in loops:
                loop_corners.append(corners[loop])

            # convert loop to edges
            for acc, loop in zip(accs, loops):
                label = 1
                edges_loop = np.zeros_like(current_edges)
                c1 = loop[0]
                #print(c1)
                # print(len(list(loop[1:])+[loop[0]]))
                for c2 in list(loop[1:])+[loop[0]]:
                    # check if is a true edge and get all edges in loop
                    is_true_edge = False
                    #print(c1, c2)
                    for k, e in enumerate(edge_corner):
                        if (np.array_equal(np.array([c1, c2]), e) == True) or (np.array_equal(np.array([c2, c1]), e) == True):
                            edges_loop[k] = 1
                            # print(np.array([c1, c2]), e)
                            if edges_gt[k] == 1:
                                is_true_edge = True
                        elif c1 == c2:
                            is_true_edge = True

                    if is_true_edge == False:
                        label = 0
                    c1 = c2
                    
                edges_loops.append(edges_loop)
                loop_labels.append(label)
                loop_acc.append(acc.detach().cpu().numpy())

                # print(np.where(edges_loop>0))
                # loop_im = draw_edges(edges_loop, self.edges_det)
                # print("label: ", label)
                # cv2.imshow('loop', loop_im)
                # cv2.waitKey(0)

        edges_loops = np.stack(edges_loops)
        loop_acc = np.stack(loop_acc)

        return loop_corners, np.array(loop_labels), edges_loops, loop_acc

    def add_colinear_gt(self):
        dot_product_threshold = np.cos(np.deg2rad(20))        
        directions = (self.edges_det[:, 2:4] - self.edges_det[:, :2]).astype(np.float32)
        directions = directions / np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-4)
        while True:
            has_change = False
            for edge_index_1, edge_gt_1 in enumerate(self.edges_gt):
                if edge_gt_1 < 0.5:
                    continue
                for edge_index_2, edge_gt_2 in enumerate(self.edges_gt):
                    if edge_index_2 <= edge_index_1 or edge_gt_2 < 0.5:
                        continue
                    if (np.expand_dims(self.edge_corner[edge_index_2], axis=-1) == self.edge_corner[edge_index_1]).max() < 0.5:
                        continue
                    corner_count = np.zeros(len(self.edges_gt))
                    np.add.at(corner_count, self.edge_corner[edge_index_1], 1)
                    np.add.at(corner_count, self.edge_corner[edge_index_2], 1)
                    corner_pair = (corner_count == 1).nonzero()[0]
                    edge_index_mask = (self.edge_corner == corner_pair).all(-1) | (self.edge_corner == np.stack([corner_pair[1], corner_pair[0]], axis=0)).all(-1)
                    other_edge_index = edge_index_mask.nonzero()[0]
                    if len(other_edge_index) == 0:
                        continue
                    other_edge_index = other_edge_index[0]
                    if self.edges_gt[other_edge_index] > 0.5:
                        continue
                    if np.abs(np.dot(directions[edge_index_1], directions[edge_index_2])) > dot_product_threshold:
                        self.edges_gt[other_edge_index] = 1
                        #print(self.edge_corner[edge_index_1], self.edge_corner[edge_index_2])
                        has_change = True
                        pass
                    continue
                continue
            if not has_change:
                break
            continue
        return
