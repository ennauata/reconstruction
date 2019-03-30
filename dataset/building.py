import numpy as np
import os
from skimage.draw import line_aa
from PIL import Image, ImageDraw
import torch
import itertools
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from utils.utils import draw_edges, draw_edge, findLoopsModuleCPU
import cv2

class Building():
    """Maintain a building to create data examples in the same format"""

    def __init__(self, options, _id, with_augmentation=True, corner_type='annots_only'):
        self.options = options
        self.with_augmentation = with_augmentation

        PREFIX = options.data_path
        # LADATA_FOLDER = '{}/building_reconstruction/la_dataset_new/'.format(PREFIX)
        # ANNOTS_FOLDER = '{}/building_reconstruction/la_dataset_new/annots'.format(PREFIX)
        # EDGES_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_{}/edges'.format(PREFIX, corner_type)
        # CORNERS_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_{}/corners'.format(PREFIX, corner_type)

        LADATA_FOLDER = '{}/'.format(PREFIX)
        ANNOTS_FOLDER = '{}/annots'.format(PREFIX)
        EDGES_FOLDER = '{}/{}/edges'.format(PREFIX, corner_type)
        CORNERS_FOLDER = '{}/{}/corners'.format(PREFIX, corner_type)
        
        self.annots_folder = ANNOTS_FOLDER
        self.edges_folder = EDGES_FOLDER
        self.corners_folder = CORNERS_FOLDER
        self.dataset_folder = LADATA_FOLDER
        self._id = _id
        
        # annots
        annot_path = os.path.join(self.annots_folder, _id +'.npy')
        annot = np.load(open(annot_path, 'rb'), encoding='bytes')
        graph = dict(annot[()])
        
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
        #e_xys = self.compute_edges_map(edges_det_from_corners)
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
        self.rgb = self.read_input_images(_id, self.dataset_folder)

        self.edge_corner_annots = edge_corner_annots
        self.corners_gt = corners_gt                
        self.corners_det = np.round(corners_det[:, :2]).astype(np.int32)
        self.edges_gt = edges_gt
        self.edges_det = np.round(edges_det_from_corners).astype(np.int32)
        #self.e_xys = e_xys
        self.ce_angles_bins = ce_angles_bins
        self.corner_edge = corner_edge
        self.corner_edge_pairs = corner_edge_pairs
        self.edge_corner = edge_corner
        self.generate_bins = False
        self.num_edges = len(self.edges_gt)
        self.num_edges_gt = self.edges_gt.sum()
        self.corners_annot = corners_annot
        self.edges_annot = edges_annot

        self.add_colinear_edges(self.edges_gt) # CHECK THIS
        #self.add_colinear_edges_annot() # CHECK THIS
        
        if options.suffix != '':
            suffix = '_' + corner_type + '_' + options.suffix
        else:
            suffix = '_' + corner_type
            pass
        self.prediction_path = self.options.test_dir + '/cache/' + self._id + '.npy'
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
            imgs, corners_det, edges_det = self.augment(self.rgb.copy(), self.corners_det, self.edges_det)
        else:
            imgs, corners_det, edges_det = self.rgb, self.corners_det, self.edges_det
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
            imgs, corners_det, edges_det = self.augment(self.rgb.copy(), self.corners_det, self.edges_det)
        else:
            imgs, corners_det, edges_det = self.rgb, self.corners_det, self.edges_det

        # import matplotlib.pyplot as plt
        # debug_im = Image.fromarray(imgs.copy())
        # draw = ImageDraw.Draw(debug_im)
        # # for c in corners_det:
        # #     y, x = c
        # #     print(y, x)
        # #     draw.ellipse((x-2, y-2, x+2, y+2), fill='red')

        # for k, e in enumerate(edges_det):
        #     if self.edges_gt[k] == 1:
        #         y1, x1, y2, x2 = e
        #         draw.line((x1, y1, x2, y2), width=3, fill='red')

        # plt.figure()
        # plt.imshow(debug_im)
        # plt.figure()
        # plt.imshow(imgs)
        # plt.show()

        imgs = imgs.transpose((2, 0, 1)).astype(np.float32) / 255
        if load_heatmaps:
            img_c, corner_masks = self.compute_corner_image(corners_det)
            imgs = np.concatenate([imgs, np.array(img_c)[np.newaxis, :, :]], axis=0)

            edge_images = []
            for edge_index in range(len(edges_det)):
                edge_images.append(draw_edge(edge_index, edges_det))
                continue
            edge_images = np.stack(edge_images, axis=0)

            edge_mask = 255 - (draw_edges(np.ones(len(self.edges_annot), dtype=bool), np.round(self.edges_annot).astype(np.int32)) * 255).astype(np.uint8)
            #edge_mask = cv2.threshold(edge_mask, 127, 255, cv2.THRESH_BINARY)[1]            
            #cv2.imwrite('test/mask.png', edge_mask)            
            ret, masks = cv2.connectedComponents(edge_mask)
            #print(ret)
            #print(edge_mask.shape, masks.shape, masks.min(), masks.max())
            if False:
                loop_masks = []
                for mask_index in range(1, masks.max() + 1):
                    mask = masks == mask_index
                    if mask[0][0] and mask[-1][0] and mask[0][-1] and mask[-1][-1]:
                        continue
                    loop_masks.append(cv2.resize(mask.astype(np.uint8), (64, 64)))
                    #print(mask.shape, mask.min(), mask.max())
                    #cv2.imwrite('test/mask_' + str(mask_index) + '.png', mask.astype(np.uint8) * 255)
                    continue
                loop_masks = np.stack(loop_masks, axis=0)
            else:
                loop_masks = np.array([1])
                pass
            #exit(1)
            #return [imgs.astype(np.float32), edge_images.astype(np.float32), corners_det.astype(np.float32) / 256, edges_det.astype(np.float32) / 256, self.corners_gt.astype(np.float32), self.edges_gt.astype(np.float32), self.graph_edge_index, self.graph_edge_attr, self.left_edges.astype(np.int64), self.right_edges.astype(np.int64)]
            return [imgs.astype(np.float32), corner_masks.astype(np.float32), edge_images.astype(np.float32), loop_masks.astype(np.float32), corners_det.astype(np.float32) / 256, edges_det.astype(np.float32) / 256, self.corners_gt.astype(np.float32), self.edges_gt.astype(np.float32), self.corner_edge_pairs, self.edge_corner, self.left_edges.astype(np.int64), self.right_edges.astype(np.int64)]
        else:
            return [imgs.astype(np.float32), corners_det.astype(np.float32) / 256, edges_det.astype(np.float32) / 256, self.corners_gt.astype(np.float32), self.edges_gt.astype(np.float32), self.graph_edge_index, self.graph_edge_attr]

    def create_sample_edge(self, edge_index, load_heatmaps=False):
        """Create one data example:
        edge_index: edge index to flip, -1 for random sampling
        num_edges_source: source graph size, -1 for using the latest
        """
        #assert(num_edges_source < len(self.predicted_edges))

        if self.with_augmentation:
            imgs, corners_det, edges_det = self.augment(self.rgb.copy(), self.corners_det, self.edges_det)
        else:
            imgs, corners_det, edges_det = self.rgb, self.corners_det, self.edges_det
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

    def read_input_images(self, _id, path):

        im = np.array(Image.open("{}/rgb/{}.jpg".format(path, _id)))#.resize((128, 128))
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

    def rotate_flip_edge(self, edges_coords, rot, flip):
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

    def rotate_flip_corner(self, corner_coords, rot, flip):
        new_coords = []
        for c in corner_coords:
            y1, x1  = c
            x1, y1  = self.rotate_coords(np.array([256, 256]), np.array([x1, y1]), rot)
            if flip:
                x1, y1 = (128-abs(128-x1), y1) if x1 > 128 else (128+abs(128-x1), y1)
            c_aug = (y1, x1)
            new_coords.append(c_aug)
        return new_coords

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
            # edges_map = Image.new("RGB", (grid_size, grid_size), (0, 0, 0)) #Image.fromarray(np.zeros((grid_size, grid_size)))
            # draw_line(edges_map, [x1, y1], [x2, y2], color=(255, 255, 255))

            edges_map = np.zeros((grid_size, grid_size), dtype=np.uint8)
            # draw = ImageDraw.Draw(edges_map)
            # draw.line((x1, y1, x2, y2), width=1, fill='white')
            # edges_map = np.array(edges_map)
            # inds = np.array(np.where(edges_map > 0))
            rr, cc, val = line_aa(y1, x1, y2, x2)
            # edges_map = Image.fromarray(edges_map)
            # plt.imshow(edges_map)
            # plt.show()
            if rr.shape[0] > mlen:
                mlen = rr.shape[0]
            xys_unpadded.append(np.array(list(zip(rr, cc, val))))

        xys = []
        for inds in xys_unpadded:
            padded_inds = np.pad(inds, ((0, mlen-inds.shape[0]), (0, 0)), 'constant', constant_values=-1)
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

        # augmentation
        flip = np.random.choice([False, True])
        rot = np.random.choice([0, 90, 180, 270])

        # rotate and flip
        edges = self.rotate_flip_edge(edges, rot, flip)
        corners = self.rotate_flip_corner(corners, rot, flip)

        edges = np.array(edges).astype(np.int32)
        corners = np.array(corners).astype(np.int32)
        
        rgb = Image.fromarray(imgs).rotate(rot)
        if flip:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        rgb = np.array(rgb)

        return rgb, corners, edges 

    # def augment(self, imgs, corners, edges):
    #     size = imgs.shape[1]
    #     #center = vertices[0] + np.random.random(2) * (vertices[-1] - vertices[0])
    #     if self.with_augmentation:
    #         angle = np.random.random() * 360
    #     else:
    #         angle = 0
    #         pass
    #     #rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    #     #print('vertices', tuple(((vertices[0] + vertices[-1]) / 2).tolist()))
    #     #rotation_matrix = cv2.getRotationMatrix2D(tuple(((vertices[0] + vertices[-1]) / 2).tolist()), angle, 1)
        
    #     # ys, xs = np.nonzero(imgs.min(-1) < 250)
    #     # vertices = np.array([[xs.min(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.min()], [xs.max(), ys.max()]])        
    #     # print(vertices)
    #     xs, ys = corners[:, 1], corners[:, 0]
    #     vertices = np.array([[xs.min(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.min()], [xs.max(), ys.max()]])

    #     # print(vertices)
    #     # exit(1)
    #     rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1)
    #     transformed_vertices = np.matmul(rotation_matrix, np.concatenate([vertices, np.ones((len(vertices), 1))], axis=-1).transpose()).transpose()

    #     center = transformed_vertices.mean(0)
    #     transformed_vertices = center + (transformed_vertices - center) * 1.1
        
    #     mins = transformed_vertices.min(0)
    #     maxs = transformed_vertices.max(0)
    #     max_range = (maxs - mins).max()
    #     if self.with_augmentation:
    #         new_size = min(max_range, size) + max(size - max_range, 0) * np.random.random()
    #         #new_size = size
    #     else:
    #         new_size = max_range
    #         #new_size = size
    #         pass
    #     scale = float(new_size) / max_range
    #     if self.with_augmentation:
    #         offset = (np.random.random(2) * 2 - 1) * (size - (maxs - mins) * scale) / 2 + (size / 2 - (maxs + mins) / 2 * scale)
    #     else:
    #         offset = (size / 2 - (maxs + mins) / 2 * scale)
    #         #offset = np.zeros(2)
    #         pass
    #     #offset = 0 * (size - (maxs - mins) * scale) + (size / 2 - (maxs + mins) / 2 * scale)
    #     translation_matrix = np.array([[scale, 0, offset[0]], [0, scale, offset[1]]])
    #     transformation_matrix = np.matmul(translation_matrix, np.concatenate([rotation_matrix, np.array([[0, 0, 1]])], axis=0))
    #     #transformation_matrix_shuffled = np.stack([transformation_matrix[1], transformation_matrix[0]], axis=0)
    #     # transformation_matrix_shuffled = transformation_matrix.copy()
    #     # transformation_matrix_shuffled[1, 2] = transformation_matrix[0, 2]
    #     # transformation_matrix_shuffled[0, 2] = transformation_matrix[1, 2]        
    #     imgs = cv2.warpAffine(imgs, transformation_matrix, (size, size), borderValue=(255, 255, 255))
    #     #print(corners)
    #     corners_ori = corners
    #     corners = np.matmul(transformation_matrix, np.concatenate([corners[:, [1, 0]], np.ones((len(corners), 1))], axis=-1).transpose()).transpose()
    #     #print(scale, offset, size, mins, maxs, corners.min(), corners.max())        
        
    #     if (corners.min() < 0 or corners.max() > 256) and False:
    #         cv2.imwrite('test/image.png', imgs.astype(np.uint8))
    #         print(vertices)
    #         print(corners_ori)
    #         print(np.matmul(rotation_matrix, np.concatenate([corners_ori[:, [1, 0]], np.ones((len(corners), 1))], axis=-1).transpose()).transpose() * scale)
    #         print(transformed_vertices * scale)
    #         print(scale, size, new_size, mins, maxs, (maxs - mins) * scale, (maxs + mins) / 2 * scale, offset, corners.min(0), corners.max(0))
    #         print((size - (maxs - mins) * scale) / 2, size / 2 - (maxs + mins) / 2 * scale, (maxs + mins) / 2 * scale)
    #         exit(1)
    #         pass
    #     corners = corners[:, [1, 0]]            
    #     corners = np.clip(np.round(corners).astype(np.int32), 0, size - 1)
    #     edge_points = edges.reshape((-1, 2))[:, [1, 0]]
    #     edge_points = np.matmul(transformation_matrix, np.concatenate([edge_points, np.ones((len(edge_points), 1))], axis=-1).transpose()).transpose()
    #     edges = edge_points[:, [1, 0]].reshape((-1, 4))
    #     edges = np.clip(np.round(edges).astype(np.int32), 0, size - 1)
    #     #print('rotation', rotation_matrix, translation_matrix)
    #     #print(corners)
    #     return imgs, corners, edges


    def visualize(self, mode='last_mistake', edge_state=None, building_idx=None, post_processing=False, color=[255, 0, 255], debug=False):
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
            edge_image_annot[edge_mask > 0.5] = np.array([255, 0, 255], dtype=np.uint8)
            images.append(edge_image_annot)

        return images, np.array([(np.logical_and(self.predicted_edges[-1] == self.edges_gt, self.edges_gt == 1)).sum(), self.predicted_edges[-1].sum(), self.edges_gt.sum(), int(np.all(self.predicted_edges[-1] == self.edges_gt))])


    def visualize_top_k_loops(self, edge_pred, loop_pred, loop_edge_mask, topk=15):

        # sort loops
        loop_pred = loop_pred.detach().cpu().numpy()
        loop_edge_mask = loop_edge_mask.detach().cpu().numpy()
        inds = np.argsort(loop_pred)[::-1]

        loop_pred = loop_pred[inds]
        loop_edge_mask = loop_edge_mask[inds]

        # draw edge preds
        im_edge = self.rgb.copy()   
        edge_pred = edge_pred.detach().cpu().numpy()
        edge_mask = draw_edges(edge_pred, self.edges_det)
        im_edge[edge_mask > 0.5] = np.array([255, 0, 0], dtype=np.uint8)

        # draw loops
        top_k_loops = []
        for k in range(topk):
            if k < loop_edge_mask.shape[0]:  
                im_loop = self.rgb.copy() 
                edge_mask = draw_edges(loop_edge_mask[k], self.edges_det)
                im_loop[edge_mask > 0.5] = np.array([255, 255, 0], dtype=np.uint8)
                cv2.putText(im_loop, "{:.2f}%".format(loop_pred[k]*100.0), (20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255))
            else:
                im_loop = np.zeros_like(self.rgb.copy())

            top_k_loops.append(im_loop)
        return [im_edge] + top_k_loops

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
        image = self.imgs.copy()        
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
        dot_threshold = np.cos(np.deg2rad(20))
        for corner_index in range(len(self.corner_edge)):
            corner = self.corners_det[corner_index]
            edge_indices = self.corner_edge[corner_index].nonzero()[0].tolist()
            for edge_index_1, edge_index_2 in itertools.combinations(edge_indices, 2):
                direction_1 = self.edges_det[edge_index_1].reshape((2, 2)).mean(0) - corner
                direction_2 = self.edges_det[edge_index_2].reshape((2, 2)).mean(0) - corner
                if np.dot(direction_1, direction_2) / (np.linalg.norm(direction_1) * np.linalg.norm(direction_2)) > dot_threshold:
                    colinear_edges.append([edge_index_1, edge_index_2])
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

    def create_loops_sample_edge_features_im(self, phase=None, scale=2.0, res=128):

        with np.load('{}/{}_0_False.npz'.format(self.options.predicted_edges_path, self._id)) as data:
            edges_confidence = data['arr_0']
            
        all_loops, loop_labels, edges_loops, loop_acc = self.find_loops(edges_confidence, self.edge_corner, self.corners_det, self.edges_gt, self.corners_det.shape[0])
        
        if phase == "train":
            loop_labels, to_keep = self.filter_loops(edges_loops, loop_labels)
            all_loops = [x for k, x in enumerate(all_loops) if k in to_keep]
            edges_loops = edges_loops[to_keep]

        loop_feats = []
        loops_edges_ims = []
        for k, loop in enumerate(all_loops):
            rot = 0
            flip = False
            if phase == "train":
                rot = np.random.choice([0, 90, 180, 270])
                flip = np.random.choice([False, True])
            with np.load('{}/{}_{}_{}.npz'.format(self.options.predicted_edges_path, self._id, rot, flip)) as data:
                edge_feats = data['arr_1']

            # compute bins
            edges_coords_aug = self.rotate_flip(self.edges_det, rot, flip)
            one_hot, angles = self.compute_angles(edges_coords_aug)


            inds = np.where(edges_loops[k]==1)[0]
            e_angles = one_hot[inds, :]

            y1, x1, y2, x2 = np.split(np.array(edges_coords_aug)[inds, :]/255.0, 4, -1)
            alg_feats = np.concatenate([y1, x1, y2, x2, y1**2, x1**2, y2**2, x2**2, y1*x1, y1*x2, y2*x1, y2*x2], -1)
            loop_feat = np.concatenate([edge_feats[inds, :], alg_feats, e_angles], -1)
            drop = np.random.choice([True, False])
            if (phase == "train") and (drop==True): 
                loop_feat[:, :128] = 0.0

            rot_rgb_im = Image.fromarray(self.rgb.copy()).resize((128, 128)).rotate(rot)
            if flip:
                rot_rgb_im = rot_rgb_im.transpose(Image.FLIP_LEFT_RIGHT)
            rot_rgb_im = np.transpose(np.array(rot_rgb_im), (2, 0, 1))

            loop_edges = []
            for e_i in inds:
                y1, x1, y2, x2 = np.array(edges_coords_aug)[e_i]/float(scale)
                im = Image.new("L", (res, res))
                draw = ImageDraw.Draw(im)
                draw.line((x1, y1, x2, y2), width=2, fill='white')
                e_im = np.concatenate([rot_rgb_im, np.array(im)[np.newaxis, :, :]], 0)
                loop_edges.append(np.array(e_im))
                # import matplotlib.pyplot as plt
                # print(angles[e_i])
                # plt.imshow(im)
                # plt.show()
            loop_edges = np.array(loop_edges)

            loop_feat = np.pad(loop_feat, ((0, 20-loop_feat.shape[0]), (0, 0)), 'constant', constant_values=0)
            loop_edges = np.pad(loop_edges, ((0, 20-loop_edges.shape[0]), (0, 0), (0, 0), (0, 0)), 'constant', constant_values=0)

            loop_feats.append(loop_feat)
            loops_edges_ims.append(loop_edges)

        loop_feats = np.stack(loop_feats)
        loops_edges_ims = np.stack(loops_edges_ims)

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
            loops_edges_ims = np.concatenate([loops_edges_ims[pos_inds, :, :], loops_edges_ims[neg_inds, :, :]])

        # print("neg", np.where(loop_labels==0)[0].shape)
        # print("pos", np.where(loop_labels==1)[0].shape)
        return loop_feats, loop_labels, edges_loops, all_loops, loop_acc, loops_edges_ims

    def create_loops_sample_edge_features(self, phase=None):

        with np.load('{}/{}_0_False.npz'.format(self.options.predicted_edges_path, self._id)) as data:
            edges_confidence = data['arr_0']
            
        all_loops, loop_labels, edges_loops, loop_acc, loops_e_inds, loops_corners = self.find_loops(edges_confidence, self.edge_corner, self.corners_det, self.edges_gt, self.corners_det.shape[0])
        
        if phase == "train":
            loop_labels, to_keep = self.filter_loops(edges_loops, loop_labels)
            all_loops = [x for k, x in enumerate(all_loops) if k in to_keep]
            edges_loops = edges_loops[to_keep]

        loop_feats = []
        loops_edges_ims = []
        for k, loop in enumerate(all_loops):
            rot = 0
            flip = False
            if phase == "train":
                rot = np.random.choice([0, 90, 180, 270])
                flip = np.random.choice([False, True])
            with np.load('{}/{}_{}_{}.npz'.format(self.options.predicted_edges_path, self._id, rot, flip)) as data:
                edge_feats = data['arr_1']

            # compute bins
            edges_coords_aug = self.rotate_flip_edge(self.edges_det, rot, flip)
            corner_coords_aug = self.rotate_flip_corner(self.corners_det, rot, flip)
            one_hot, angles = self.compute_angles(edges_coords_aug)

            # e_inds = np.where(edges_loops[k]==1)[0]
            inds = loops_e_inds[k]
            e_angles = angles[inds]

            loop_corners = loops_corners[k] 
            loop_coords = np.array([np.concatenate([corner_coords_aug[c1], corner_coords_aug[c2]]) for c1, c2 in loop_corners])
            y1, x1, y2, x2 = np.split(loop_coords/255.0, 4, -1)

            # include alg features
            alg_feats = np.concatenate([y1, x1, y2, x2, y1**2, x1**2, y2**2, x2**2, y1*x1, y1*x2, y2*x1, y2*x2], -1)
            alg_feats = np.concatenate([alg_feats, e_angles[:, np.newaxis]], -1)

            #loop_feat = edge_feats[inds, :]
            drop = np.random.choice([True, False])
            if (drop == True) and (phase == "train"):
                edge_feats = np.zeros_like(edge_feats)
            loop_feat = np.concatenate([edge_feats[inds, :], alg_feats], -1)

            #loop_feat = np.pad(loop_feat, ((0, 20-loop_feat.shape[0]), (0, 0)), 'constant', constant_values=0)
            loop_feats.append(loop_feat)
            #loop_feats = np.stack(loop_feats)

            # # DEBUG
            # import matplotlib.pyplot as plt
            # rgb_im = Image.fromarray(self.rgb.copy())
            # rgb_im = rgb_im.rotate(rot)
            # if flip == True:
            #     rgb_im = rgb_im.transpose(Image.FLIP_LEFT_RIGHT)

            # print("LOOP")
            # print(loop_labels[k])
            # draw = ImageDraw.Draw(rgb_im)
            # for l in range(np.array(corner_coords_aug).shape[0]):
            #     y, x = corner_coords_aug[l]
            #     draw.ellipse((x-2, y-2, x+2, y+2), fill='red')

            # im = Image.new("L", (256, 256))
            # draw = ImageDraw.Draw(im)
            # for l in range(loop_feat.shape[0]):
            #     y1, x1, y2, x2 = loop_feat[l, -13]*255.0, loop_feat[l, -12]*255.0, loop_feat[l, -11]*255.0, loop_feat[l, -10]*255.0 
            #     angle = loop_feat[l, -1]
            #     print(y1, x1, y2, x2)
            #     print(angle)
            #     draw.line((x1, y1, x2, y2), width=3, fill='white')
            #     plt.figure()
            #     plt.imshow(im)
            #     plt.figure()
            #     plt.imshow(rgb_im)
            #     plt.show()

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

            loop_feats = [loop_feats[x] for x in pos_inds] + [loop_feats[x] for x in neg_inds]
            loop_labels = np.concatenate([loop_labels[pos_inds], loop_labels[neg_inds]])

        # print("neg", np.where(loop_labels==0)[0].shape)
        # print("pos", np.where(loop_labels==1)[0].shape)
        return loop_feats, loop_labels, edges_loops, all_loops, loop_acc, np.zeros_like(loop_labels)

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

    def create_loops_sample_feats_imgs(self, phase=None, n_bins=36):

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
            if phase == "train":
                rot = np.random.choice([0, 90, 180, 270])
                flip = np.random.choice([False, True])

            with np.load('{}/{}_{}_{}.npz'.format(self.options.predicted_edges_path, self._id, rot, flip)) as data:
                edge_feats = data['arr_1']
                edge_confs = data['arr_0']

            e_inds = np.where(edges_loops[k]==1)[0]

            # compute bins
            edges_coords_aug = self.rotate_flip(self.edges_det, rot, flip)
            one_hot, angles = self.compute_angles(edges_coords_aug)

            # place features in grid
            rot_rgb = Image.fromarray(self.rgb.copy()).resize((128, 128)).rotate(rot)
            if flip == True:
                rot_rgb = rot_rgb.transpose(Image.FLIP_LEFT_RIGHT)
            rot_rgb = np.array(rot_rgb)/255.0
            rot_rgb = np.transpose(rot_rgb, (2, 0, 1))

            e_ims = []
            for e in e_inds:
                e_im = Image.new("L", (128, 128))
                draw = ImageDraw.Draw(e_im)
                y1, x1, y2, x2 = np.array(edges_coords_aug[e])/2.0
                draw.line((x1, y1, x2, y2), width=2, fill='white')
                e_im = np.array(e_im)[np.newaxis, :, :]
                im_feat = np.concatenate([e_im, rot_rgb], 0)
                e_ims.append(im_feat)
                # import matplotlib.pyplot as plt
                # plt.imshow(e_im)
                # plt.show()
            e_ims = np.stack(e_ims)
            # debug_arr = np.sum(grid, 0)
            # inds = np.where(debug_arr!=0)
            # debug_arr[inds] = 255.0
            # print(loop_labels[k])
            # debug_im = Image.fromarray(debug_arr)
            # debug_im = Image.fromarray((rot_im[0, :, :]*255).astype("int32"))
            # import matplotlib.pyplot as plt
            # plt.imshow(debug_im)
            # plt.show()

            loop_feats.append(e_ims)
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

        return loop_feats, loop_labels, edges_loops, all_loops, loop_acc, np.zeros_like(edges_loops)

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
            grid = np.zeros((129, 128, 128))

            #im_loop = draw_edges(edges_loops[k], np.array(edges_coords_aug).astype(np.int32))

            for e in e_inds:
                xs_inds = np.array(np.where(e_xys[e, 0, :]>=0)[0])
                ys_inds = np.array(np.where(e_xys[e, 1, :]>=0)[0])

                if xs_inds.shape[0] > 0:
                    xs = e_xys[e, 0, xs_inds]
                    ys = e_xys[e, 1, ys_inds]

                    feat_in = np.concatenate([edge_feats[e, :], [angles[e]/180.0]], 0)

                    feat_in = feat_in[:, np.newaxis]
                    feat_in = np.repeat(feat_in, xs_inds.shape[0], axis=1)

                    #feat_in = edge_confs[e]
                    grid[:, xs, ys] = feat_in

            # print(angles[e_inds])
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(np.array(im_loop) * 255)
            # plt.show()

            # print(angles[e_inds])
            # debug_arr = np.sum(grid, 0)
            # inds = np.where(debug_arr!=0)
            # debug_arr[inds] = 255.0
            # print(loop_labels[k])
            # debug_im = Image.fromarray(debug_arr)
            # debug_im = Image.fromarray((rot_im[0, :, :]*255).astype("int32"))
            # import matplotlib.pyplot as plt
            # plt.imshow(debug_im)
            # plt.show()

            # drop edge features
            drop = np.random.choice([True, False])
            if (drop == True) and (phase == "train"):
                grid[:128, :, :] = 0.0

            drop = np.random.choice([True, False])
            if (drop == True) and (phase == "train"):
                rot_rgb[:, :, :] = 0.0

            grid = np.concatenate([grid, rot_rgb], 0)

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

        return loop_feats, loop_labels, edges_loops, all_loops, loop_acc, np.zeros_like(edges_loops)

    def create_loops_sample(self, phase=None):

        with np.load('{}/{}_0_False.npz'.format(self.options.predicted_edges_path, self._id)) as data:
            edges_confidence = data['arr_0']
            
        all_loops, loop_labels, edges_loops, loop_acc = self.find_loops(edges_confidence, self.edge_corner, self.corners_det, self.edges_gt, self.corners_det.shape[0])
        if phase == "train":
            #print("WARNING: FILTERING LOOPS!")
            loop_labels, to_keep = self.filter_loops(edges_loops, loop_labels)
            all_loops = [x for k, x in enumerate(all_loops) if k in to_keep]
            edges_loops = edges_loops[to_keep]

        e_xys_dict = {}
        for rot in [0, 90, 180, 270]:
            for flip in [False, True]:
                e_xys_dict["{}_{}".format(rot, flip)] = self.compute_edges_map(self.edges_det, rot=rot, flip=flip)

        loop_ims = []
        rgb_imgs = []
        for k, loop in enumerate(all_loops):
            im = draw_edges(edges_loops[k], self.edges_det)

            # augmentation
            flip = np.random.choice([False, True])
            rot = np.random.choice([0, 90, 180, 270])
            rot_loop_im = Image.fromarray(im).resize((128, 128)).rotate(rot)
            rot_rgb_im = Image.fromarray(self.rgb.copy()).resize((128, 128)).rotate(rot)

            if flip:
                rot_loop_im = rot_loop_im.transpose(Image.FLIP_LEFT_RIGHT)
                rot_rgb_im = rot_rgb_im.transpose(Image.FLIP_LEFT_RIGHT)

            # compute bins
            edges_coords_aug = self.rotate_flip(self.edges_det, rot, flip)
            one_hot, angles = self.compute_angles(edges_coords_aug)
            e_xys = e_xys_dict["{}_{}".format(rot, flip)]

            e_inds = np.where(edges_loops[k]==1)[0]
            grid = np.zeros((1, 128, 128))
            for e in e_inds:
                xs_inds = np.array(np.where(e_xys[e, 0, :]>=0)[0])
                ys_inds = np.array(np.where(e_xys[e, 1, :]>=0)[0])

                if xs_inds.shape[0] > 0:
                    xs = e_xys[e, 0, xs_inds]
                    ys = e_xys[e, 1, ys_inds]
                    grid[:, xs, ys] = angles[e]/180.0

            rot_loop_im = np.array(rot_loop_im)
            rot_rgb_im = np.array(rot_rgb_im)

            # print(angles[e_inds])
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(np.array(rot_loop_im) * 255)

            # debug_arr = np.sum(grid, 0)
            # inds = np.where(debug_arr!=0)
            # debug_arr[inds] = 255.0
            # print(loop_labels[k])
            # debug_im = Image.fromarray(debug_arr)
            # debug_im_rgb = Image.fromarray((rot_rgb_im).astype("uint8"))
            # plt.figure()
            # plt.imshow(debug_im)
            # plt.figure()
            # plt.imshow(debug_im_rgb)
            # plt.show()

            im_in = np.concatenate([rot_loop_im[np.newaxis, :, :], grid], axis=0)
            rgb_imgs.append(np.transpose(rot_rgb_im/255.0, (2, 0, 1)))
            loop_ims.append(im_in)

        loop_ims = np.stack(loop_ims)
        rgb_imgs = np.stack(rgb_imgs)

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

            loop_ims = np.concatenate([loop_ims[pos_inds, :, :, :], loop_ims[neg_inds, :, :, :]])
            rgb_imgs = np.concatenate([rgb_imgs[pos_inds, :, :, :], rgb_imgs[neg_inds, :, :, :]])
            loop_labels = np.concatenate([loop_labels[pos_inds], loop_labels[neg_inds]])

        return loop_ims, loop_labels, edges_loops, all_loops, loop_acc, rgb_imgs

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
        all_loops = findLoopsModuleCPU(torch.from_numpy(current_edges.astype(np.float32)), torch.from_numpy(edge_corner), num_corners, max_num_loop_corners=15, confidence_threshold=0.5, corners=torch.from_numpy(self.corners_det).float(), disable_colinear=False)        

        loop_corners = []
        loop_labels = []
        edges_loops = []
        loop_acc = []
        loops_e_inds = []
        loops_coords = []
        for accs, loops in all_loops:

            loops = loops.detach().cpu().numpy()
            for loop in loops:
                #print(loop)
                loop_corners.append(corners[loop])

            # convert loop to edges
            for acc, loop in zip(accs, loops):
                label = 1
                edges_loop = np.zeros_like(current_edges)
                c1 = loop[0]
                #print(c1)
                # print(len(list(loop[1:])+[loop[0]]))
                e_inds_ordered = []
                corners_ordered = []
                for c2 in list(loop[1:])+[loop[0]]:
                    # check if is a true edge and get all edges in loop
                    is_true_edge = False
                    #print(c1, c2)
                    for k, e in enumerate(edge_corner):
                        if (np.array_equal(np.array([c1, c2]), e) == True) or (np.array_equal(np.array([c2, c1]), e) == True):
                            edges_loop[k] = 1
                            e_inds_ordered.append(k)
                            corners_ordered.append(np.array([c1, c2]))
                            # print(np.array([c1, c2]), e)
                            if edges_gt[k] == 1:
                                is_true_edge = True
                        elif c1 == c2:
                            is_true_edge = True

                    if is_true_edge == False:
                        label = 0
                    c1 = c2
                
                corners_ordered = np.stack(corners_ordered)
                loops_e_inds.append(e_inds_ordered)
                edges_loops.append(edges_loop)
                loop_labels.append(label)
                loop_acc.append(acc.detach().cpu().numpy())
                loops_coords.append(corners_ordered)

                # print(np.where(edges_loop>0))
                # loop_im = draw_edges(edges_loop, self.edges_det)
                # print("label: ", label)
                # cv2.imshow('loop', loop_im)
                # cv2.waitKey(0)

        edges_loops = np.stack(edges_loops)
        loop_acc = np.stack(loop_acc)
        loops_e_inds = np.array(loops_e_inds)

        return loop_corners, np.array(loop_labels), edges_loops, loop_acc, loops_e_inds, loops_coords

    def add_colinear_edges(self, edge_state):
        dot_product_threshold = np.cos(np.deg2rad(20))        
        directions = (self.edges_det[:, 2:4] - self.edges_det[:, :2]).astype(np.float32)
        directions = directions / np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-4)
        while True:
            has_change = False
            for edge_index_1, edge_gt_1 in enumerate(edge_state):
                if edge_gt_1 < 0.5:
                    continue
                for edge_index_2, edge_gt_2 in enumerate(edge_state):
                    if edge_index_2 <= edge_index_1 or edge_gt_2 < 0.5:
                        continue
                    if (np.expand_dims(self.edge_corner[edge_index_2], axis=-1) == self.edge_corner[edge_index_1]).max() < 0.5:
                        continue
                    corner_count = np.zeros(len(edge_state))
                    np.add.at(corner_count, self.edge_corner[edge_index_1], 1)
                    np.add.at(corner_count, self.edge_corner[edge_index_2], 1)
                    corner_pair = (corner_count == 1).nonzero()[0]
                    edge_index_mask = (self.edge_corner == corner_pair).all(-1) | (self.edge_corner == np.stack([corner_pair[1], corner_pair[0]], axis=0)).all(-1)
                    other_edge_index = edge_index_mask.nonzero()[0]
                    if len(other_edge_index) == 0:
                        continue
                    other_edge_index = other_edge_index[0]
                    if edge_state[other_edge_index] > 0.5:
                        continue
                    if np.abs(np.dot(directions[edge_index_1], directions[edge_index_2])) > dot_product_threshold:
                        edge_state[other_edge_index] = 1
                        #print(self.edge_corner[edge_index_1], self.edge_corner[edge_index_2])
                        has_change = True
                        pass
                    continue
                continue
            if not has_change:
                break
            continue
        return

    def add_colinear_edges_annot(self):
        dot_product_threshold = np.cos(np.deg2rad(20))
        edge_corner = self.edge_corner_annots
        edges = np.array(self.corners_annot)[:, :2][edge_corner].reshape((-1, 4))
        num_edges = len(edges)
        directions = (edges[:, 2:4] - edges[:, :2]).astype(np.float32)
        directions = directions / np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-4)
        new_edge_corner = []
        while True:
            has_change = False
            for edge_index_1 in range(num_edges):
                for edge_index_2 in range(num_edges):
                    if edge_index_2 <= edge_index_1:
                        continue
                    if (np.expand_dims(edge_corner[edge_index_2], axis=-1) == edge_corner[edge_index_1]).max() < 0.5:
                        continue
                    if np.abs(np.dot(directions[edge_index_1], directions[edge_index_2])) < dot_product_threshold:
                        continue
                    corner_count = np.zeros(num_edges)
                    np.add.at(corner_count, edge_corner[edge_index_1], 1)
                    np.add.at(corner_count, edge_corner[edge_index_2], 1)
                    corner_pair = (corner_count == 1).nonzero()[0]
                    corner_pair = np.sort(corner_pair)
                    new_edge_corner.append(corner_pair)
                    continue
                continue
            if not has_change:
                break
            continue
        if len(new_edge_corner) > 0:
            self.edge_corner_annots = np.concatenate([self.edge_corner_annots, np.stack(new_edge_corner, axis=0)], axis=0)
            pass
        return    

    def post_process(self, edge_state):
        self.add_colinear_edges(edge_state)
        dot_product_threshold = np.cos(np.deg2rad(20))    
        distance_threshold = 0.01

        edge_corner = self.edge_corner[edge_state]
        corners = self.corners_det.astype(np.float32) / 256
        
        num_edges = len(edge_corner)
        
        edges = corners[edge_corner]
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
        tangent_distance = ((np.expand_dims(corners, 1) - edges[:, 0]) * edge_directions).sum(-1)
        #independent_mask = np.ones((len(corners), num_edges))
        #independent_mask[edge_corner[:, 0], np.arange(len(edge_corner), dtype=np.int32)] = 0
        #independent_mask[edge_corner[:, 1], np.arange(len(edge_corner), dtype=np.int32)] = 0

        corner_edge_mask = (tangent_distance > 0) & (tangent_distance < 1) & (normal_distance < distance_threshold)
        parallel_mask = np.abs((np.expand_dims(edge_directions, 1) * edge_directions).sum(-1)) > dot_product_threshold
        colinear_mask = np.maximum(corner_edge_mask[edge_corner[:, 0]], corner_edge_mask[edge_corner[:, 1]]) & parallel_mask
        # print(corners)
        # print(edges)
        # print(corner_edge_mask)
        # print(edge_corner)
        # print(colinear_mask)
        # exit(1)
        edge_groups = []
        visited_mask = np.zeros(num_edges, dtype=np.bool)
        for edge_index in range(num_edges):
            if visited_mask[edge_index] > 0:
                continue
            edge_group = np.arange(num_edges, dtype=np.int32) == edge_index
            while True:
                new_edge_group = np.maximum(colinear_mask[edge_group].max(0), edge_group)
                if (new_edge_group == edge_group).all():
                    break
                edge_group = new_edge_group
                continue
            visited_mask = np.maximum(visited_mask, edge_group)
            edge_groups.append(edge_group.nonzero()[0])
            continue
        edge_lines = np.zeros((num_edges, 2))
        for edge_group in edge_groups:
            corner_indices = np.unique(edge_corner[edge_group].reshape(-1))
            edge_corners = corners[corner_indices]
            line = np.linalg.lstsq(edge_corners, np.ones(edge_corners.shape[0]), rcond=None)[0]
            edge_lines[edge_group] = line
            pass

        corner_edge_mask[edge_corner[:, 0], np.arange(len(edge_corner), dtype=np.int32)] = 1
        corner_edge_mask[edge_corner[:, 1], np.arange(len(edge_corner), dtype=np.int32)] = 1
        visited_corner_mask = np.zeros(len(corners), dtype=np.bool)
        #intersection_mask = np.maximum(corner_edge_mask[edge_corner[:, 0]], corner_edge_mask[edge_corner[:, 1]]) & np.logical_not(parallel_mask)
        for edge_index_1, corner_indices_1 in enumerate(edge_corner):
            line_1 = edge_lines[edge_index_1]
            for corner_index in corner_indices_1:
                for edge_index_2 in range(num_edges):        
                    if corner_edge_mask[corner_index, edge_index_2] and not parallel_mask[edge_index_1, edge_index_2]:
                        line_2 = edge_lines[edge_index_2]
                        corner = np.linalg.lstsq(np.stack([line_1, line_2], axis=0), np.ones(2), rcond=None)[0]
                        corners[corner_index] = corner
                        visited_corner_mask[corner_index] = True
                        pass
                    continue
                continue
            continue
        corner_mapping = np.arange(len(corners), dtype=np.int32)
        new_corner_index = 0
        for corner_index in range(len(corners)):
            if (edge_corner == corner_index).sum() >= 2:
                corner_mapping[corner_index] = new_corner_index                
                new_corner_index += 1
            else:
                corner_mapping[corner_index] = -1
                pass
            continue
        for corner_index_1, corner_1 in enumerate(corners):
            for corner_index_2, corner_2 in enumerate(corners):
                if corner_index_2 == corner_index_1:
                    break
                if ((edge_corner == np.array([corner_index_1, corner_index_2])).all(-1).any() or (edge_corner == np.array([corner_index_2, corner_index_1])).all(-1).any()):
                    continue
                if np.linalg.norm(corner_1 - corner_2) > distance_threshold:
                    continue
                corner_mapping[corner_index_1] = corner_mapping[corner_index_2]
                continue
            continue

        corner_indices, corner_mapping = np.unique(corner_mapping, return_inverse=True)
        new_corners = []
        new_corner_mapping = corner_mapping
        new_corner_index = 0
        for corner_index, ori_corner_index in enumerate(corner_indices):
            if ori_corner_index < 0:
                continue
            mask = corner_mapping == corner_index
            new_corners.append(corners[mask].mean(0))
            new_corner_mapping[mask] = new_corner_index
            new_corner_index += 1
            continue
        if len(new_corners) == 0:
            return
        corners = np.stack(new_corners, axis=0)
        corner_mapping = new_corner_mapping
        #corners = corners[corner_indices]
        edge_corner = corner_mapping[edge_corner]
        edge_values = edge_corner.max(-1) * len(corner_indices) + edge_corner.min(-1)
        _, edge_indices = np.unique(edge_values, return_index=True)
        edge_corner = edge_corner[edge_indices]
        edge_corner = edge_corner[edge_corner[:, 0] != edge_corner[:, 1]]
        # print(self.corners_det)
        # print(corners)
        # print(edge_corner)
        # exit(1)
        #print(self.edge_corner)
        self.edge_corner = edge_corner
        corners = (corners * 256).astype(np.int32)
        self.corners_det = corners
        self.edges_det = corners[edge_corner].reshape((-1, 4))
        return
