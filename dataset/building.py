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

class Building():
    """Maintain a building to create data examples in the same format"""

    def __init__(self, options, _id, with_augmentation=True, corner_type='dets_only'):
        self.options = options
        self.with_augmentation = with_augmentation

        PREFIX = options.data_path
        LADATA_FOLDER = '{}/building_reconstruction/la_dataset_new/'.format(PREFIX)
        ANNOTS_FOLDER = '{}/building_reconstruction/la_dataset_new/annots'.format(PREFIX)
        EDGES_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_{}/edges'.format(PREFIX, corner_type)
        CORNERS_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_{}/corners'.format(PREFIX, corner_type)
        
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

        inds = np.load(edges_embs_path.replace('edges_feats', 'filters'))[1, :]
        edges_gt = edges_gt.astype(np.int32)
        ce_t0 = np.zeros_like(edges_gt)
        #ce_t0[inds] = 1.0
        corner_edge = self.compute_dists(corners_det, edges_det_from_corners, ce_assignment)

        
        #ce_angles_bins, ca_gt = self.compute_corner_angle_bins(corner_edge, edges_det_from_corners, edges_gt, corners_det)
        ce_angles_bins = None

        # read images
        imgs = self.read_input_images(_id, self.dataset_folder)

        self.imgs = imgs
        self.corners_gt = corners_gt                
        self.corners_det = np.round(corners_det[:, :2]).astype(np.int32)
        self.edges_gt = edges_gt
        self.edges_det = np.round(edges_det_from_corners).astype(np.int32)
        #self.e_xys = e_xys
        self.ce_angles_bins = ce_angles_bins
        self.corner_edge = corner_edge
        self.generate_bins = False
        self.num_edges = len(self.edges_gt)
        self.num_edges_gt = self.edges_gt.sum()
        self.corners_annot = corners_annot
        self.edges_annot = edges_annot

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
        
        img_c = self.compute_corner_image(corners_det)
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
            img_c = self.compute_corner_image(corners_det)
            imgs = np.concatenate([imgs, np.array(img_c)[np.newaxis, :, :]], axis=0)

            edge_images = []
            for edge_index in range(len(edges_det)):
                edge_images.append(draw_edge(edge_index, edges_det))
                continue
            edge_images = np.stack(edge_images, axis=0)
            return [imgs.astype(np.float32), edge_images.astype(np.float32), corners_det.astype(np.float32) / 256, edges_det.astype(np.float32) / 256, self.corners_gt.astype(np.float32), self.edges_gt.astype(np.float32), self.graph_edge_index, self.graph_edge_attr, self.left_edges.astype(np.int64), self.right_edges.astype(np.int64)]
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
        img_c = self.compute_corner_image(corners_det)
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
        im_c = np.zeros((256, 256))
        for c in corners:
            cv2.circle(im_c, (c[1], c[0]), color=1, radius=5, thickness=-1)
            # x, y, _, _ = np.array(c)
            # x, y = int(x), int(y)
            # im_c[x, y] = 1
        return im_c

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

    def compute_edges_map(self, edges_det, grid_size=256, scale=1.0):

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
        left_edges = [np.array([[edge_index, neighbor] for neighbor in neighbors[0]]) for neighbors in edge_neighbors]
        left_edges = np.concatenate(left_edges, axis=0)
        left_edges = left_edges[left_edges[:, 0] != left_edges[:, 1]]
        right_edges = [np.array([[edge_index, neighbor] for neighbor in neighbors[1]]) for neighbors in edge_neighbors]
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
        for i, c in enumerate(corners_det):
            for j in range(edges_det.shape[0]):
                c1 = c_e_assignment[j][0]
                c2 = c_e_assignment[j][1]
                if np.array_equal(c1, c) or np.array_equal(c2, c):
                    r_dist[i, j] = 1.0

        return r_dist

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
                corner_map_det_to_annot[j].append(i)
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

    def visualize(self, mode='last_mistake', edge_state=None, building_idx=None):
        image = self.imgs.copy()        
        corner_image = self.compute_corner_image(self.corners_det)
        image[corner_image > 0.5] = np.array([255, 0, 0], dtype=np.uint8)
        edge_image = image.copy()
        if 'last' in mode:
            edge_mask = draw_edges(self.predicted_edges[-1], self.edges_det)
        else:
            edge_mask = draw_edges(edge_state, self.edges_det)
            pass
        edge_image[edge_mask > 0.5] = np.array([255, 0, 255], dtype=np.uint8)
        images = [edge_image]
        if 'mistake' in mode:
            if (self.predicted_edges[-1] - self.edges_gt).max() > 0:
                for edges in self.predicted_edges:
                    if (edges - self.edges_gt).max() > 0:
                        edge_image = image.copy()
                        edge_mask = draw_edges(edges, self.edges_det)
                        edge_image[edge_mask > 0.5] = np.array([255, 0, 255], dtype=np.uint8)
                        images.append(edge_image)
                        break
                    continue
            else:            
            #if 'mistake' in mode and len(images) == 1:
                images.append(np.zeros(edge_image.shape, dtype=np.uint8))
                pass
            pass

        if 'draw_annot' in mode:    
            corner_annot = self.compute_corner_image(np.array(self.corners_annot).astype('int'))
            corner_image_annot = self.imgs.copy()
            corner_image_annot[corner_annot > 0.5] = np.array([255, 0, 0], dtype=np.uint8)
            images.append(corner_image_annot)

            edge_image_annot = corner_image_annot.copy()
            edge_mask = draw_edges(np.ones(self.edges_annot.shape[0]), np.array(self.edges_annot.astype('int')))
            edge_image_annot[edge_mask > 0.5] = np.array([255, 0, 255], dtype=np.uint8)
            images.append(edge_image_annot)

        return images, np.array([(np.logical_and(self.predicted_edges[-1] == self.edges_gt, self.edges_gt == 1)).sum(), self.predicted_edges[-1].sum(), self.edges_gt.sum(), int(np.all(self.predicted_edges[-1] == self.edges_gt))])

    def post_processing(self, edge_confidence):
        corner_edge = self.corner_edge > 0.5
        num_corners = len(corner_edge)
        num_edges = len(edge_confidence)
        edge_corner = {}
        for conrer_index, edge_index in np.nonzero(corner_edge):
            if edge_index not in edge_corner:
                edge_corner[edge_index] = []
                pass
            edge_corner[edge_index].append(corner_index)
            continue
        print(edge_corner.keys())
        edge_corner = np.array(edge_corner.values())
        print(edge_corner.shape)
        exit(1)
        
        edge_state = edge_confidence > 0.5

        ## Compute the shortest path between any pair of corners (not passing chosen edges)
        corner_pair_distances = np.full((num_corners, num_corners), fill_value=100)
        for edge_index in range(num_edges):
            if not edge_state[edge_index]:
                distance = 1 - edge_confidence[edge_index]
                corner_pair_distances[edge_corner[edge_index][0], edge_corner[edge_index][1]] = distance
                corner_pair_distances[edge_corner[edge_index][1], edge_corner[edge_index][0]] = distance
                pass
            continue
        corner_nearest_neighbors = []
        while True:
            corner_nearest_neighbor = corner_pair_distances.argmin(-1)
            new_corner_pair_distances = np.mimumum(corner_pair_distances, distances.min(-1, keepdims=True) + corner_pair_distances[corner_nearest_neighbor])
            corner_nearest_neighbors.append(corner_nearest_neighbor)            
            if np.all(new_corner_pair_distances == corner_pair_distances):
                break
            continue

        ## Compute the path with the maximum mean confidence between any pair of corners (passing chosen edges)
        corner_pair_distances = np.zeros((num_corners, num_corners))
        for edge_index in range(num_edges):
            if edge_state[edge_index]:
                distance = edge_confidence[edge_index]
                corner_pair_distances[edge_corner[edge_index][0], edge_corner[edge_index][1]] = distance
                corner_pair_distances[edge_corner[edge_index][1], edge_corner[edge_index][0]] = distance
                pass
            continue
        corner_furthest_neighbors = []
        while True:
            corner_furthest_neighbor = corner_distances.argmax(-1)
            corner_furthest_neighbors.append(corner_furthest_neighbor)                        
            iteration = len(corner_furthest_neighbors)
            
            new_corner_pair_distances = np.maximum(corner_pair_distances, (corner_pair_distances.max(-1, keepdims=True) * iteration + distances[corner_nearest_neighbor]) / (iteration + 1))
            if np.all(new_corner_pair_distances == corner_pair_distances):
                break
            continue
        
        corner_edge_state = edge_state[corner_edge]
        corner_degrees = corner_edge_state.sum(-1)
        #valid_corners = corner_degrees > 0
        valid_corner_mask = corner_degrees >= 2
        valid_edge_mask = np.logical_and(np.logical_and(valid_corner_mask[edge_corner[:, 0]], valid_corner_mask[edge_corner[:, 1]]), edge_state)
        
        
        singular_corners = (corner_degrees == 1).nonzero()
            
        for corner_index in singular_corners:
            neighbor_corners = corner_index

            
