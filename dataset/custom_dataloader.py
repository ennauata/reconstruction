import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pickle as p
import glob
import random
from collections import OrderedDict
from utils.intersections import doIntersect
from math import atan2,degrees

class GraphData(Dataset):

    def __init__(self, id_list, annots_folder, edges_folder, corners_folder, dataset_folder, with_augmentation=False, with_filter=True):
        self.corners_folder = corners_folder
        self.edges_folder = edges_folder
        self.annots_folder = annots_folder
        self._data_refs = id_list
        self.with_augmentation = with_augmentation
        self.with_filter = with_filter
        self.dataset_folder = dataset_folder

    def __getitem__(self, index):

        # retrieve id 
        _id = self._data_refs[index]

        # annots
        annot_path = os.path.join(self.annots_folder, _id +'.npy')
        annot = np.load(open(annot_path, 'rb'), encoding='bytes')
        graph = dict(annot[()])

        # augment data
        if self.with_augmentation:
            rot = random.choice([0, 90, 180, 270])
            flip = random.choice([True, False])
            use_gt = random.choice([False])
        else:
            rot = 0
            flip = False
            use_gt = False

        imgs = self.read_input_images(_id, rot, flip, self.dataset_folder)

        # load annots
        corners_annot, edges_annot = self.load_annot(graph, rot, flip)

        # retrieve corners and edges detections
        coners_path = os.path.join(self.corners_folder, "{}_{}_{}.npy".format(_id, rot, flip))
        edges_path = os.path.join(self.edges_folder, "{}_{}_{}.npy".format(_id, rot, flip))
        coners_embs_path = os.path.join(self.corners_folder.replace('corners', 'corners_feats'), "{}_{}_{}.npy".format(_id, rot, flip))
        edges_embs_path = os.path.join(self.edges_folder.replace('edges', 'edges_feats'), "{}_{}_{}.npy".format(_id, rot, flip))

        # load primitives and features
        corners_det = np.load(open(coners_path, 'rb'))
        corners_embs = np.load(open(coners_embs_path, 'rb'))
        edges_embs = np.load(open(edges_embs_path, 'rb'))

        # extract edges from corners
        edges_det_from_corners, ce_assignment = self.extract_edges_from_corners(corners_det, None)
        e_xys = self.compute_edges_map(edges_det_from_corners)

        # match compute true/false samples for detections
        corners_gt, edges_gt = self.compute_gt(corners_det, edges_det_from_corners, corners_annot, edges_annot, placeholder=False)
        inds = np.load(edges_embs_path.replace('edges_feats', 'filters'))[1, :]
        ce_t0 = np.zeros_like(edges_gt)
        ce_t0[inds] = 1.0
        corner_edge = self.compute_dists(corners_det, edges_det_from_corners, ce_assignment)

        # generate states
        samples, targets = self.generate_input_tuple(corners_gt, edges_gt, e_xys, corner_edge, ce_t0, imgs)

        # convert to tensor
        samples = torch.from_numpy(samples)
        targets = torch.from_numpy(targets)

        return samples, targets

    def read_input_images(self, _id, rot, flip, path):

        im = Image.open("{}/rgb/{}.jpg".format(path, _id)).resize((128, 128))
        # dp_im = Image.open(info['path'].replace('rgb', 'depth')).convert('L')
        # surf_im = Image.open(info['path'].replace('rgb', 'surf'))
        # gray_im = Image.open(info['path'].replace('rgb', 'gray')).convert('L')
        out_im = Image.open("{}/outlines/{}.jpg".format(path, _id)).convert('L').resize((128, 128))

        im = im.rotate(rot)
        # dp_im = dp_im.rotate(rot)
        # surf_im = surf_im.rotate(rot)
        # gray_im = gray_im.rotate(rot)
        out_im = out_im.rotate(rot)
        if flip == True:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            out_im = out_im.transpose(Image.FLIP_LEFT_RIGHT)

        # dp_im = dp_im.transpose(Image.FLIP_LEFT_RIGHT)
        # surf_im = surf_im.transpose(Image.FLIP_LEFT_RIGHT)
        # gray_im = gray_im.transpose(Image.FLIP_LEFT_RIGHT)
        # out_im = out_im.transpose(Image.FLIP_LEFT_RIGHT)

        return np.concatenate([np.array(im), np.array(out_im)[:, :, np.newaxis]], axis=-1)
        #return np.array(out_im)[:, :, np.newaxis]
        #return np.concatenate([np.array(im), np.array(dp_im)[:, :, np.newaxis], np.array(gray_im)[:, :, np.newaxis], np.array(surf_im)], axis=-1)
        #return np.array(im)

    def compute_edges_map(self, edges_det, grid_size=128, scale=2.0):

        xys_unpadded = []
        mlen = 0
        for e in edges_det:
            y1, x1, y2, x2 = e/scale
            edges_map = Image.fromarray(np.zeros((grid_size, grid_size)))
            draw = ImageDraw.Draw(edges_map)
            draw.line((x1, y1, x2, y2), width=1, fill='white')
            inds = np.array(np.where(np.array(edges_map) > 0))
            if inds.shape[1] > mlen:
                mlen = inds.shape[1]
            xys_unpadded.append(inds)

        xys = []
        for inds in xys_unpadded:
            padded_inds = np.pad(inds, ((0, 0), (0, mlen-inds.shape[1])), 'constant', constant_values=-1)
            xys.append(padded_inds)

        xys = np.stack(xys, 0)
        return xys

    def draw_edges(self, edges_on, e_xys):
        im = np.zeros((128, 128))
        for k in range(edges_on.shape[0]):
            if edges_on[k] == 1:
                xs, ys = e_xys[k]
                pos_inds = np.where(xs >= 0)[0]
                xs = xs[pos_inds]
                ys = ys[pos_inds]
                im[xs, ys] += 1.0
        return im

    # def generate_input_tuple(self, corners_gt, edges_gt, e_xys, corner_edge, current_edges, imgs):

    #     samples, targets = [], []
    #     current_edges = np.zeros_like(current_edges)
    #     for i in range(corner_edge.shape[0]):

    #         # draw current state
    #         im_s0 = self.draw_edges(current_edges, e_xys)

    #         # pick two edges
    #         inds =  np.where(corner_edge[i, :] == 1)
    #         inds_pos = np.where((corner_edge[i, :] == 1) & (edges_gt == 1))[0]
    #         n_pos = inds_pos.shape[0]
    #         inds_neg = np.where((corner_edge[i, :] == 1) & (edges_gt == 0))[0]
            
    #         # generate positive samples
    #         if n_pos > 0:
    #             inds_pos = np.random.choice(inds_pos, 1, replace=False)
    #             for j in inds_pos:

    #                 # draw new state
    #                 new_state = np.array(current_edges)
    #                 new_state[j] = 1.0 #abs(1.0-new_state[l])
    #                 im_s1 = self.draw_edges(new_state, e_xys)
    #                 samples.append(np.concatenate([imgs.transpose(2, 0, 1)/255.0, im_s0[np.newaxis, :, :], im_s1[np.newaxis, :, :]]))
    #                 targets.append(1)
                    
    #                 # # Debug
    #                 # rgb = Image.fromarray(imgs)
    #                 # im0 = Image.fromarray((im_s0>0)*255.0)
    #                 # im1 = Image.fromarray((im_s1>0)*255.0)
    #                 # print(edges_gt[j])
    #                 # plt.figure()
    #                 # plt.imshow(rgb)
    #                 # plt.figure()
    #                 # plt.imshow(im0)
    #                 # plt.figure()
    #                 # plt.imshow(im1)
    #                 # plt.show()

    #             # generate positive samples
    #             inds_neg = np.random.choice(inds_neg, 2, replace=True)
    #             for j in inds_neg:

    #                 # draw new state
    #                 new_state = np.array(current_edges)
    #                 new_state[j] = 1.0 #abs(1.0-new_state[l])
    #                 im_s1 = self.draw_edges(new_state, e_xys)
    #                 samples.append(np.concatenate([imgs.transpose(2, 0, 1)/255.0, im_s0[np.newaxis, :, :], im_s1[np.newaxis, :, :]]))
    #                 targets.append(0)

    #         # solve current corner
    #         for j in inds:
    #             current_edges[j] = edges_gt[j]

    #             # # draw new state
    #             # new_state = np.array(current_edges)
    #             # new_state[m] = 1.0 #abs(1.0-new_state[m])
    #             # im_s2 = self.draw_edges(new_state, e_xys)
    #             # t2 = int(new_state[m] == edges_gt[m])

    #             # # solve current corner
    #             # for k in inds:
    #             #     current_edges[k] = edges_gt[k]

    #             # # accumulate 
    #             # #ts = np.array([1, 0, 0] if t1 == t2 else [0, t1, t2])
    #             # if t1 == 1:
    #             #     samples.append(np.concatenate([imgs.transpose(2, 0, 1)/255.0, im_s0[np.newaxis, :, :], im_s1[np.newaxis, :, :]]))
    #             #     targets.append(np.argmax(np.array([0, 1])))
    #             # if t2 == 1:
    #             #     samples.append(np.concatenate([imgs.transpose(2, 0, 1)/255.0, im_s0[np.newaxis, :, :], im_s2[np.newaxis, :, :]]))
    #             #     targets.append(np.argmax(np.array([0, 1])))
    #             # if (t1 == 0) and (t2 == 0):
    #             #     samples.append(np.concatenate([imgs.transpose(2, 0, 1)/255.0, im_s0[np.newaxis, :, :], im_s1[np.newaxis, :, :]]))
    #             #     targets.append(np.argmax(np.array([1, 0])))

    #             # # Debug
    #             # print(im_s0.shape)
    #             # rgb = Image.fromarray(imgs)
    #             # im0 = Image.fromarray((im_s0>0)*255.0)
    #             # im1 = Image.fromarray((im_s1>0)*255.0)
    #             # im2 = Image.fromarray((im_s2>0)*255.0)
    #             # print(l, m)
    #             # print(t1, t2)
    #             # plt.figure()
    #             # plt.imshow(rgb)
    #             # plt.figure()
    #             # plt.imshow(im0)
    #             # plt.figure()
    #             # plt.imshow(im1)
    #             # plt.figure()
    #             # plt.imshow(im2)
    #             # plt.show()

    #     samples = np.stack(samples)
    #     targets = np.stack(targets)

    #     return samples, targets

    # def generate_input_tuple(self, corners_gt, edges_gt, e_xys, corner_edge, current_edges, imgs):

    #     samples, targets = [], []
    #     current_edges = np.zeros_like(current_edges)
    #     for i in range(corner_edge.shape[0]):

    #         # draw current state
    #         im_s0 = self.draw_edges(current_edges, e_xys)

    #         # pick two edges
    #         inds =  np.where((corner_edge[i, :] == 1) & (current_edges == 0))[0]
    #         j, k = np.random.choice(inds, 2, replace=True)

    #         # draw new state
    #         new_state = np.array(current_edges)
    #         new_state[j] = abs(1.0-new_state[j])
    #         im_s1 = self.draw_edges(new_state, e_xys)
    #         samples.append(np.concatenate([imgs.transpose(2, 0, 1)/255.0, im_s0[np.newaxis, :, :], im_s1[np.newaxis, :, :]]))
    #         targets.append(1.0 if new_state[j] == edges_gt[j] else 0.0)

    #         # Debug
    #         rgb = Image.fromarray(imgs)
    #         im0 = Image.fromarray((im_s0>0)*255.0)
    #         im1 = Image.fromarray((im_s1>0)*255.0)
    #         print('A')
    #         print(1.0 if new_state[j] == edges_gt[j] else 0.0)
    #         plt.figure()
    #         plt.imshow(rgb)
    #         plt.figure()
    #         plt.imshow(im1)
    #         plt.figure()
    #         plt.imshow(im0)
    #         plt.show()

    #         # draw new state
    #         new_state = np.array(current_edges)
    #         new_state[k] = abs(1.0-new_state[k])
    #         im_s1 = self.draw_edges(new_state, e_xys)
    #         samples.append(np.concatenate([imgs.transpose(2, 0, 1)/255.0, im_s1[np.newaxis, :, :], im_s0[np.newaxis, :, :]]))
    #         targets.append(0.0 if new_state[k] == edges_gt[k] else 1.0)

    #         # Debug
    #         print('B')
    #         rgb = Image.fromarray(imgs)
    #         im0 = Image.fromarray((im_s0>0)*255.0)
    #         im1 = Image.fromarray((im_s1>0)*255.0)
    #         print(0.0 if new_state[k] == edges_gt[k] else 1.0)
    #         plt.figure()
    #         plt.imshow(rgb)
    #         plt.figure()
    #         plt.imshow(im1)
    #         plt.figure()
    #         plt.imshow(im0)
    #         plt.show()

    #         # solve current corner
    #         for l in inds:
    #             current_edges[l] = edges_gt[l]

    #     samples = np.stack(samples)
    #     targets = np.stack(targets)

    #     return samples, targets

    def generate_input_tuple(self, corners_gt, edges_gt, e_xys, corner_edge, current_edges, imgs, n_max=128):

        samples, targets = [], []
        current_edges = np.zeros_like(current_edges)
        gt_dists = []
        inds = np.array(range(corner_edge.shape[1]))
        np.random.shuffle(inds)
        inds = inds[:n_max]
        inds = np.concatenate([inds, np.array([0])])

        for k in inds:

            #############################################################
            ##################### ADD/REMOVE Action #####################
            #############################################################

            # draw current state
            im_s0 = self.draw_edges(current_edges, e_xys)

            # draw new state
            new_state = np.array(current_edges)
            new_state[k] = abs(1.0-new_state[k])
            im_s1 = self.draw_edges(new_state, e_xys)
            samples.append(np.concatenate([imgs.transpose(2, 0, 1)/255.0, im_s0[np.newaxis, :, :], im_s1[np.newaxis, :, :]]))
            targets.append(1.0 if new_state[k] == edges_gt[k] else 0.0)

            # draw new state
            samples.append(np.concatenate([imgs.transpose(2, 0, 1)/255.0, im_s1[np.newaxis, :, :], im_s0[np.newaxis, :, :]]))
            targets.append(0.0 if new_state[k] == edges_gt[k] else 1.0)

            # solve current corner
            current_edges[k] = edges_gt[k]
            
            # # Debug
            # print('A')
            # rgb = Image.fromarray(imgs)
            # im0 = Image.fromarray((im_s0>0)*255.0)
            # im1 = Image.fromarray((im_s1>0)*255.0)
            # print(0.0 if new_state[k] == edges_gt[k] else 1.0)
            # plt.figure()
            # plt.imshow(rgb)
            # plt.figure()
            # plt.imshow(im1)
            # plt.figure()
            # plt.imshow(im0)
            # plt.show()

            # print(edges_gt)
            # print(new_state)

            # # Debug
            # rgb = Image.fromarray(imgs)
            # im1 = Image.fromarray((im_s1>0)*255.0)
            # print(np.sum(np.abs(new_state-edges_gt)))
            # plt.figure()
            # plt.imshow(rgb)
            # plt.figure()
            # plt.imshow(im1)
            # plt.show()

            

        samples = np.stack(samples)
        targets = np.stack(targets)

        return samples, targets

    def extract_edges_from_corners(self, corners_det, inds):
        c_e_assignment = []
        
        if inds is None:
            for i, c1 in enumerate(corners_det):
                for c2 in corners_det[i+1:]:
                    c_e_assignment.append((c1, c2))
        else:
            k = 0
            for i, c1 in enumerate(corners_det):
                for c2 in corners_det[i+1:]:
                    if (k in list(inds)):
                        c_e_assignment.append((c1, c2))
                    k+=1

        e_from_corners = []
        for c1, c2 in c_e_assignment:
            y1, x1 = c1[:2]
            y2, x2 = c2[:2]
            e_from_corners.append([y1, x1, y2, x2])
        e_from_corners = np.array(e_from_corners)

        return e_from_corners, c_e_assignment

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


    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self._data_refs)

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

    def rotate_coords(self, image_shape, xy, angle):
        org_center = (image_shape-1)/2.
        rot_center = (image_shape-1)/2.
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a), -org[0]*np.sin(a) + org[1]*np.cos(a)])
        return new+rot_center

    def compute_gt(self, corners_det, edges_det, corners_annot, edges_annot, max_dist=8.0, shape=256, placeholder=False):

        # init
        corners_det = np.array(corners_det)
        edges_det = np.array(edges_det)
        corners_annot = np.array(corners_annot)
        edges_annot = np.array(edges_annot)

        gt_c = np.zeros(corners_det.shape[0])
        gt_e = np.zeros(edges_det.shape[0])
        tg_to_gt_c = np.zeros(corners_annot.shape[0])-1
        tg_to_gt_e = np.zeros(edges_annot.shape[0])-1

        if not placeholder:

            # load ground-truth
            y1, x1 = corners_annot[:, 0], corners_annot[:, 1]
            y1, x1 = x1[:, np.newaxis], y1[:, np.newaxis]

            # load detections
            y2, x2 = corners_det[:, 0], corners_det[:, 1]
            y2, x2 = x2[:, np.newaxis], y2[:, np.newaxis]

            # select closest corner
            dist = np.sqrt((x1 - x2.transpose(1, 0))**2 + (y1 - y2.transpose(1, 0))**2)
            ind = np.argmin(dist, axis=1)
            d_min = np.min(dist, axis=1)

            for k in range(np.max(ind)+1):
                pos = np.where(k == ind)
                min_vals = d_min[pos]
                if min_vals.shape[0] > 0 and np.min(min_vals) < 2*max_dist:
                    l = np.argmin(min_vals)
                    tg_to_gt_c[pos[0][l]] = k
            
            for k, val in zip(ind, d_min):
                if val < 2*max_dist:
                    gt_c[k] = 1

            # select closest edges
            y1, x1, y2, x2 = np.split(edges_annot, 4, axis=-1)
            y3, x3, y4, x4 = np.split(edges_det, 4, axis=-1)

            dist13 = np.sqrt((x1 - x3.transpose(1, 0))**2 + (y1 - y3.transpose(1, 0))**2)
            dist14 = np.sqrt((x1 - x4.transpose(1, 0))**2 + (y1 - y4.transpose(1, 0))**2)
            dist23 = np.sqrt((x2 - x3.transpose(1, 0))**2 + (y2 - y3.transpose(1, 0))**2)
            dist24 = np.sqrt((x2 - x4.transpose(1, 0))**2 + (y2 - y4.transpose(1, 0))**2)
            
            d1 = dist13 + dist24
            d2 = dist14 + dist23
            d_comb = np.stack([d1, d2], axis=-1)
            d_comb = np.min(d_comb, axis=-1)
            ind = np.argmin(d_comb, axis=-1)
            d_min = np.min(d_comb, axis=-1)

            for k in range(np.max(ind)+1):
                pos = np.where(k == ind)
                min_vals = d_min[pos]
                if min_vals.shape[0] > 0 and np.min(min_vals) < max_dist*4:
                    l = np.argmin(min_vals)
                    tg_to_gt_e[pos[0][l]] = k

            for k, val in zip(ind, d_min):
                if val < max_dist*4:
                    gt_e[k] = 1

        return gt_c, gt_e

    def compute_overlaps_masks(self, masks1, masks2):
        '''Computes IoU overlaps between two sets of masks.
        masks1, masks2: [Height, Width, instances]
        '''
        # flatten masks
        masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
        masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
        area1 = np.sum(masks1, axis=0)
        area2 = np.sum(masks2, axis=0)

        # intersections and union
        intersections = np.dot(masks1.T, masks2)
        union = area1[:, None] + area2[None, :] - intersections
        overlaps = intersections / union

        return overlaps
