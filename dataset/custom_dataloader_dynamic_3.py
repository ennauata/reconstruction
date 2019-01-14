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

    def __init__(self, id_list, annots_folder, edges_folder, corners_folder, dataset_folder, cnn, with_augmentation=False, with_filter=True):
        self.corners_folder = corners_folder
        self.edges_folder = edges_folder
        self.annots_folder = annots_folder
        self._data_refs = id_list
        self.with_augmentation = with_augmentation
        self.with_filter = with_filter
        self.dataset_folder = dataset_folder
        self.cnn = cnn
        self.deb_count = 0

    def __getitem__(self, index):

        # retrieve id 
        _id = self._data_refs[index]
        self._id = _id

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

        #print(rot, flip)
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
        #ce_t0[inds] = 1.0
        corner_edge = self.compute_dists(corners_det, edges_det_from_corners, ce_assignment)
        ce_angles_bins, ca_gt = self.compute_corner_angle_bins(corner_edge, edges_det_from_corners, edges_gt, corners_det)

        if _id == '1524977362.86':
            print('test')
            return self.__getitem__(index+1)

        # read images
        imgs = self.read_input_images(_id, rot, flip, corners_det, self.dataset_folder)

        # generate states
        samples, targets, bins = self.generate_input_tuple(corners_gt, edges_gt, e_xys, corner_edge, ce_t0, imgs, ce_angles_bins)

        # convert to tensor
        samples = torch.from_numpy(samples)
        targets = torch.from_numpy(targets)
        bins = torch.from_numpy(bins)

        return samples, targets, bins

    def compute_corner_image(self, corners_annot):
        im_c = np.zeros((128, 128))
        for c in corners_annot:
            x, y, _, _ = np.array(c)/2.0
            x, y = int(x), int(y)
            im_c[x, y] = 255.0
        return im_c

    def read_input_images(self, _id, rot, flip, corners_annot, path):

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

        # print(rot, flip)
        # dp_im = dp_im.transpose(Image.FLIP_LEFT_RIGHT)
        # surf_im = surf_im.transpose(Image.FLIP_LEFT_RIGHT)
        # gray_im = gray_im.transpose(Image.FLIP_LEFT_RIGHT)
        #out_im = out_im.transpose(Image.FLIP_LEFT_RIGHT)
        #ims_c = self.compute_corner_image(corners_annot)

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
            draw.line((x1, y1, x2, y2), width=2, fill='white')
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

    def generate_bins(self, edges_state, ce_angles_bins, n_bins=72):
        ce_bins = np.zeros((ce_angles_bins.shape[0], n_bins))
        for c_ind in range(ce_angles_bins.shape[0]):
            e_inds = np.where((ce_angles_bins[c_ind] >= 0) & ((edges_state == 1))) 
            bins = ce_angles_bins[c_ind, e_inds].ravel().astype(np.int32)
            ce_bins[c_ind, bins] += 1.0
        return ce_bins

    def add_remove_edges(self, imgs, edges_state, edges_gt, e_xys, k, ce_angles_bins, corner_edge):

        # draw current state
        im_s0 = self.draw_edges(edges_state, e_xys)

        # draw new state
        new_state = np.array(edges_state)
        new_state[k] = 1.0-new_state[k]
        im_s1 = self.draw_edges(new_state, e_xys)
        sample = np.concatenate([imgs.transpose(2, 0, 1)/255.0, im_s0[np.newaxis, :, :], im_s1[np.newaxis, :, :]])
        sample_rev = np.concatenate([imgs.transpose(2, 0, 1)/255.0, im_s1[np.newaxis, :, :], im_s0[np.newaxis, :, :]])
        label = 1.0 if new_state[k] == edges_gt[k] else 0.0
        label_rev = 0.0 if new_state[k] == edges_gt[k] else 1.0

        # generate bins
        curr_bins = self.generate_bins(edges_state, ce_angles_bins)
        new_bins = self.generate_bins(new_state, ce_angles_bins)
        c1, c2 = None, None
        try:
            c1, c2 = np.where(corner_edge[:, k] == 1)[0]
        except:
            print(self._id)
            print(np.where(corner_edge[:, k] == 1)[0])

        # print(np.where(curr_bins[c1] == 1)[0]*5.0)
        # print(np.where(curr_bins[c2] == 1)[0]*5.0)
        # print(np.where(new_bins[c1] == 1)[0]*5.0)
        # print(np.where(new_bins[c2] == 1)[0]*5.0)

        curr_bin_state = np.concatenate([curr_bins[c1], curr_bins[c2]])
        new_bin_state = np.concatenate([new_bins[c1], new_bins[c2]])

        if np.random.choice([True, False]):
            curr_bin_state = np.zeros_like(curr_bin_state)
            new_bin_state = np.zeros_like(new_bin_state)

        bin_sample = np.concatenate([curr_bin_state, new_bin_state])
        bin_sample_rev = np.concatenate([new_bin_state, curr_bin_state])
        # # run CNN
        # with torch.no_grad():
        #     probs, logits = self.cnn(torch.from_numpy(sample).unsqueeze(0).float())
        #     prob = probs[0][1]
        #     dist_from_gt = abs(label-prob) # try to add more samples based on this diff (hard negative mining)
        return sample, sample_rev, label, label_rev, bin_sample, bin_sample_rev

    def remove_incident_edges(self, imgs, edges_state, corners_gt, e_xys, corner_edge):

        # draw current state
        im_s0 = self.draw_edges(edges_state, e_xys)

        # pick all incident edges
        in_es = np.where(corner_edge[c, :] == 1)

        # draw new state 1
        new_state = np.array(edges_state)
        new_state[in_es] = 0.0
        im_s = self.draw_edges(new_state, e_xys)
        sample = np.concatenate([imgs.transpose(2, 0, 1)/255.0, im_s0[np.newaxis, :, :], im_s1[np.newaxis, :, :]], 0)
        label = 0.0 if corners_gt[c] == 1 else 1.0

            # # run CNN
            # with torch.no_grad():
            #     probs, logits = self.cnn(torch.from_numpy(sample).unsqueeze(0).float())
            #     prob = probs[0][1]

        return sample, label#, prob, in_es

    def generate_input_tuple(self, corners_gt, edges_gt, e_xys, corner_edge, current_edges, imgs, ce_angles_bins, n_stages=8, n_max=16):

        # init sample list
        samples, targets, bins = [], [], []

        # pick positive indices
        pos_inds = np.where(edges_gt == 1)[0]
        np.random.shuffle(pos_inds)

        # sample some reconstruction stages
        rec_stage = range(0, pos_inds.shape[0]+1)
        rec_stage = np.random.choice(rec_stage, min(len(rec_stage), n_stages), replace=False)

        # generate samples fo reach stage of reconstruction
        #print(rec_stage)
        for N in rec_stage:

            # init
            aux_current_edges = np.zeros_like(current_edges)
            if N > 0:

                # turn on N edges
                inds = np.random.choice(pos_inds, N, replace=False)
                aux_current_edges[inds] = 1.0

                # gather connected edges (excluding noisy edges) 
                e_connected_inds = set()
                for k in inds:
                    c_ind = np.where(corner_edge[:, k] == 1)[0]
                    es = np.where(corner_edge[c_ind, :] == 1)[1]
                    e_connected_inds.update(list(es))
                e_connected_inds = np.array(list(e_connected_inds))
            else:
                e_connected_inds = np.array(range(aux_current_edges.shape[0]))

            # add some noise (turn on some false edges)
            n_flip = np.random.choice([0, 1, 2])
            if n_flip > 0:
                false_inds = np.where((edges_gt == 0))[0]
                flip_inds = np.random.choice(false_inds, n_flip, replace=False)
                aux_current_edges[flip_inds] = 1.0-aux_current_edges[flip_inds]

            # get edges for sampling
            e_off_true = []
            e_off_false = []
            for k in e_connected_inds:
                # edges off and true
                if (edges_gt[k] == 1) & (aux_current_edges[k] == 0):
                    e_off_true.append(k)
                # edges off and false
                if (edges_gt[k] == 0) & (aux_current_edges[k] == 0):
                    e_off_false.append(k)
            e_off_true = np.array(e_off_true)
            e_off_false = np.array(e_off_false)

            # print('test')
            # print(n_flip)
            # print(N)
            # print(pos_inds.shape)
            # print(e_off_false.shape)
            # print(e_off_true.shape)]

            e_inds = []
            if e_off_true.shape[0] > 0:
                s = np.random.choice(e_off_true, min(e_off_true.shape[0], int(n_max/2)), replace=False)
                e_inds.append(s)
            if e_off_false.shape[0] > 0:
                s = np.random.choice(e_off_false, min(e_off_false.shape[0], int(n_max/2)), replace=False)
                e_inds.append(s)
            if len(e_inds) > 0:
                e_inds = np.concatenate(e_inds, 0)

            # add/remove edges
            #print(e_inds)
            for k in e_inds:

                # flip k-th edge
                k = int(k)
                sample, sample_rev, label, label_rev, bin_sample, bin_sample_rev = self.add_remove_edges(imgs, aux_current_edges, edges_gt, e_xys, k, ce_angles_bins, corner_edge)
                #print(label)
                if label_rev == label:
                    print('ERR')

                # accumul samples
                samples.append(sample)
                targets.append(label)
                samples.append(sample_rev)
                targets.append(label_rev)
                bins.append(bin_sample)
                bins.append(bin_sample_rev)

                # Debug
                # draw current state
                # print(N)

                # im_s0 = self.draw_edges(aux_current_edges, e_xys)
                # rgb = Image.fromarray((sample[:3, :, :]*255.0).astype('uint8').transpose(1, 2, 0))
                # im1 = Image.fromarray((sample[3, :, :, ])*255.0)
                # im2 = Image.fromarray((sample[4, :, :]>0)*255.0)
                # im3 = Image.fromarray((sample[5, :, :]>0)*255.0)

                # f, axarr = plt.subplots(1,4)
                # f.set_size_inches(16, 4)

                # axarr[0].imshow(rgb)
                # plt.axis('off')
                # axarr[1].imshow(im1)
                # plt.axis('off')
                # axarr[2].imshow(im2)
                # plt.axis('off')
                # axarr[3].imshow(im3)
                # plt.axis('off')

                # import matplotlib.backends.backend_pdf
                # pdf = matplotlib.backends.backend_pdf.PdfPages("./debug/{}_{}_{}.pdf".format(N, k, self.deb_count))
                # self.deb_count += 1
                # pdf.savefig(f)
                # plt.close()
                # pdf.close()


                # plt.figure(figsize=((4, 4)))
                # plt.imshow(rgb)
                # plt.figure(figsize=((4, 4)))
                # plt.imshow(im1)
                # plt.figure(figsize=((4, 4)))
                # plt.imshow(im2)
                # plt.figure(figsize=((4, 4)))
                # plt.imshow(im3)
                # plt.show()

        # stack samples
        samples = np.stack(samples)
        targets = np.stack(targets)
        bins = np.stack(bins)

        return samples, targets, bins



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
