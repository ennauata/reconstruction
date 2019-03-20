import numpy as np
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw



class Metrics(): 
    def __init__(self):

        # corners
        self.curr_corner_tp = 0.0
        self.curr_corner_fp = 0.0
        self.n_corner_samples = 0.0
        self.per_corner_sample_score = {}

        # edges
        self.curr_edge_tp = 0.0
        self.curr_edge_fp = 0.0
        self.n_edge_samples = 0.0
        self.per_edge_sample_score = {}

        # loops
        self.curr_loop_tp = 0.0
        self.curr_loop_fp = 0.0
        self.n_loop_samples = 0.0
        self.per_loop_sample_score = {}

    def calc_corner_metrics(self):
        recall = self.curr_corner_tp/self.n_corner_samples
        precision = self.curr_corner_tp/(self.curr_corner_tp+self.curr_corner_fp+1e-8)
        return recall, precision

    def calc_edge_metrics(self):
        recall = self.curr_edge_tp/self.n_edge_samples
        precision = self.curr_edge_tp/(self.curr_edge_tp+self.curr_edge_fp+1e-8)
        return recall, precision

    def calc_loop_metrics(self):
        recall = self.curr_loop_tp/self.n_loop_samples
        precision = self.curr_loop_tp/(self.curr_loop_tp+self.curr_loop_fp+1e-8)
        return recall, precision

    def print_metrics(self):

        # print scores
        values = []
        recall, precision = self.calc_corner_metrics()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        values += [recall, precision, f_score]
        print('Overall Scores\n-- corners \nrecall: %.3f\nprecision: %.3f\nf_score: %.3f\n' % (recall, precision, f_score))

        # print scores
        recall, precision = self.calc_edge_metrics()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        values += [recall, precision, f_score]        
        print('Overall Scores\n-- edges \nrecall: %.3f\nprecision: %.3f\nf_score: %.3f\n' % (recall, precision, f_score))

        recall, precision = self.calc_loop_metrics()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        values += [recall, precision, f_score]
        print('-- loops \nrecall: %.3f\nprecision: %.3f\nf_score: %.3f\n' % (recall, precision, f_score))

        # print('Per sample')
        # for k in self.per_sample_score.keys():
        #     recall = self.per_sample_score[k]['recall']
        #     precision = self.per_sample_score[k]['precision']
        #     print('id: %s; recall: %.3f; precision: %.3f' % (k, recall, precision))
        print(' '.join(['%.1f'%(value * 100) for value in values]))
        
        return 

    def reset(self):

        # corners
        self.curr_corner_tp = 0.0
        self.curr_corner_fp = 0.0
        self.n_corner_samples = 0.0
        self.per_corner_sample_score = {}

        # edges
        self.curr_edge_tp = 0.0
        self.curr_edge_fp = 0.0
        self.n_edge_samples = 0.0
        self.per_edge_sample_score = {}

        # loops
        self.curr_loop_tp = 0.0
        self.curr_loop_fp = 0.0
        self.n_loop_samples = 0.0
        self.per_loop_sample_score = {}
        return

    def forward(self, building, thresh=4.0, iou_thresh=0.7):

        ## Compute corners precision/recall
        gts = np.array(building.corners_annot)[:, :2]
        dets = building.corners_det
        per_sample_corner_tp = 0.0
        per_sample_corner_fp = 0.0
        found = [False] * gts.shape[0]
        c_det_annot = {}

        # for each corner detection
        for i, det in enumerate(dets):

            # get closest gt
            near_gt = [0, 9999999.0, (0.0, 0.0)]
            for k, gt in enumerate(gts):
                dist = np.linalg.norm(gt-det)
                if dist < near_gt[1]:
                    near_gt = [k, dist, gt] 

            # hit (<= thresh) and not found yet 
            if near_gt[1] <= thresh and not found[near_gt[0]]:
                per_sample_corner_tp += 1.0
                found[near_gt[0]] = True
                c_det_annot[i] = near_gt[0]

            # not hit or already found
            else:
                per_sample_corner_fp += 1.0

        # update counters for corners
        self.curr_corner_tp += per_sample_corner_tp
        self.curr_corner_fp += per_sample_corner_fp
        self.n_corner_samples += gts.shape[0]
        self.per_corner_sample_score.update({building._id: {'recall': per_sample_corner_tp/gts.shape[0], 'precision': per_sample_corner_tp/(per_sample_corner_tp+per_sample_corner_fp+1e-8)}}) 

        ## Compute edges precision/recall
        per_sample_edge_tp = 0.0
        per_sample_edge_fp = 0.0

        # for each detected edge
        for l, e_det in enumerate(building.edge_corner):
            c1, c2 = e_det
            
            # check if corners are mapped
            if (c1 not in c_det_annot.keys()) or (c2 not in c_det_annot.keys()):
                per_sample_edge_fp += 1.0                
                continue

            # check hit
            c1_prime = c_det_annot[c1]
            c2_prime = c_det_annot[c2]
            is_hit = False
            for k, e_annot in enumerate(building.edge_corner_annots):
                c3, c4 = e_annot
                if ((c1_prime == c3) and (c2_prime == c4)) or ((c1_prime == c4) and (c2_prime == c3)):
                    is_hit = True

            # hit
            if is_hit == True:
                per_sample_edge_tp += 1.0
            # not hit 
            else:
                per_sample_edge_fp += 1.0

        # update counters for edges
        self.curr_edge_tp += per_sample_edge_tp
        self.curr_edge_fp += per_sample_edge_fp
        self.n_edge_samples += building.edge_corner_annots.shape[0]
        self.per_edge_sample_score.update({building._id: {'recall': per_sample_edge_tp/building.edge_corner_annots.shape[0], 'precision': per_sample_edge_tp/(per_sample_edge_tp+per_sample_edge_fp+1e-8)}}) 

        ## Compute loops precision/recall
        per_sample_loop_tp = 0.0
        per_sample_loop_fp = 0.0

        pred_edge_map = draw_edges(building.edge_corner, building.corners_det, mode="det")
        pred_edge_map = fill_regions(pred_edge_map)
        annot_edge_map = draw_edges(building.edge_corner_annots, building.corners_annot, mode="annot")
        annot_edge_map = fill_regions(annot_edge_map)

        pred_rs = extract_regions(pred_edge_map)
        annot_rs = extract_regions(annot_edge_map)

        # for each predicted region
        found = [False] * len(annot_rs)
        for i, r_det in enumerate(pred_rs):

            # get closest gt
            near_gt = [0, 0, (0.0, 0.0)]
            for k, r_gt in enumerate(annot_rs):
                iou = np.logical_and(r_gt, r_det).sum()/np.logical_or(r_gt, r_det).sum()
                #print(i, k, iou)
                if iou > near_gt[1]:
                    near_gt = [k, iou, r_gt] 

            # hit (<= thresh) and not found yet 
            if near_gt[1] >= iou_thresh and not found[near_gt[0]]:
                per_sample_loop_tp += 1.0
                found[near_gt[0]] = True

            # not hit or already found
            else:
                per_sample_loop_fp += 1.0

        # import cv2
        # print(pred_edge_map.shape, pred_edge_map.max())
        # cv2.imwrite('test/mask.png', (pred_edge_map).astype(np.uint8))
        # for index, mask in enumerate(pred_rs):
        #     print(mask.shape, mask.max())
        #     cv2.imwrite('test/mask_' + str(index) + '.png', (mask * 255).astype(np.uint8))
        #     continue
        # exit(1)
        
                
        # update counters for corners
        self.curr_loop_tp += per_sample_loop_tp
        self.curr_loop_fp += per_sample_loop_fp
        self.n_loop_samples += len(annot_rs)
        self.per_loop_sample_score.update({building._id: {'recall': per_sample_loop_tp/len(annot_rs), 'precision': per_sample_loop_tp/(per_sample_loop_tp+per_sample_loop_fp+1e-8)}}) 
        return np.array([per_sample_corner_tp, per_sample_corner_fp, gts.shape[0], per_sample_edge_tp, per_sample_edge_fp, building.edge_corner_annots.shape[0], per_sample_loop_tp, per_sample_loop_fp, len(annot_rs)])

def extract_regions(region_mask):
    inds = np.where((region_mask > 1) & (region_mask < 255))
    tags = set(region_mask[inds])
    tag_depth = dict()
    rs = []
    for t in tags:
        if t > 0:
            r = np.zeros_like(region_mask)
            inds = np.where(region_mask == t)
            r[inds[1], inds[0]] = 1
            if r[0][0] == 0 and r[0][-1] == 0 and r[-1][0] == 0 and r[-1][-1] == 0:
                rs.append(r)
                pass
    return rs

def fill_regions(edge_mask):
    edge_mask = edge_mask
    tag = 2
    for i in range(edge_mask.shape[0]):
        for j in range(edge_mask.shape[1]):
            if edge_mask[i, j] == 0:
                edge_mask = _flood_fill(edge_mask, i, j, tag)
                tag += 1
    return edge_mask

def _flood_fill(edge_mask, x0, y0, tag):
    new_edge_mask = np.array(edge_mask)
    nodes = [(x0, y0)]
    new_edge_mask[x0, y0] = tag
    while len(nodes) > 0:
        x, y = nodes.pop(0)
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (0 <= x+dx < new_edge_mask.shape[0]) and (0 <= y+dy < new_edge_mask.shape[0]) and (new_edge_mask[x+dx, y+dy] == 0):
                new_edge_mask[x+dx, y+dy] = tag
                nodes.append((x+dx, y+dy))
    return new_edge_mask

def draw_edges(edge_corner, corners, mode="det"):

    im = Image.new('L', (256, 256))
    draw = ImageDraw.Draw(im)
    for e in edge_corner:
        c1, c2 = e
        if "annot" in mode:
            y1, x1, _, _ = corners[c1]
            y2, x2, _, _ = corners[c2]
        elif "det" in mode:
            y1, x1 = corners[c1]
            y2, x2 = corners[c2]
        draw.line((x1, y1, x2, y2), width=1, fill='white')

    # import matplotlib.pyplot as plt
    # plt.imshow(im)
    # plt.show(im)
    return np.array(im)
