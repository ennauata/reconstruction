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

    # def calc_loop_metrics(self):
    #     recall = self.curr_loop_tp/self.n_loop_samples
    #     precision = self.curr_loop_tp/(self.curr_loop_tp+self.curr_loop_fp+1e-8)
    #     return recall, precision

    def print_metrics(self):

        # print scores
        recall, precision = self.calc_corner_metrics()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        print('Overall Scores\n-- corners \nrecall: %.3f\nprecision: %.3f\nf_score: %.3f\n' % (recall, precision, f_score))

        # print scores
        recall, precision = self.calc_edge_metrics()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        print('Overall Scores\n-- edges \nrecall: %.3f\nprecision: %.3f\nf_score: %.3f\n' % (recall, precision, f_score))

        # recall, precision = self.calc_loop_metrics()
        # f_score = 2.0*precision*recall/(precision+recall+1e-8)
        # print('-- loops \nrecall: %.3f\nprecision: %.3f\nf_score: %.3f\n' % (recall, precision, f_score))

        # print('Per sample')
        # for k in self.per_sample_score.keys():
        #     recall = self.per_sample_score[k]['recall']
        #     precision = self.per_sample_score[k]['precision']
        #     print('id: %s; recall: %.3f; precision: %.3f' % (k, recall, precision))
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

    # def forward_loops(self, building, edges_pred, thresh=0.8):

    #     edges_pred[edges_pred>0.5] = 1.0	
    #     edges_pred[edges_pred<=0.5] = 0.0

    #     ## Use annots here for kinds of corners
    #     annots_edges_state = np.ones(building.edge_corner_annots.shape[0]) # dummy vector all 1's
    #     corners_annot = np.array(building.corners_annot)[:, :2]
    #     gt_loops = np.array(building.find_loops(annots_edges_state, building.edge_corner_annots, corners_annot, corners_annot.shape[0])) 
    #     pred_loops = np.array(building.find_loops(edges_pred, building.edge_corner, building.corners_det, building.corners_det.shape[0]))

    #     per_sample_tp = 0.0
    #     per_sample_fp = 0.0
    #     found = [False] * gt_loops.shape[0]

    #     for pred_l in pred_loops:

    #         # get closest gt
    #         near_gt = [0, 0, None]
    #         pred_loop_im = Image.new("L", (256, 256))
    #         draw = ImageDraw.Draw(pred_loop_im)
    #         loop_coords= [(x, y) for (y, x) in pred_l]
    #         draw.polygon(loop_coords, fill='white')

    #         for k, gt_l in enumerate(gt_loops):
    #             gt_loop_im = Image.new("L", (256, 256))
    #             draw = ImageDraw.Draw(gt_loop_im)
    #             loop_coords= [(x, y) for (y, x) in gt_l]
    #             draw.polygon(loop_coords, fill='white')
    #             iou = np.logical_and(gt_loop_im, pred_loop_im).sum()/np.logical_or(gt_loop_im, pred_loop_im).sum()

    #             # print(iou)
    #             # plt.figure()
    #             # plt.imshow(pred_loop_im)
    #             # plt.figure()
    #             # plt.imshow(gt_loop_im)
    #             # plt.show()

    #             # hit (>= thresh) and not found yet
    #             if iou >= near_gt[1]:   
    #                 near_gt = [k, iou, gt_l]

    #         # hit (<= thresh) and not found yet 
    #         if near_gt[1] >= thresh and not found[near_gt[0]]:
    #             per_sample_tp += 1.0
    #             found[near_gt[0]] = True

    #         # not hit or already found
    #         else:
    #             per_sample_fp += 1.0

    #     # update counters
    #     self.curr_loop_tp += per_sample_tp
    #     self.curr_loop_fp += per_sample_fp
    #     self.n_loop_samples += gt_loops.shape[0]
    #     self.per_corner_sample_score.update({building._id: {'recall': per_sample_tp/gt_loops.shape[0], 'precision': per_sample_tp/(per_sample_tp+per_sample_fp+1e-8)}}) 
        
    #     return

    def forward(self, building, thresh=4.0):

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

        # for each edge detection
        #edge_list_pred = np.array(edge_list_pred)
        per_sample_edge_tp = 0.0
        per_sample_edge_fp = 0.0

        # for each detected edge
        #for l, e_det in enumerate(edge_list_pred):
        for l, e_det in enumerate(building.edge_corner):
            c1, c2 = e_det
            
            # check if corners are mapped
            if (c1 not in c_det_annot.keys()) or (c2 not in c_det_annot.keys()):
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

        # update counters for corners
        self.curr_edge_tp += per_sample_edge_tp
        self.curr_edge_fp += per_sample_edge_fp
        self.n_edge_samples += building.edge_corner_annots.shape[0]
        self.per_edge_sample_score.update({building._id: {'recall': per_sample_edge_tp/building.edge_corner_annots.shape[0], 'precision': per_sample_edge_tp/(per_sample_edge_tp+per_sample_edge_fp+1e-8)}}) 

        statistics = np.array([per_sample_corner_tp, per_sample_corner_fp, gts.shape[0], per_sample_edge_tp, per_sample_edge_fp, building.edge_corner_annots.shape[0]])
        return statistics

    # def _forward_edges(building, thresh=8.0):

    #     per_sample_tp = 0.0
    #     per_sample_fp = 0.0
    #     found = [False] * gts.shape[0]

    #     for det in dets:

    #         # get closest gt
    #         near_gt = [0, 99999.0, (0.0, 0.0)]
    #         for k, gt in enumerate(gts):
    #             dist = np.linalg.norm(gt-det)
    #             if dist < near_gt[1]:
    #                 near_gt = [k, dist, gt] 

    #         # hit (<= threshold) and not found yet 
    #         if near_gt[1] <= thresh and not found[near_gt[0]]:
    #                 per_sample_tp += 1.0
    #                 found[near_gt[0]] = True

    #         # not hit or already found
    #         else:
    #             per_sample_fp += 1.0

    #     # update counters
    #     self.curr_tp += per_sample_tp
    #     self.curr_fp += per_sample_fp
    #     self.n_samples += gts.shape[0]
    #     self.per_sample_score.update({im_id: {'recall': per_sample_tp/gts.shape[0],
    #                                    'precision': per_sample_tp/(per_sample_tp+per_sample_fp+1e-8)}}) 
    # 	return

