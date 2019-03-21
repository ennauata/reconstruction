import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob 
import cv2
from rdp import rdp
from torch.utils.data import DataLoader
from dataset.metrics import Metrics
from options import parse_args
from dataset.graph_dataloader import GraphData
import os
from tqdm import tqdm
from utils.utils import tileImages

def draw_edges(edges_on, edges):
    im = np.zeros((256, 256))
    for edge in edges[edges_on > 0.5]:
        cv2.line(im, (edge[1], edge[0]), (edge[3], edge[2]), thickness=3, color=1)
        continue
    return im

def main(options):
	PREFIX = "/home/nelson/Workspace/ruyhei_exp/building/output_tta/"
	fnames = glob.glob("{}*".format(PREFIX))
	buildings = {}
	for fn in fnames:
		im = cv2.imread(fn)*255

		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(im, 127, 255, 0)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = [rdp(c, epsilon=10) for c in contours]

		# filter contours
		contours_filtered = []
		for c in contours:
			c_im = np.zeros((256, 256))
			cv2.fillPoly(c_im,[c],color=(255, 255, 255))
			px_area = (c_im/255.0).sum()
			if px_area > 100:
				contours_filtered.append(c)
		rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

		# construct graph
		class building_ryu:
			corners_det = []
			edge_det = []
			edge_corner = []
			output_im = rgb
			_id = fn.replace(".jpg", "").replace(PREFIX, "").strip()

		ccount = 0
		for contour in contours_filtered:
			contour = contour.reshape(-1, 2)
			for curr_c in contour:
				x1, y1 = curr_c
				building_ryu.corners_det.append((y1, x1))
				# cv2.circle(rgb, (x1, y1), 2, (0, 0, 255), 2)

			prev_c = contour[-1]
			for curr_c in contour:
				x1, y1 = curr_c
				x2, y2 = prev_c
				prev_c = curr_c
				building_ryu.edge_det.append((y1, x1, y2, x2))
				# cv2.line(rgb, (x1, y1), (x2, y2),(0, 255, 0), 2)
		
		for e in building_ryu.edge_det:
			edge_c = []
			y1, x1, y2, x2 = e
			c1 = np.array([y1, x1])
			c2 = np.array([y2, x2])
			for k, c in enumerate(building_ryu.corners_det):
				cx = np.array(c)
				if np.array_equal(c1, cx) or np.array_equal(c2, cx):
					edge_c.append(k)
			building_ryu.edge_corner.append(edge_c)
		buildings[building_ryu._id] = building_ryu

	# compute metrics
	with open('/home/nelson/Workspace/dataset_atlanta/processed/valid_list_V2.txt') as f:
	    valid_list = [line.strip() for line in f.readlines()]
	    valid_list = valid_list[-100:]

	metrics = Metrics()
	dset_val = GraphData(options, valid_list, split='val', num_edges=0, load_heatmaps=True)
	data_loader = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=1)    
	data_iterator = tqdm(data_loader, total=len(dset_val)) 
	all_images = []
	row_images = [] 
	for sample_index, sample in enumerate(data_iterator):
		im_arr, corner_images, edge_images, corners, edges, corner_gt, edge_gt, corner_edge_pairs, edge_corner, left_edges, right_edges, building_index = sample[0].cuda().squeeze(0), sample[1].cuda().squeeze(0), sample[2].cuda().squeeze(0), sample[3].cuda().squeeze(0), sample[4].cuda().squeeze(0), sample[5].cuda().squeeze(0), sample[6].cuda().squeeze(0), sample[7].cuda().squeeze(), sample[8].cuda().squeeze(0), sample[9].cuda().squeeze(), sample[10].cuda().squeeze(), sample[11].squeeze().item()	
		building = dset_val.buildings[building_index]
		building_ryu = buildings[building._id]
		building.corners_det = building_ryu.corners_det
		building.edge_det = building_ryu.edge_det
		building.edge_corner = building_ryu.edge_corner
		metrics.forward(building)
		
		# draw images
		if np.array(building_ryu.corners_det).shape[0] > 0:
			corner_annot, corner_masks = building.compute_corner_image(np.array(building_ryu.corners_det).astype('int'))
			corner_image_annot = building.rgb.copy()
			corner_image_annot[corner_annot > 0.5] = np.array([255, 0, 0], dtype=np.uint8)
		else:
			corner_image_annot = building.rgb.copy()

		if np.array(building_ryu.edge_det).shape[0] > 0:
			edge_image_annot = corner_image_annot.copy()
			edge_mask = draw_edges(np.ones(np.array(building_ryu.edge_det).shape[0]), np.array(building_ryu.edge_det).astype('int'))
			edge_image_annot[edge_mask > 0.5] = np.array([255, 0, 255], dtype=np.uint8)
		else:
			edge_image_annot = building.rgb.copy()

		row_images.append(building_ryu.output_im)
		row_images.append(corner_image_annot)
		row_images.append(edge_image_annot)
		all_images.append(row_images)
		row_images = []    

	image = tileImages(all_images, background_color=0)
	cv2.imwrite('results_ryu.png', image)   

	metrics.print_metrics()

if __name__ == '__main__':
    args = parse_args()
    args.keyname = 'image'
    args.keyname += '_' + args.corner_type
    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass
    if args.conv_type != '':
        args.keyname += '_' + args.conv_type
        pass    
    args.test_dir = 'test/' + args.keyname
    args.checkpoint_dir = 'checkpoint/' + args.keyname

    if not os.path.exists(args.checkpoint_dir):
        os.system("mkdir -p %s"%args.checkpoint_dir)
        pass
    if not os.path.exists(args.test_dir):
        os.system("mkdir -p %s"%args.test_dir)
        pass
    if not os.path.exists(args.test_dir + '/cache'):
        os.system("mkdir -p %s"%args.test_dir + '/cache')
        pass

    main(args)