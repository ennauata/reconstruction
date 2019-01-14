import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import glob
import numpy as np
import pickle as p
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset.custom_dataloader_lift import GraphData
from torch.utils.data import DataLoader
from dataset.collate import PadCollate
from utils.losses import balanced_binary_cross_entropy
import os
from dataset.metrics import Metrics
from collections import OrderedDict
import svgwrite
from cairosvg import svg2png
from utils.utils import compose_im
from sklearn import linear_model
from utils.utils import reconstruct
from model.graph import EdgeClassifier
import triangle
import triangle.plot as plot
import matplotlib.pyplot as plt
from model.reconstruction import ReconstructionModule

##############################################################################################################
############################################### Define Model #################################################
##############################################################################################################
rec_module = ReconstructionModule()
edge_classifier = EdgeClassifier()
edge_classifier = edge_classifier.cuda()
edge_classifier = edge_classifier.eval()

epoch = 240
src = 'saved_models'
edge_classifier.load_state_dict(torch.load('./{}/edge_classifier_iter_{}.pth'.format(src, epoch)))


##############################################################################################################
############################################# Setup Training #################################################
##############################################################################################################
PREFIX = '/home/nelson/Workspace'
SVG_FOLDER = '{}/building_reconstruction/la_dataset_new/svg'.format(PREFIX)
REF_FOLDER = '{}/building_reconstruction/la_dataset_new/refs'.format(PREFIX)
OBJ_FOLDER = '{}/building_reconstruction/la_dataset_new/obj'.format(PREFIX)
DEPTH_FOLDER = '{}/building_reconstruction/la_dataset_new/depth'.format(PREFIX)
RGB_FOLDER = '{}/building_reconstruction/la_dataset_new/rgb'.format(PREFIX)
OUT_FOLDER = '{}/building_reconstruction/la_dataset_new/outlines'.format(PREFIX)
ANNOTS_FOLDER = '{}/building_reconstruction/la_dataset_new/annots'.format(PREFIX)
EDGES_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_dets/edges'.format(PREFIX)
CORNERS_FOLDER = '{}/building_reconstruction/la_dataset_new/expanded_primitives_dets/corners'.format(PREFIX)
#EDGES_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/expanded_primitives_dets/edges'
#CORNERS_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/expanded_primitives_dets/corners'
with open('{}/building_reconstruction/la_dataset_new/sample_list.txt'.format(PREFIX)) as f:
    valid_list = [line.strip() for line in f.readlines()][::-1]

# create loaders
dset_valid = GraphData(valid_list, ANNOTS_FOLDER, EDGES_FOLDER, CORNERS_FOLDER, with_augmentation=False)
valid_loader = DataLoader(dset_valid, batch_size=1, shuffle=False, num_workers=1)

##############################################################################################################
################################################ Start Search ################################################
##############################################################################################################

def triangulate_building(corners, edges, current_edges, _id):

    # get set of corners
    edges_set = set()
    corners_set = set()

    # add terrain boundaries
    corners_set.add(tuple([0, 0]))
    corners_set.add(tuple([0, 256]))
    corners_set.add(tuple([256, 0]))
    corners_set.add(tuple([256, 256]))
    edges_set.add(tuple([0, 0, 0, 256]))
    edges_set.add(tuple([0, 256, 256, 256]))
    edges_set.add(tuple([256, 0, 256, 256]))
    edges_set.add(tuple([256, 0, 0, 0]))

    # add building primitives
    for k, e in enumerate(edges):
        x1, y1, x2, y2 = e.cpu().numpy()
        if current_edges[k] == 1:
            edges_set.add(tuple([x1, y1, x2, y2]))
            for c in corners:
                x, y, _, _ = c.cpu().numpy()
                if (x1 == x) and (y1 == y):
                    corners_set.add(tuple([x, y]))
                elif (x2 == x) and (y2 == y):
                    corners_set.add(tuple([x, y]))

    # get segments
    edges_set = list(edges_set)
    corners_set = list(corners_set) 
    segments = []
    for e in edges_set:
        x1, y1, x2, y2 = e
        inds = []
        for k, c in enumerate(corners_set):
            x, y = c
            if (x1 == x) and (y1 == y):
                inds.append(k)
            elif (x2 == x) and (y2 == y):
                inds.append(k)
        segments.append(inds)
    
    # draw mesh
    poly = dict(vertices=np.array(corners_set), segments=np.array(segments))
    t = triangle.triangulate(poly, 'p')

    # plt.figure()
    # ax1 = plt.subplot(121, aspect='equal')
    # plot.plot(ax1, **poly)
    # ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
    # plot.plot(ax2, **t)
    plt.savefig('{}/{}.png'.format(REF_FOLDER, _id))
    plt.close()
    plt.show()
    
    return t

def compute_edges_mask(edges, labels):
    im_arr = np.zeros((256, 256))
    im = Image.fromarray(im_arr)
    draw = ImageDraw.Draw(im)
    for k, e in enumerate(edges):
        x1, y1, x2, y2 = e
        if labels[k] == 1:
            draw.line((x1, y1, x2, y2), width=1, fill='white')

    inds = np.array(np.where(np.array(im) > 0))
    return np.array(im) 

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

def fill_regions(edge_mask):
    edge_mask = edge_mask
    tag = 2
    for i in range(edge_mask.shape[0]):
        for j in range(edge_mask.shape[1]):
            if edge_mask[i, j] == 0:
                edge_mask = _flood_fill(edge_mask, i, j, tag)
                tag += 1
    return edge_mask

def compute_depth_per_region(region_mask, dmap, mult=0.1):
    inds = np.where((region_mask > 1) & (region_mask < 255))
    tags = set(region_mask[inds])
    tag_depth = dict()
    for t in tags:
        if t > 0:
            inds = np.where(region_mask == t)
            #print(dmap[inds])
            med = np.median(dmap[inds[1], inds[0]])
            avg = np.mean(dmap[inds[1], inds[0]])
            #print(med, mult*avg)
            val = med + mult*avg
        else:
            val = 0

        # fall outside depth map
        tag_depth[t] = val
    return tag_depth

def add_verticals(triangles):
    to_add = []
    for i, t1 in enumerate(triangles):
        for j, t2 in enumerate(triangles[i:]):
            if i != j:

                # first check if triangles are in diff heights
                h1 = t1[0][2]
                h2 = t2[0][2]
                if h1 != h2:

                    # check if triangles share vertices
                    shared_xy = []
                    non_shared_v1 = None
                    for v1 in t1:
                        is_shared = False
                        for v2 in t2:
                            if (v1[0] == v2[0]) and (v1[1] == v2[1]):
                                shared_xy.append((v1[0], v1[1]))
                                is_shared = True
                        if not is_shared:
                            non_shared_v1 = v1

                    non_shared_v2 = None
                    for v2 in t2:
                        for (x, y) in shared_xy:
                            is_shared = False
                            if (x == v2[0]) and (y == v2[1]):
                                is_shared = True
                        if not is_shared:
                            non_shared_v2 = v2

                    # check if triangles share at least 2 vertices
                    if len(shared_xy) == 2:
                        
                        # create vertical triangle
                        A = (*shared_xy[0], h1)
                        B = (*shared_xy[1], h1)
                        C = (*shared_xy[0], h2)
                        v_t1 = (A, B, C)

                        # create vertical triangle
                        D = (*shared_xy[1], h2)
                        v_t2 = (D, C, B)
                        
                        # accumul
                        to_add.append(v_t1)
                        to_add.append(v_t2)

    return triangles + to_add

def write_obj(_id, triangles):
    dst = '{}/{}.obj'.format(OBJ_FOLDER, _id)
    with open(dst, 'w') as f:
        f.write("# OBJ file\n")
        for k, t in enumerate(triangles):
            for v in t:
                f.write("v %.4f %.4f %.4f\n" % tuple([v[0]/255.0, 1-(v[2]/255.0), 1-(v[1]/255.0)]))
            f.write("f %d %d %d\n" % tuple(range(3*k+1, 3*(k+1)+1)))
    return

def compute_depth_per_triangle(mesh, region_mask, tag_depth_map):
    triangles = []
    for t in mesh['triangles']:
        
        # get vertices coordinates
        A = mesh['vertices'][t[0]]
        B = mesh['vertices'][t[1]]
        C = mesh['vertices'][t[2]]

        # compute center of mass
        Ox = (A[0] + B[0] + C[0])/3.0
        Oy = (A[1] + B[1] + C[1])/3.0

        # get tag of center of mass
        Ot = region_mask[int(Oy), int(Ox)]

        # triangle fall outside ROI
        if Ot != 255.0:
            A_prime = (A[0], A[1], tag_depth_map[Ot])
            B_prime = (B[0], B[1], tag_depth_map[Ot])
            C_prime = (C[0], C[1], tag_depth_map[Ot])
        else:
            A_prime = (A[0], A[1], 0)
            B_prime = (B[0], B[1], 0)
            C_prime = (C[0], C[1], 0)
        triangles.append([A_prime, B_prime, C_prime])

    return triangles

def __debug(dmap, t_coords, region_mask):
    import random
    inds = np.where(np.array(region_mask) > 0)
    regions = set(np.array(region_mask)[inds])
    dmap = np.array(Image.fromarray(dmap).convert('RGB'))
    for r in regions:
        inds = np.where(region_mask == r)
        r = int(random.random() * 256)
        g = int(random.random() * 256)
        b = int(random.random() * 256)
        c = (r, g, b)

        dmap[inds[1], inds[0], :] = c
    dmap = Image.fromarray(dmap)
    plt.figure()
    plt.imshow(dmap)
    plt.show()

    # region_mask = np.array(region_mask)
    # map_debug = Image.fromarray(np.zeros((256, 256, 3)).astype('uint8'))
    # draw = ImageDraw.Draw(map_debug)
    # colors = dict()
    # for t in t_coords:
    #     A, B, C = t
    #     Ox = (A[0] + B[0] + C[0])/3.0
    #     Oy = (A[1] + B[1] + C[1])/3.0
    #     Ot = region_mask[int(Oy), int(Ox)]
    #     if Ot in colors:
    #         c = colors[Ot]
    #     else:
    #         r = int(random.random() * 256)
    #         g = int(random.random() * 256)
    #         b = int(random.random() * 256)
    #         c = (r, g, b)
    #         colors[Ot] = c
    #     draw.polygon([(A[0], A[1]), (B[0], B[1]), (C[0], C[1]), (A[0], A[1])], fill = c)
    # plt.figure()
    # plt.imshow(map_debug)
    # plt.show()
    return

def compute_depth(_id, mesh, edges, labels):

    # compute edge mask
    edge_mask = compute_edges_mask(edges, labels)

    # compute region mask
    region_mask = fill_regions(edge_mask)

    # read depth image
    dmap = np.array(Image.open('{}/{}.jpg'.format(DEPTH_FOLDER, _id)).convert('L'))

    # compute depth per region
    tag_depth_map = compute_depth_per_region(region_mask, dmap)

    # assign depth to each triangle
    t_coords = compute_depth_per_triangle(mesh, region_mask, tag_depth_map)

    #__debug(dmap, t_coords, region_mask)

    # add vertical triangles to the mesh
    t_coords = add_verticals(t_coords)

    # write to 3D obj file
    write_obj(_id, t_coords)
    return

def remove_dangling_edges(current_edges, corner_edge):

    new_current_edges = np.array(current_edges)
    while np.sum(new_current_edges) > 0:

        # get corners degree roughly
        inds = np.where(corner_edge==1)
        deg = np.zeros_like(corner_edge)
        deg[inds[0], inds[1]] = new_current_edges[inds[1]]
        deg = np.sum(deg, 1)
        inds = np.where(deg == 1)[0]

        if inds.shape[0] > 0:
            for c_idx in inds:
                e_inds = np.where(corner_edge[c_idx, :]==1)
                new_current_edges[e_inds] = 0
        else:
            break

    return new_current_edges

def write_to_svg(corners, edges, edges_conf, connections, _id, tresh=.5):

     # draw final state
    im_path = os.path.join(RGB_FOLDER, _id + '.jpg')
    dwg = svgwrite.Drawing('{}/{}.svg'.format(SVG_FOLDER, _id), (256, 256))
    dwg.add(svgwrite.image.Image(im_path, size=(256, 256)))

    # # Draw corners
    # for i in range(corners.shape[1]):
    #     y, x = corners[0, i, :2]
    #     y, x = float(y), float(x)
    #     dwg.add(dwg.circle(center=(x,y),r=2, stroke='yellow', fill='white', stroke_width=1, opacity=.8))
    
    # Collect relations
    edges_to_draw = []
    corners_to_draw = set()
    all_corners = set()
    for i in range(edges.shape[0]):
        y1, x1, y2, x2 = edges[i, :]
        if edges_conf[i] > tresh:
            inds = np.where(connections[:, i] == 1)
            if inds[0].shape[0] > 2:
                print('ERROR', _id)
                print(inds[0].shape)
            c1_idx = inds[0][0]
            c2_idx = inds[0][1]
            pt1 = corners[c1_idx, :2]
            pt2 = corners[c2_idx, :2]
            edges_to_draw.append([float(pt1[1]), float(pt1[0]), float(pt2[1]), float(pt2[0])])
            corners_to_draw.update([(float(pt1[1]), float(pt1[0]), c1_idx)])
            corners_to_draw.update([(float(pt2[1]), float(pt2[0]), c2_idx)])

    
    for k in range(corners.shape[0]):
        pt = corners[k, :2]
        all_corners.update([(float(pt[1]), float(pt[0]), k)])

    # Draw edges and corner
    for e in edges_to_draw:
        x1, y1, x2, y2 = e
        dwg.add(dwg.line((x1, y1), (x2, y2), stroke='magenta', stroke_width=1, opacity=.7))

    for c in corners_to_draw:
        x, y, c_id  = c
        dwg.add(dwg.circle(center=(x,y),r=1.5, stroke='blue', fill='white', stroke_width=0.8, opacity=.8))
    
    for c in all_corners:
        if c not in corners_to_draw:
            x, y, c_id  = c
            dwg.add(dwg.circle(center=(x,y),r=1.5, stroke='red', fill='white', stroke_width=0.8, opacity=.8))
    dwg.save()
    return

# for each building
for l, data in enumerate(valid_loader):
   

    # get the inputs
    _id = valid_list[l]
    print(_id)
    corners_det, edges_det, corner_edge, e_xys, current_edges = data

    # get image
    im = np.array(Image.open("{}/{}.jpg".format(RGB_FOLDER, _id)).resize((128, 128)))
    out = np.array(Image.open("{}/{}.jpg".format(OUT_FOLDER, _id)).resize((128, 128)).convert('L'))
    im = np.concatenate([im, out[:, :, np.newaxis]], -1)

    # format input
    corners_det = corners_det.squeeze(0)
    edges_det = edges_det.squeeze(0)
    corner_edge = corner_edge.squeeze(0)
    e_xys = e_xys.squeeze(0)
    #current_edges = torch.zeros_like(current_edges).squeeze(0)
    current_edges = current_edges.squeeze(0)

    current_edges = rec_module.greedy_over_edges(current_edges, corner_edge, e_xys, im, edge_classifier)
    current_edges = remove_dangling_edges(current_edges, corner_edge)

    mesh = triangulate_building(corners_det, edges_det, current_edges, _id)
    compute_depth(_id, mesh, edges_det, current_edges)

    write_to_svg(corners_det, edges_det, current_edges, corner_edge, _id)