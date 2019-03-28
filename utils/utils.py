import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt 
import cv2
import os

def draw_loops(loop_corners, conf, corners, dst="debug/"):

    if not os.path.exists(dst):
        os.makedirs(dst)

    for c, loop in zip(conf, loop_corners):
        im = np.zeros((256, 256, 3))
        pts = (corners[loop].detach().cpu().numpy()*255.0).astype("int")
        pts = np.stack([pts[:, 1], pts[:, 0]], -1)
        cv2.polylines(im, [pts], True, (0,255,255), 3)
        cv2.imwrite('{}/{}_image.png'.format(dst, c), im.astype(np.uint8))
    return

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

def findLoopsModuleCPU(edge_confidence, edge_corner, num_corners, max_num_loop_corners=10, confidence_threshold=0, corners=None, disable_colinear=True):
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


def loop_nms(loops_conf, loop_edges, loop_corners, conf_thresh=.5, nms_thresh =.8):

    inds = np.argsort(loops_conf)[::-1]
    loops_conf_sorted = np.array(loops_conf[inds])
    loop_corners_sorted = [loop_corners[i] for i in inds]  
    loop_edges_sorted = np.array([loop_edges[i] for i in inds])

    # get loop 
    ious = loops_iou(loop_corners_sorted)

    # apply nms
    keep_track = np.zeros(loops_conf_sorted.shape[0])
    nms_inds = []
    for i in range(loops_conf_sorted.shape[0]):
        if (keep_track[i] == 0) and (loops_conf_sorted[i] > conf_thresh):
            nms_inds.append(i)
            for j in range(loops_conf_sorted.shape[0]):
                if ious[i, j] > nms_thresh:
                    keep_track[j] = 1

    return loop_edges_sorted[nms_inds], loops_conf_sorted[nms_inds]

def loops_iou(loop_corners):

    # generate polygon images
    loop_imgs = []
    for loop in loop_corners:
        poly = [(x, y) for (y, x) in loop]
        l_im = Image.new('L', (256, 256))
        draw = ImageDraw.Draw(l_im)
        draw.polygon(poly, fill='white')
        loop_imgs.append(l_im)

    # compute ious
    ious = np.zeros((len(loop_imgs), len(loop_imgs)))
    for i, l1 in enumerate(loop_imgs):
        for j, l2 in enumerate(loop_imgs):
            ious[i, j] = np.logical_and(l1, l2).sum()/np.logical_or(l1, l2).sum()
    return ious

def compose_im(im_arr, alpha, fill=None, shape=256):

    color = np.random.random_integers(0, 256, 3) if fill is None else fill
    for i in range(shape):
        for j in range(shape):
            im_arr[i, j, :] = (1-alpha[i, j])*im_arr[i, j, :] + alpha[i, j]*np.array(color)
    im_cmp = Image.fromarray(im_arr.astype('uint8')).resize((shape, shape))
    return im_cmp

def reconstruct(dwg, corners, c_prob, relations):

    # format
    relations = relations.reshape(relations.shape[1], -1)
    relations = relations.transpose(1, 0)
    corners = 2*np.array(corners.squeeze(0))
    c_prob = np.array(c_prob.squeeze(0))

    # get top 2
    ind = np.argsort(relations, axis=-1)
    val = np.sort(relations, axis=-1)
    ind = ind[:, -2:]
    val = val[:, -2:]
    for k, (i, j) in zip(val, ind):
        c1 = corners[i, :]
        c2 = corners[j, :]
        y1, x1 = c1[0], c1[1]
        y2, x2 = c2[0], c2[1]
        v1, v2 = k
        if c_prob[i] > .3 and c_prob[j] > .3: # and v1 > .3 and v2 >.3:
            dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='blue', stroke_width=3, opacity=1))

    for k, (i, j) in zip(val, ind):
        c1 = corners[i, :]
        c2 = corners[j, :]
        y1, x1 = c1[0], c1[1]
        y2, x2 = c2[0], c2[1]
        v1, v2 = k
        if c_prob[i] > .3 and c_prob[j] > .3:# and v1 > .3 and v2 >.3:
            dwg.add(dwg.circle(center=(x1, y1),r=3, stroke='red', fill='white', stroke_width=1, opacity=1))
            dwg.add(dwg.circle(center=(x2, y2),r=3, stroke='red', fill='white', stroke_width=1, opacity=1))
    return

def tileImages(image_list, background_color=0, padding=5):
    image_width = image_list[0][0].shape[1]
    image_height = image_list[0][0].shape[0]
    width = image_width * len(image_list[0]) + padding * (len(image_list[0]) + 1)
    height = image_height * len(image_list) + padding * (len(image_list) + 1)
    tiled_image = np.zeros((height, width, 3), dtype=np.uint8)
    tiled_image[:, :] = background_color
    for y, images in enumerate(image_list):
        offset_y = image_height * y + padding * (y + 1)        
        for x, image in enumerate(images):
            offset_x = image_width * x + padding * (x + 1)                    
            tiled_image[offset_y:offset_y + image_height, offset_x:offset_x + image_width] = image
            continue
        continue
    return tiled_image
