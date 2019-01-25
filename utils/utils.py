import numpy as np
from PIL import Image

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
