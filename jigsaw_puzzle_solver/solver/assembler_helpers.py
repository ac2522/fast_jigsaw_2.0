""" assembler_helpers.py
similarity function for stitching images
and 3D distance matrix index mapper (third axis) for performance optimization
"""

import numpy as np

# direction ENUM
DIR_ENUM = {
    'u': 0,
    'r': 1,
    'd': 2,
    'l': 3,
}




def edge_similarity(edge1, edge2):

    return 1 /  (1 + np.mean(np.linalg.norm(np.subtract(edge1, edge2), axis=1)))


def img_borders_similarity(img1, img2, direction):
    """
        Evaluates stitching score of two images.

        Computes Euclidean distance between each pixels of two image borderlines
        and returns their mean.

        Todo1: skip pixels for optimization
        Todo2: compute colors around border of both sides by applying Sobel filter
        Todo3: use RMSE with weights for distance measure.
        Todo4: use Mahalanobis Gradient Compatability (MGC) Andrew C Gallagher in CVPR 2012

        @Parameters
        img1 (npArray):     raw image (h, w, c)
        img2 (npArray):     raw image (h, w, c)
        dir (uint2):        stitching direction (down, up, right, left)

        @Returns
        similarity (float): border similarity score
    """
    if direction < 2 and len(img1[0]) != len(img2[0]):
        return -1
    if len(img1) != len(img2):
        return -1

    distance = -2
    if direction == DIR_ENUM['d']:
        distance = np.mean(np.linalg.norm(np.subtract(img1[-1], img2[0]), axis=1)) 
    if direction == DIR_ENUM['u']:
        distance = np.mean(np.linalg.norm(np.subtract(img2[-1], img1[0]), axis=1))
    if direction == DIR_ENUM['r']:
        distance = np.mean(np.linalg.norm(np.subtract(img1[:, -1], img2[:, 0]), axis=1))
    if direction == DIR_ENUM['l']:
        distance = np.mean(np.linalg.norm(np.subtract(img2[:, -1], img1[:, 0]), axis=1))
    return 1/(1+distance)


def get_width_and_height(area, perimeter):
    p_half = perimeter / 2
    width = (p_half - (p_half**2 - 4 * area)**0.5) / 2
    return int(width), int(area / width)
