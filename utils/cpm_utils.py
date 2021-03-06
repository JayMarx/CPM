#coding:utf-8

import numpy as np
import math
import cv2


M_PI = 3.14159


# Compute gaussian kernel for input image
def gaussian_img(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map


def read_image(file, cam, boxsize, type):
    # from file
    if type == 'IMAGE':
        oriImg = cv2.imread(file)
    # from webcam
    elif type == 'WEBCAM':
        _, oriImg = cam.read()
    # from video
    elif type == 'VIDEO':
        oriImg = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)

    if oriImg is None:
        print('oriImg is None')
        return None

    scale = boxsize / (oriImg.shape[0] * 1.0)
    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    output_img = np.ones((boxsize, boxsize, 3)) * 128

    img_h = imageToTest.shape[0]
    img_w = imageToTest.shape[1]
    if img_w < boxsize:
        offset = img_w % 2
        # make the origin image be the center
        output_img[:, int(boxsize / 2 - math.floor(img_w / 2)):int(
            boxsize / 2 + math.floor(img_w / 2) + offset), :] = imageToTest
    else:
        # crop the center of the origin image
        output_img = imageToTest[:,
                     int(img_w / 2 - boxsize / 2):int(img_w / 2 + boxsize / 2), :]
    return output_img


def make_gaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        return np.zeros((size, size))
    else:
        x0 = center[0]
        y0 = center[1]
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / fwhm / fwhm)


def make_gaussian_batch(heatmaps, size, fwhm):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    stride = heatmaps.shape[1] // size

    batch_datum = np.zeros(shape=(heatmaps.shape[0], size, size, heatmaps.shape[3]))

    for data_num in range(heatmaps.shape[0]):
        for joint_num in range(heatmaps.shape[3] - 1):
            heatmap = heatmaps[data_num, :, :, joint_num]
            center = np.unravel_index(np.argmax(heatmap), (heatmap.shape[0], heatmap.shape[1]))

            x = np.arange(0, size, 1, float)
            y = x[:, np.newaxis]

            if center is None:
                x0 = y0 = size * stride // 2
            else:
                x0 = center[1]
                y0 = center[0]

            batch_datum[data_num, :, :, joint_num] = np.exp(
                -((x * stride - x0) ** 2 + (y * stride - y0) ** 2) / 2.0 / fwhm / fwhm)
        batch_datum[data_num, :, :, heatmaps.shape[3] - 1] = np.ones((size, size)) - np.amax(
            batch_datum[data_num, :, :, 0:heatmaps.shape[3] - 1], axis=2)

    return batch_datum


def make_heatmaps_from_joints(input_size, heatmap_size, gaussian_variance, batch_joints):
    # Generate ground-truth heatmaps from ground-truth 2d joints
    scale_factor = input_size // heatmap_size
    batch_gt_heatmap_np = []
    for i in range(len(batch_joints)):
        gt_heatmap_np = []
        # 背景heatmap
        invert_heatmap_np = np.ones(shape=(heatmap_size, heatmap_size))
        
        for j in range(len(batch_joints[0])):
            if batch_joints[i][j] != None:
                center = [x // scale_factor for x in batch_joints[i][j]]
            else:
                center = None
            cur_joint_heatmap = make_gaussian(heatmap_size,
                                              gaussian_variance,
                                              center=center)
            gt_heatmap_np.append(cur_joint_heatmap)
            invert_heatmap_np -= cur_joint_heatmap
        gt_heatmap_np.append(invert_heatmap_np)
        batch_gt_heatmap_np.append(gt_heatmap_np)
        
    batch_gt_heatmap_np = np.asarray(batch_gt_heatmap_np)
    batch_gt_heatmap_np = np.transpose(batch_gt_heatmap_np, (0, 2, 3, 1))

    return batch_gt_heatmap_np

def make_center_maps(input_size, batch_size, center_radius):
    batch_center_map = []
    center = [input_size//2, input_size//2]
    center_map = make_gaussian(input_size, center_radius, center)
    for i in range(batch_size):
        batch_center_map.append(center_map)

    batch_center_map = np.asarray(batch_center_map)
    batch_center_map = np.reshape(batch_center_map, [batch_size, input_size, input_size, 1])

    return batch_center_map


# 未使用 joint_map不明其意
def make_heatmaps_from_joints_openpose(input_size, heatmap_size, gaussian_variance, batch_joints):
    joint_map = [4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17, 0]
    # Generate ground-truth heatmaps from ground-truth 2d joints
    scale_factor = input_size // heatmap_size
    batch_gt_heatmap_np = []
    for i in range(batch_joints.shape[0]):
        gt_heatmap_np = []
        invert_heatmap_np = np.ones(shape=(heatmap_size, heatmap_size))
        for j in range(batch_joints.shape[1]):
            cur_joint_heatmap = make_gaussian(heatmap_size,
                                              gaussian_variance,
                                              center=(batch_joints[i][joint_map[j]] // scale_factor))
            gt_heatmap_np.append(cur_joint_heatmap)
            invert_heatmap_np -= cur_joint_heatmap
        gt_heatmap_np.append(invert_heatmap_np)
        batch_gt_heatmap_np.append(gt_heatmap_np)
    batch_gt_heatmap_np = np.asarray(batch_gt_heatmap_np)
    batch_gt_heatmap_np = np.transpose(batch_gt_heatmap_np, (0, 2, 3, 1))

    return batch_gt_heatmap_np