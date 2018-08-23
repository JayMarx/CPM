#coding:utf-8

import random
import os
import cv2
import numpy as np


def get_neck(point1, point2):
    if point1 and point2:
        x = (point1[0] + point2[0]) // 2
        y = (point1[1] + point2[1]) // 2
        return [x, y]
    else:
        return None

# 将coco关键点顺序转为cpm对应顺序，并增加neck点
def coco2cpm(coco_point):
    # cpm关键点顺序
    # [Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, RHip, 
    #  RKnee, RAnkle, LHip, LKnee, LAnkle, REye, LEye, REar, LEar]
    neck_point = get_neck(coco_point[5], coco_point[6])
    idx_in_coco = [0, neck_point, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    cpm_point = []
    for i in range(len(idx_in_coco)):
        if isinstance(idx_in_coco[i], int):
            cpm_point.append(coco_point[idx_in_coco[i]])
        else:
            cpm_point.append(idx_in_coco[i])

    return cpm_point


# 将图片固定长宽比resize，并padding成正方形
# 同时对point坐标更新
def resize_pad_img(img, keypoints, scale, input_size):
    resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    pad_h = (input_size - resized_img.shape[0]) // 2
    pad_w = (input_size - resized_img.shape[1]) // 2
    pad_h_offset = (input_size - resized_img.shape[0]) % 2
    pad_w_offset = (input_size - resized_img.shape[1]) % 2
    
    # 对标注节点进行转换
    point = []
    for i in range(len(keypoints)//3):
        if keypoints[i*3+2] != 0:
            x = int(keypoints[i*3] * scale) + pad_w + pad_w_offset
            y = int(keypoints[i*3+1] * scale) + pad_h + pad_h_offset
            tmp = [x, y]
            point.append(tmp)
        else:
            point.append(None)
            
    resized_pad_img = np.pad(resized_img, ((pad_h, pad_h+pad_h_offset),(pad_w, pad_w+pad_w_offset), (0, 0)),
                             mode='constant', constant_values=128)

    return resized_pad_img, point

# 图片数据处理并生成batch
def train_generator(input_size, batch_size, img_dir, annotation):
    while True:
        random.shuffle(annotation)
        for start in range(0, len(annotation), batch_size):
            img_batch = []
            point_batch = []
            if (start + batch_size) < len(annotation):
                end = start + batch_size
                anno_batch = annotation[start:end]
                for anno in anno_batch:
                    img_name = anno['img_name']
                    points_location = anno['data']['keypoints']
                    img_path = os.path.join(img_dir, img_name)

                    if not os.path.exists(img_path):
                        raise IOError('image doses not exist!')
                    
                    img = cv2.imread(img_path)
                    scale = input_size/img.shape[0] if img.shape[0]>img.shape[1] else input_size/img.shape[1]
                    
                    img_pad, coco_point = resize_pad_img(img, points_location, scale, input_size)
                    cpm_point = coco2cpm(coco_point)

                    img_batch.append(img_pad)
                    point_batch.append(cpm_point)

                yield np.array(img_batch), np.array(point_batch)