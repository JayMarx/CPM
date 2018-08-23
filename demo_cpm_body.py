#coding:utf-8

import numpy as np
from utils import cpm_utils
import cv2
import time
import math
import sys
import os
import imageio
import tensorflow as tf
from models.nets import cpm_body_slim
from tqdm import tqdm

"""Parameters
"""
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('DEMO_TYPE',
                           default_value='test_imgs/roger.png',
                           # default_value='test_imgs/test.mp4',
                           # default_value='SINGLE',
                           docstring='SINGLE: only last stage,'
                                     'paths to .jpg or .png image'
                                     'paths to .avi or .flv or .mp4 video')
tf.app.flags.DEFINE_string('model_path',
                           default_value='models/weights/model4freeze/cpm_body_slim-44000',
                           docstring='Your model')
tf.app.flags.DEFINE_integer('input_size',
                            default_value=368,
                            docstring='Input image size')
# hmap_size = input_size / 8
tf.app.flags.DEFINE_integer('hmap_size',
                            default_value=46,
                            docstring='Output heatmap size')
tf.app.flags.DEFINE_integer('cmap_radius',
                            default_value=21,
                            docstring='Center map gaussian variance')
tf.app.flags.DEFINE_integer('joints',
                            default_value=18,
                            docstring='Number of joints')
tf.app.flags.DEFINE_integer('stages',
                            default_value=6,
                            docstring='How many CPM stages')
tf.app.flags.DEFINE_string('color_channel',
                           default_value='RGB',
                           docstring='')

# Set color for each finger
joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]

"""
# 原代码模型，预测14个点
limbs = [[0, 1],
         [1, 2],
         [2, 3],
         [3, 4],
         [1, 5],
         [5, 6],
         [6, 7],
         [1, 8],
         [8, 9],
         [9, 10],
         [1, 11],
         [11, 12],
         [12, 13]]
"""

# heatmap:[Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, RHip, 
#          RKnee, RAnkle, LHip, LKnee, LAnkle, REye, LEye, REar, LEar]

limbs = [[0, 1],
         [0, 14],
         [14, 16],
         [0, 15],
         [15, 17],
         [1, 2],
         [2, 3],
         [3, 4],
         [1, 5],
         [5, 6],
         [6, 7],
         [1, 8],
         [8, 9],
         [9, 10],
         [1, 11],
         [11, 12],
         [12, 13]]


def main(argv):
    tf_device = '/gpu:0'
    with tf.device(tf_device):
        
        """
        if FLAGS.color_channel == 'RGB':
            input_data = tf.placeholder(dtype=tf.float32,
                                        shape=[None, FLAGS.input_size, FLAGS.input_size, 3],
                                        name='input_image')
        else:
            input_data = tf.placeholder(dtype=tf.float32,
                                        shape=[None, FLAGS.input_size, FLAGS.input_size, 1],
                                        name='input_image')

        center_map = tf.placeholder(dtype=tf.float32,
                                    shape=[None, FLAGS.input_size, FLAGS.input_size, 1],
                                    name='center_map')
        """
        # model = cpm_body_slim.CPM_Model(FLAGS.stages, FLAGS.joints + 1)
        # model.build_model(input_data, center_map, 1)
        # 没有背景
        model = cpm_body_slim.CPM_Model(FLAGS.input_size, FLAGS.hmap_size, FLAGS.stages, FLAGS.joints, 1, img_type = FLAGS.color_channel)

    saver = tf.train.Saver()

    """Create session and restore weights
    """
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    if FLAGS.model_path.endswith('pkl'):
        model.load_weights_from_file(FLAGS.model_path, sess, False)
    else:
        saver.restore(sess, FLAGS.model_path)

    # 不要centermap, 意义不明
    """
    # 创建center_map，正方形的中心放置高斯响应
    test_center_map = cpm_utils.gaussian_img(FLAGS.input_size,
                                             FLAGS.input_size,
                                             FLAGS.input_size / 2,
                                             FLAGS.input_size / 2,
                                             FLAGS.cmap_radius)
    test_center_map = np.reshape(test_center_map, [1, FLAGS.input_size,
                                                   FLAGS.input_size, 1])
    """

    # read in video / flow frames
    if FLAGS.DEMO_TYPE.endswith(('avi', 'flv', 'mp4')):
        # OpenCV can only read in '.avi' files
        cam = imageio.get_reader(FLAGS.DEMO_TYPE)

    # iamge processing
    with tf.device(tf_device):
        if FLAGS.DEMO_TYPE.endswith(('avi', 'flv', 'mp4')):
            ori_fps = cam.get_meta_data()['fps']

            cap = cv2.VideoCapture(FLAGS.DEMO_TYPE)
            total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            (W,H) = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            pbar = tqdm(total=total_frame)

            print('This video fps is %f' % ori_fps)
            video_length = cam.get_length()
            # writer_path = os.path.join('results', os.path.basename(FLAGS.DEMO_TYPE))
            # !! OpenCV can only write in .avi
            cv_writer = cv2.VideoWriter('results/result.mp4',
                                        # cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'),
                                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                        ori_fps,
                                        (W,H))
            # imageio_writer = imageio.get_writer(writer_path, fps=ori_fps)

            try:
                for it, im in enumerate(cam):
                    test_img_t = time.time()

                    full_img = im.copy()    # 把结果画在原图上
                    # test_img是原图像resize得到，并且进行了padding
                    test_img = cpm_utils.read_image(im, [], FLAGS.input_size, 'VIDEO')
                    # 多余的resize
                    # test_img_resize = cv2.resize(test_img, (FLAGS.input_size, FLAGS.input_size))
                    # print('img read time %f' % (time.time() - test_img_t))

                    if FLAGS.color_channel == 'GRAY':
                        test_img  = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)

                    test_img_input = test_img / 256.0 - 0.5
                    test_img_input = np.expand_dims(test_img_input, axis=0)

                    # Inference
                    # fps_t = time.time()
                    """
                    predict_heatmap, stage_heatmap_np = sess.run([model.current_heatmap,
                                                                  model.stage_heatmap,
                                                                  ],
                                                                 feed_dict={'input_placeholder:0': test_img_input,
                                                                            'cmap_placeholder:0': test_center_map})
                    """
                    predict_heatmap, stage_heatmap_np = sess.run([model.current_heatmap,
                                                                  model.stage_heatmap,
                                                                  ],
                                                                 feed_dict={'input_placeholder:0': test_img_input})

                    # Show visualized image
                    demo_img = visualize_result(test_img, full_img, FLAGS, stage_heatmap_np)

                    #cv2.putText(demo_img, "FPS: %.1f" % (1 / (time.time() - fps_t)), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
                    cv_writer.write(demo_img.astype(np.uint8))
                    pbar.update(1)
                    # print(1/(time.time()-test_img_t))
                    cv2.imwrite('results/pics_test/{}.png'.format(str(it).zfill(5)), demo_img[:,:,::-1])
                    #exit(0)
                    # imageio_writer.append_data(demo_img[:, :, 1])
            except KeyboardInterrupt:
                print('Stopped! {}/{} frames captured!'.format(it, video_length))
            finally:
                cv_writer.release()
                # imageio_writer.close()

        elif FLAGS.DEMO_TYPE.endswith(('png', 'jpg')):
            test_img = cpm_utils.read_image(FLAGS.DEMO_TYPE, [], FLAGS.input_size, 'IMAGE')
            
            test_img_input = test_img / 256.0 - 0.5
            test_img_input = np.expand_dims(test_img_input, axis=0)

            # Inference
            fps_t = time.time()
            """
            stage_heatmap_np = sess.run([model.stage_heatmap[5]],
                                    feed_dict={'input_placeholder:0': test_img_input,
                                               'cmap_placeholder:0': test_center_map})
            """
            stage_heatmap_np = sess.run([model.stage_heatmap[5]],
                                    feed_dict={'input_placeholder:0': test_img_input})

            # Show visualized image
            ori_img = cv2.imread(FLAGS.DEMO_TYPE)
            demo_img = visualize_result(test_img, ori_img, FLAGS, stage_heatmap_np)
            cv2.imwrite('results/test.jpg', demo_img)
            print('fps: %.1f' % (1 / (time.time() - fps_t)))
        
        else:
            print('Demo type is not defined, please check it!')


def visualize_result(test_img, ori_img, FLAGS, stage_heatmap_np):
    hm_t = time.time()
    demo_stage_heatmaps = []
    # last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.joints].reshape(
    #     (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
    last_heatmap = stage_heatmap_np[-1][0, :, :, 0:FLAGS.joints].reshape(
        (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
    last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    
    #print('hm resize time %f' % (time.time() - hm_t))

    joint_t = time.time()
    joint_coord_set = np.zeros((FLAGS.joints, 2))

    # Plot joint colors
    for joint_num in range(FLAGS.joints):
        # 存放关键点坐标
        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                       (test_img.shape[0], test_img.shape[1]))
        
        # 把关键点坐标对应到原图上
        joint_coord = np.array(joint_coord).astype(np.float32)
        resize_scale = FLAGS.input_size / (ori_img.shape[0] * 1.0)
        # resize back
        joint_coord /= resize_scale
        # resize的时候只对宽度进行，所以这里只减去宽上面的padding
        width_sub = FLAGS.input_size / resize_scale - ori_img.shape[1]
        padding_width = width_sub // 2 + width_sub % 2
        joint_coord[1] -= padding_width
        # 变为int
        joint_coord = list(map(int, joint_coord))

        joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

        color_code_num = (joint_num // 4)

        joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
        cv2.circle(ori_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)

    # print('plot joint time %f' % (time.time() - joint_t))
    limb_t = time.time()
    # Plot limb colors
    for limb_num in range(len(limbs)):

        x1 = joint_coord_set[limbs[limb_num][0], 0]
        y1 = joint_coord_set[limbs[limb_num][0], 1]
        x2 = joint_coord_set[limbs[limb_num][1], 0]
        y2 = joint_coord_set[limbs[limb_num][1], 1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 200 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 3),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
            cv2.fillConvexPoly(ori_img, polygon, color=limb_color)
    
    return ori_img

if __name__ == '__main__':
    tf.app.run()