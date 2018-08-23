class FLAGS(object):
    """ """
    """
    General settings
    """
    input_size = 368
    heatmap_size = 46
    cpm_stages = 6
    joint_gaussian_variance = 1.0
    center_radius = 21
    num_of_joints = 18
    color_channel = 'RGB'
    normalize_img = True
    use_gpu = True
    gpu_id = 0

    """
    Demo settings
    """
    model_path = 'cpm_body_slim'

    """
    Training settings
    """
    network_def = 'cpm_body_slim'
    train_img_dir = '/root/COCO/COCO_new/image'
    anns_path = '/root/COCO/COCO_new/annotation/anns.json'
    pretrained_model = ''
    batch_size = 32
    init_lr = 0.0001
    lr_decay_rate = 0.8
    lr_decay_step = 15000
    training_iters = 300000     # 100多 epochs
    validation_iters = 2000
    model_save_iters = 2000
    augmentation_config = {'hue_shift_limit': (-5, 5),
                           'sat_shift_limit': (-10, 10),
                           'val_shift_limit': (-15, 15),
                           'translation_limit': (-0.15, 0.15),
                           'scale_limit': (-0.3, 0.5),
                           'rotate_limit': (-90, 90)}
    hnm = True  # Make sure generate hnm files first
    do_cropping = True

    """
    For Freeze graphs
    """
    output_node_names = 'stage_6/mid_conv7/BiasAdd'


    """
    For Drawing
    """
    # Default Pose
    default_hand = [[259, 335],
                    [245, 311],
                    [226, 288],
                    [206, 270],
                    [195, 261],
                    [203, 308],
                    [165, 290],
                    [139, 287],
                    [119, 284],
                    [199, 328],
                    [156, 318],
                    [128, 314],
                    [104, 318],
                    [204, 341],
                    [163, 340],
                    [133, 347],
                    [108, 349],
                    [206, 359],
                    [176, 368],
                    [164, 370],
                    [144, 377]]

    # Limb connections
    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    # Finger colors
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]