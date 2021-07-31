import os
from pathlib import Path

def get_kitti_path(kitti_root):

    kitti_2012_pair = os.path.join(kitti_root, '2012/training/colored_0')
    kitti_2012_nnf = os.path.join(kitti_root, r'2012/training/nnf_colored_0')
    kitti_2012_flow_occ = os.path.join(kitti_root, r'2012/training/flow_occ')
    kitti_2012_image = os.path.join(kitti_root, '2012/training/image_2')

    kitti_2015_image = os.path.join(kitti_root, r'2015/training/image_2')
    kitti_2015_nnf = os.path.join(kitti_root, r'2015/training/nnf_image_2')
    kitti_2015_flow_occ = os.path.join(kitti_root, r'2015/training/flow_occ')

    kitti_path = dict()

    kitti_path['2012' + 'pair'] = kitti_2012_pair
    kitti_path['2012' + 'nnf'] = kitti_2012_nnf
    kitti_path['2012' + 'flow_occ'] = kitti_2012_flow_occ
    kitti_path['2012' + 'image'] = kitti_2012_image

    kitti_path['2015' + 'image'] = kitti_2015_image
    kitti_path['2015' + 'nnf'] = kitti_2015_nnf
    kitti_path['2015' + 'flow_occ'] = kitti_2015_flow_occ
    
    kitti_2012_pair_test = os.path.join(kitti_root, r'2012/testing/colored_0')
    kitti_2012_image_test = os.path.join(kitti_root, r'2012/testing/image_2')
    kitti_2012_nnf_test = os.path.join(kitti_root, r'2012/testing/nnf_colored_0')

    kitti_2015_image_test = os.path.join(kitti_root, r'2015/testing/image_2')
    kitti_2015_nnf_test = os.path.join(kitti_root, r'2015/testing/nnf_image_2')

    kitti_path['2012' + 'pair' + 'test'] = kitti_2012_pair_test
    kitti_path['2012' + 'image' + 'test'] = kitti_2012_image_test
    kitti_path['2012' + 'nnf' + 'test'] = kitti_2012_nnf_test

    kitti_path['2015' + 'image' + 'test'] = kitti_2015_image_test
    kitti_path['2015' + 'nnf' + 'test'] = kitti_2015_nnf_test

    return kitti_path
