import json
import contextlib
import io
from pycocotools.coco import COCO
import numpy as np
import pycocotools.mask as mask_utils
import os.path as osp

CSE_ANN_KEYS = ['bbox', 'dp_x', 'dp_y', 'dp_I', 'dp_U', 'dp_V', 'dp_masks']
CSE_IUV_KEYS = ['dp_x', 'dp_y', 'dp_I', 'dp_U', 'dp_V']

N_BODY_PARTS = 14
MASK_SIZE = 256
DP_CLOTHES_KEY = 'dp_clothes'
N_DP_CLOTHES_PARTS = 10


def init_from_coco_json_file(json_file):
    # 256*256
    img_infos = dict()
    for k in CSE_ANN_KEYS:
        img_infos[k] = []
    img_infos['img_path'] = []

    def id2image(images):
        image_id_vs_image = {}
        for img in images:
            image_id = img['id']
            image_id_vs_image[image_id] = img
        return image_id_vs_image

    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    # dataset_name = json_file.split('/')[-2]
    if 'DP3D_train' in json_file:
        image_root = './data/DP3D/train'
    else:
        image_root = ''
        print('wrong file chosen !')

    images = coco_api.dataset['images']
    annotations = coco_api.dataset['annotations']
    id2image_dict = id2image(images)
    for ann in annotations:
        if any([k not in ann for k in CSE_ANN_KEYS]):
            continue
        image_id = ann['image_id']
        img = id2image_dict[image_id]
        for k in CSE_ANN_KEYS:
            img_infos[k].append(ann[k])
        img_infos['img_path'].append(osp.join(image_root, img['file_name']))
    return img_infos


def extract_segmentation_mask(poly_specs):
    segm = np.zeros([MASK_SIZE, MASK_SIZE])
    for i in range(N_BODY_PARTS):
        ploy_i = poly_specs[i]
        if ploy_i:
            mask_i = mask_utils.decode(ploy_i)
            segm[mask_i > 0] = i + 1
    return segm[:, :, np.newaxis]


def extract_dp_clothes_mask(ploy_specs):
    segm = np.zeros([MASK_SIZE, MASK_SIZE])
    for i in range(N_DP_CLOTHES_PARTS):
        ploy_i = ploy_specs[i]
        if ploy_i:
            mask_i = mask_utils.decode(ploy_i)
            segm[mask_i > 0] = i + 1
    return segm[:, :, np.newaxis]


