import torch
import cv2
import ctypes
import numpy as np
import random
import io
from PIL import Image
from torch.utils.data import Dataset
from reid_cse.cse_tools import CSE_ANN_KEYS, CSE_IUV_KEYS, init_from_coco_json_file
from reid_cse.cse_tools import extract_segmentation_mask
from torchvision import transforms


class ColorJitterWithProb(transforms.ColorJitter):
    def __init__(self, prob, **kwargs):
        super(ColorJitterWithProb, self).__init__(**kwargs)
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            return super(ColorJitterWithProb, self).__call__(img)
        else:
            return img



class GaussianBlurWithProb(transforms.GaussianBlur):
    def __init__(self, prob, **kwargs):
        super(GaussianBlurWithProb, self).__init__(**kwargs)
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            return super(GaussianBlurWithProb, self).__call__(img)
        else:
            return img


class CSEDataset(Dataset):
    def __init__(self, json_list, train=False, height=256, width=128, repeat=1):
        img_infos = dict()
        for k in CSE_ANN_KEYS:
            img_infos[k] = []
        img_infos['img_path'] = []
        for json_file in json_list:
            info = init_from_coco_json_file(json_file)
            for k, v in info.items():
                img_infos[k].extend(v)
                print(json_file)
                print('bbox num is', len(info['img_path']))
        print('total bbox num is ', len(img_infos['img_path']))
        self.total_bbox_num = len(img_infos['img_path'])
        self.img_infos = img_infos
        self.height = height
        self.width = width
        self.mean = torch.Tensor([102.9801, 115.9465, 122.7717])[:, None, None]
        self.std = torch.tensor([58.8235294117647, 58.8235294117647, 58.8235294117647])[:, None, None]
        if train:
            self.transform = transforms.Compose([
                ColorJitterWithProb(prob=0.8, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlurWithProb(prob=0.5, kernel_size=(11, 11), sigma=(0.1, 2.0))
            ])
        else:
            self.transform = None
        self.count = 0
        self.repeat = repeat

    def _crop(self, image, x1_ratio, x2_ratio, y1_ratio, y2_ratio, x1, x2, y1, y2):
        '''
        :param image:
        large input picture
        :param x1_ratio:
            expand ratio of left corner
        :param x2_ratio:
            expand ratio of right corner
        :param y1_ratio:
            expand ratio of top corner
        :param y2_ratio:
            expand ratio of bottom corner
        :return:
            cropped image
        '''
        hei, wid = image.shape[:2]
        crop_x1 = int((x2 - x1 + 1) * x1_ratio) + x1
        crop_x2 = int((x2 - x1 + 1) * x2_ratio) + x2
        crop_y1 = int((y2 - y1 + 1) * y1_ratio) + y1
        crop_y2 = int((y2 - y1 + 1) * y2_ratio) + y2
        image = image[max(0, crop_y1):crop_y2 + 1, max(0, crop_x1):crop_x2 + 1, :]
        image = np.pad(image,
                       [[max(0, -crop_y1), max(0, crop_y2 - hei + 1)], [max(0, -crop_x1), max(0, crop_x2 - wid + 1)],
                        [0, 0]])
        return image

    def _mask_resize(self, mask, crop_img_shape):
        h, w = crop_img_shape
        ratio1 = float(h) / float(w)
        ratio2 = float(w) / float(h)
        if ratio1 > 1:
            target_h = mask.shape[0]
            target_w = int(mask.shape[1] / ratio1)
        else:
            target_w = mask.shape[1]
            target_h = int(mask.shape[0] / ratio2)
        resize_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        if len(resize_mask.shape) == 2:
            resize_mask = resize_mask[:, :, np.newaxis]
        return resize_mask

    def crop_img(self, image, bbox, dp_masks=None, dp_x=None, dp_y=None, bbox_type='xywh'):
        if bbox_type == 'xywh':
            x1, y1, w, h = [int(item) for item in bbox]
            x2 = x1 + w - 1
            y2 = y1 + h - 1
        elif bbox_type == 'xyxy':
            x1, y1, x2, y2 = [int(item) for item in bbox]
        else:
            assert False, f'Type of {bbox_type} is not supported'
        image = np.array(image)[:, :, ::-1]
        hei, wid = image.shape[:2]
        x1 = min(wid - 1, max(0, x1))
        x2 = min(wid - 1, max(0, x2))
        y1 = min(hei - 1, max(0, y1))
        y2 = min(hei - 1, max(0, y2))
        bbox_wid, bbox_hei = x2 - x1 + 1, y2 - y1 + 1
        # dp_clothes = self._mask_resize(dp_clothes, (bbox_hei, bbox_wid))
        dp_masks = self._mask_resize(dp_masks, (bbox_hei, bbox_wid))
        if self.transform is not None:
            x1_ratio, y1_ratio = np.random.uniform(-0.15, 0.05, 2)
            x2_ratio, y2_ratio = np.random.uniform(-0.05, 0.15, 2)
        else:
            x1_ratio, x2_ratio, y1_ratio, y2_ratio = 0, 0, 0, 0
        # transform img and mask
        crop_image = self._crop(image, x1_ratio, x2_ratio, y1_ratio, y2_ratio, x1, x2, y1, y2)
        dp_masks = self._crop(dp_masks, x1_ratio, x2_ratio, y1_ratio, y2_ratio, 0, dp_masks.shape[1] - 1, 0,
                              dp_masks.shape[0] - 1)
        # dp_clothes = self._crop(dp_clothes, x1_ratio, x2_ratio, y1_ratio, y2_ratio, 0, dp_clothes.shape[1] - 1, 0,
        # dp_clothes.shape[0] - 1)
        # transform points coordinates
        j_valid = ((dp_x / 255.0) >= x1_ratio) & ((dp_x / 255.0) <= 1 + x2_ratio) & ((dp_y / 255.0) >= y1_ratio) & (
                (dp_y) / 255.0 <= 1 + y2_ratio)
        dp_x = ((dp_x / 255.0) * (bbox_wid - 1) - int(bbox_wid * x1_ratio)) / (crop_image.shape[1] - 1) * 255.0
        dp_y = ((dp_y / 255.0) * (bbox_hei - 1) - int(bbox_hei * y1_ratio)) / (crop_image.shape[0] - 1) * 255.0
        crop_image = Image.fromarray(crop_image)
        return crop_image, dp_masks, dp_x, dp_y, j_valid

    def pad_resize_img(self, img, mode=cv2.INTER_LINEAR):
        # pad
        h, w, c = img.shape
        target_ratio = self.height / (float(self.width))
        bbox_ratio = h / (float(w))
        padding_hei = h if bbox_ratio >= target_ratio else int(w * target_ratio)
        padding_wid = w if bbox_ratio <= target_ratio else int(h * target_ratio)
        assert padding_hei >= h and padding_wid >= w
        padding_top = int((padding_hei - h) / 2.)
        padding_left = int((padding_wid - w) / 2.)
        pad_img = np.zeros((padding_hei, padding_wid, c), dtype=img.dtype)
        pad_img[padding_top:padding_top + h, padding_left: padding_left + w, :] = img
        # resize
        resize_img = cv2.resize(pad_img, (self.width, self.height), interpolation=mode)
        if len(resize_img.shape) == 2:
            resize_img = resize_img[:, :, np.newaxis]
        return resize_img

    def pad_resize_kp(self, dp_x, dp_y, img_shape):
        h, w, c = img_shape
        dp_x = dp_x / 255.0 * (w - 1)
        dp_y = dp_y / 255.0 * (h - 1)
        target_ratio = self.height / float(self.width)
        bbox_ratio = h / float(w)
        padding_hei = h if bbox_ratio >= target_ratio else int(w * target_ratio)
        padding_wid = w if bbox_ratio <= target_ratio else int(h / target_ratio)
        assert (padding_hei >= h) and (padding_wid >= w)
        padding_top = int((padding_hei - h) / 2.)
        padding_left = int((padding_wid - w) / 2.)
        dp_x += padding_left
        dp_y += padding_top
        dp_x = dp_x / (padding_wid - 1) * (self.width - 1)
        dp_y = dp_y / (padding_hei - 1) * (self.height - 1)

        return dp_x, dp_y

    def read_img(self, file_name):
        if 'COCO' in file_name:
            img = Image.open(file_name).convert('RGB')
            return img
        with open(file_name, 'rb') as f:
            src_buf = f.read()
            src_len = len(src_buf)
            dst_buf = ctypes.create_string_buffer(src_len)
            dst_len = ctypes.c_int(0)
            fimg = io.BytesIO(dst_buf)
            img = Image.open(fimg).convert('RGB')
            return img

    def __len__(self):
        return self.total_bbox_num * self.repeat

    def __getitem__(self, index):
        index = index % self.total_bbox_num
        ret = {k: self.img_infos[k][index] for k in self.img_infos.keys()}
        for k, v in ret.items():
            if k in CSE_IUV_KEYS:
                ret[k] = torch.Tensor(v)
        img = self.read_img(ret['img_path'])
        # dp_clothes = extract_dp_clothes_mask(ret['dp_clothes'])
        dp_masks = extract_segmentation_mask(ret['dp_masks'])
        dp_x = ret['dp_x']
        dp_y = ret['dp_y']
        img, dp_masks, dp_x, dp_y, j_valid = self.crop_img(img, ret['bbox'], dp_masks=dp_masks,
                                                           dp_x=dp_x, dp_y=dp_y)
        if self.transform is not None:
            img = self.transform(img)
        img = np.array(img)
        img_shape = img.shape
        img = self.pad_resize_img(img, mode=cv2.INTER_LINEAR)
        # dp_clothes = self.pad_resize_img(dp_clothes, mode=cv2.INTER_NEAREST)
        dp_masks = self.pad_resize_img(dp_masks, mode=cv2.INTER_NEAREST)
        dp_x, dp_y = self.pad_resize_kp(dp_x, dp_y, img_shape)
        j_valid = j_valid & (dp_x >= 0) & (dp_x <= self.width - 1) & (dp_y >= 0) & (dp_y <= self.height - 1)
        ret['img'] = (torch.from_numpy(img).permute(2, 0, 1) - self.mean) / self.std
        ret['dp_masks_gt'] = torch.from_numpy(dp_masks)[:, :, 0]
        ret['dp_x'] = dp_x[j_valid]
        ret['dp_y'] = dp_y[j_valid]
        ret['dp_I'] = ret['dp_I'][j_valid]
        ret['dp_U'] = ret['dp_U'][j_valid]
        ret['dp_V'] = ret['dp_V'][j_valid]
        ret.pop('bbox')
        ret.pop('dp_masks')
        # ret.pop('dp_clothes')
        ret.pop('img_path')
        return ret


if __name__ == '__main__':
    test = ['./data/reid_cse/DP3D_train.json',
            './data/reid_cse/LTCC_train.json',
            './data/reid_cse/VC_Clothes_train.json',
            './data/reid_cse/PRCC_train.json',
            './data/reid_cse/VC_Clothes_train.json',
            './data/reid_cse/densepose_coco_2014_train.json'
            ]
    dataset = CSEDataset(test)
    for i in range(1000):
        img = dataset[i]

