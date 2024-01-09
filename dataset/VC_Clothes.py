# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from reid_cse.cse_tools import CSE_ANN_KEYS, CSE_IUV_KEYS, init_from_coco_json_file, extract_segmentation_mask


class VC_clothes(BaseImageDataset):

    dataset_dir = 'VC_clothes'

    def __init__(self, root='', cse_root='', verbose=True, **kwargs):
        super(VC_clothes, self).__init__()
        # self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_dir = root
        self.cse_dataset_dir = cse_root
        self.is_video = False
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self.surface_corr_ori = self._load_surface_corr()

        self._check_before_run()
        self.id_label_list = []

        train = self._process_dir(self.train_dir, relabel=True,training=True)
        # query = self._process_dir(self.query_dir, relabel=False)
        # gallery = self._process_dir(self.gallery_dir, relabel=False)
        change_gallery = self._process_changedir(self.gallery_dir, relabel=False,name = 'gallery')
        gallery = change_gallery
        change_query = self._process_changedir(self.query_dir, relabel=False, name = 'query2')
        query = self._process_changedir(self.query_dir, relabel=False, name = 'query1')
        segms, dp_x, dp_y, surf_corr = self._process_surface_corr()

        if verbose:
            print("=> LTCC-Reid loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.change_gallery = change_gallery
        self.change_query = change_query
        self.surf_corr = surf_corr


        self.num_train_pids, self.num_train_imgs, self.num_train_cams,self.train_cloths = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.query_cloths = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.gallery_cloths = self.get_imagedata_info(self.gallery)
        self.num_change_gallery_pids, self.num_change_gallery_imgs, self.num_change_gallery_cams, self.change_query_cloths = self.get_imagedata_info(self.change_gallery)
        self.num_change_query_pids, self.num_change_query_imgs, self.num_change_query_cams, self.change_gallery_cloths = self.get_imagedata_info(self.change_query)
        self.segms, self.dp_x, self.dp_y = segms, dp_x, dp_y
        self.print_dataset_statistics(train, self.change_query, self.change_gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))


    def _load_surface_corr(self):
        img_infos = dict()
        for k in CSE_ANN_KEYS:
            img_infos[k] = []
        img_infos['img_path'] = []
        info = init_from_coco_json_file(self.cse_dataset_dir)
        for k, v in info.items():
            img_infos[k].extend(v)
            print('bbox num is', len(info['img_path']))
        print('total bbox num is ', len(img_infos['img_path']))
        self.total_bbox_num = len(img_infos['img_path'])
        return img_infos

    def _process_surface_corr(self):
        org = {k: self.surface_corr_ori[k] for k in self.surface_corr_ori.keys()}
        corr_dict = dict()
        for k, v in org.items():
            if k in CSE_IUV_KEYS:
                corr_dict[k] = v
        dp_masks = extract_segmentation_mask(corr_dict['dp_masks'])
        dp_x, dp_y = corr_dict['dp_x'], corr_dict['dp_y']
        return corr_dict, dp_masks, dp_x, dp_y


    def _process_dir(self, dir_path, relabel=False,training = False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)-([-\d]+)-([-\d]+)-([-\d]+)')
        clothid_set = {}
        pid_container = set()
        for img_path in img_paths:
            pid, camid, clothid, _ = map(int, pattern.search(img_path).groups())
            if not camid == 3 and not camid ==4:
                continue
            if pid not in clothid_set:
                clothid_set[pid] = clothid
            if clothid_set[pid] < clothid:
                clothid_set[pid] = clothid


            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        all_cloth = 0


        j = 0
        clothid_sets = {}
        for key,value in clothid_set.items():
            if key not in clothid_sets.keys():
                clothid_sets[key] = [-1] * value
            for m in range(value):
                clothid_sets[key][m] = j
                j += 1

        print('clo',j)
        dataset = []
        all_clothid = 0

        # if training:
        #     with open('LTCC.txt','r') as file:
        #         lines = file.readlines()
        #         for line in lines:
        #             prefix = line.split(' ')
        #             clothid_set[prefix[0]] = int(prefix[1])


        for img_path in img_paths:
            img_prefix = img_path.split('.')
            img_path = img_prefix[0] + '.jpg'
            mask_path = img_prefix[0] + '_vis.png'
            pid, camid, clothid, _ = map(int, pattern.search(img_path).groups())
            if not camid == 3 and not camid == 4:
                continue
            if training:
                self.id_label_list.append(pid)
            if pid == -1: continue  # junk images are just ignored
             # pid == 0 means background
            assert 1 <= pid <= 512
            assert 1 <= camid <= 4
            assert 1 <= clothid <= 100
            # camid -= 1  # index starts from 0
            if training:
                all_clothid = clothid_sets[pid][clothid-1]
                assert 0 <= clothid <= 618
            # camid -= 1  # index starts from
            if relabel: pid = pid2label[pid]
            if training: assert 0 <= pid <= 255

            dataset.append((img_path, mask_path, pid, clothid, all_clothid, camid))
            # dataset.append((img_path, mask_path, pid, camid))

        return dataset

    def _process_changedir(self, dir_path, relabel=False,name='gallery',training = False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)-([-\d]+)-([-\d]+)-([-\d]+)')
        clothid_set = {}
        pid_container = set()
        for img_path in img_paths:
            pid, camid, clothid, _ = map(int, pattern.search(img_path).groups())
            if pid not in clothid_set:
                clothid_set[pid] = clothid
            if clothid_set[pid] < clothid:
                clothid_set[pid] = clothid

            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        all_cloth = 0

        j = 0
        clothid_sets = {}
        for key, value in clothid_set.items():
            if key not in clothid_sets.keys():
                clothid_sets[key] = [-1] * value
            for m in range(value):
                clothid_sets[key][m] = j
                j += 1

        dataset = []
        all_clothid = 0

        # if training:
        #     with open('LTCC.txt','r') as file:
        #         lines = file.readlines()
        #         for line in lines:
        #             prefix = line.split(' ')
        #             clothid_set[prefix[0]] = int(prefix[1])

        for img_path in img_paths:
            img_prefix = img_path.split('.')
            img_path = img_prefix[0] + '.jpg'
            mask_path = img_prefix[0] + '_vis.png'
            pid, camid, clothid, _ = map(int, pattern.search(img_path).groups())
            if name == 'gallery' and (camid==3 or camid==4):
                pass
            elif name == 'query1' and (camid==2):pass
            elif name == 'query2' and (camid==1):pass
            else:continue
            # if name == 'query1' and (camid==2):
            #     pass
            # else:continue
            # if name == 'query2' and (camid==1):
            #     pass
            # else:continue
            if training:
                self.id_label_list.append(pid)
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 512  # pid == 0 means background
            assert 1 <= camid <= 12
            assert 1 <= clothid <= 100
            # camid -= 1  # index starts from 0
            if clothid_set[pid]<=1:
                continue
            if relabel: pid = pid2label[pid]
            if training:
                all_clothid = clothid_sets[pid][clothid]
            dataset.append((img_path, mask_path, pid, clothid, all_clothid, camid))
            # dataset.append((img_path, mask_path, pid, camid))

        return dataset


