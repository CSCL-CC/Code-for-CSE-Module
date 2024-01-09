import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from reid_cse.cse_tools import CSE_ANN_KEYS, CSE_IUV_KEYS, init_from_coco_json_file, extract_segmentation_mask


class LTCC_ORI(BaseImageDataset):
    dataset_dir = 'ltcc'

    def __init__(self, root='', cse_root='', verbose=True, **kwargs):
        super(LTCC_ORI, self).__init__()
        # self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_dir = root
        self.cse_root = ''
        self.is_video = False
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self.surface_corr_ori = self._load_surface_corr()

        self.id_label_list = []
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True, training=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        change_gallery = self._process_changedir(self.gallery_dir, relabel=False, name='gallery')
        change_query = self._process_changedir(self.query_dir, relabel=False, name='query')
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

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.train_cloths = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.query_cloths = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.gallery_cloths = self.get_imagedata_info(
            self.gallery)
        self.num_change_gallery_pids, self.num_change_gallery_imgs, self.num_change_gallery_cams, self.change_query_cloths = self.get_imagedata_info(
            self.change_gallery)
        self.num_change_query_pids, self.num_change_query_imgs, self.num_change_query_cams, self.change_gallery_cloths = self.get_imagedata_info(
            self.change_query)
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

    def _process_dir(self, dir_path, relabel=False, training=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, clothid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        all_clothid = 0
        clothid_set = {}
        if training:
            with open('LTCC.txt', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    prefix = line.split(' ')
                    clothid_set[prefix[0]] = int(prefix[1])

        for img_path in img_paths:
            img_prefix = img_path.split('.')
            img_path = img_prefix[0] + '.png'
            mask_path = img_prefix[0] + '_vis4.png'
            pid, clothid, camid = map(int, pattern.search(img_path).groups())
            if training:
                self.id_label_list.append(pid)
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 151  # pid == 0 means background
            assert 1 <= camid <= 12
            assert 1 <= clothid <= 100
            # camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            if training:
                all_clothid = clothid_set[osp.basename(img_path)]
            dataset.append((img_path, mask_path, pid, clothid, all_clothid, camid))
            # dataset.append((img_path, mask_path, pid, camid))

        return dataset

    def _process_changedir(self, dir_path, relabel=False, name='gallery'):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_([-\d]+)_c([-\d]+)')
        txt_dir = osp.join(self.dataset_dir, 'info', 'cloth-change_id_test.txt')
        change_id = []
        with open(txt_dir, 'r') as tx:
            ids = tx.readlines()
            for id in ids:
                change_id.append(int(id))

        pid_container = set()
        for img_path in img_paths:
            pid, clothid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            if pid not in change_id: continue  # unchange ids are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        all_clothid = 0
        for img_path in img_paths:
            img_prefix = img_path.split('.')
            img_path = img_prefix[0] + '.png'
            mask_path = img_prefix[0] + '_vis4.png'
            pid, clothid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            if pid not in change_id: continue  # unchange ids are just ignored
            assert 0 <= pid <= 151  # pid == 0 means background
            assert 1 <= camid <= 12
            assert 1 <= clothid <= 100
            # camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, mask_path, pid, clothid, all_clothid, camid))
            # dataset.append((img_path, mask_path, pid, camid))

        return dataset
