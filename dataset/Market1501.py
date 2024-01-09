import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from reid_cse.cse_tools import CSE_ANN_KEYS, CSE_IUV_KEYS, init_from_coco_json_file, extract_segmentation_mask


class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(self, root='', cse_root='', verbose=True, **kwargs):
        super(Market1501, self).__init__()
        # self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_dir = root
        self.cse_dataset_dir = cse_root
        self.is_video = False
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.surface_corr_ori = self._load_surface_corr()
        self._check_before_run()
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        segms, dp_x, dp_y, surf_corr = self._process_surface_corr()
        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.surf_corr = surf_corr
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        self.segms, self.dp_x, self.dp_y = segms, dp_x, dp_y

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

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            img_prefix = img_path.split('.')
            mask_path = img_prefix[0] + '_vis4.png'
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, mask_path, pid, camid))

        return dataset
