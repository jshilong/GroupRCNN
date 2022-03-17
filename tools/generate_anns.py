# --------------------------------------------------------
# Group R-CNN
# Copyright (c) OpenMMLab. All rights reserved.
# Written by Shilong Zhang
# --------------------------------------------------------
import json
import sys

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmdet.datasets.api_wrappers import COCO

from projects.datasets.point_coco import local_numpy_seed

sys.path.insert(0, './')


class PointGenerator(object):
    def __init__(self, ann_file):
        self.ann_file = ann_file
        self.coco = COCO(ann_file)
        self.seed = 0

    def generate_points(self):

        save_json = dict()
        save_json['images'] = self.coco.dataset['images']
        save_json['annotations'] = []
        annotations = self.coco.dataset['annotations']
        save_json['categories'] = self.coco.dataset['categories']

        id_info = dict()
        for img_info in self.coco.dataset['images']:
            id_info[img_info['id']] = img_info
        prog_bar = mmcv.ProgressBar(len(annotations))
        with local_numpy_seed(self.seed):
            for ann in annotations:
                prog_bar.update()
                img_info = id_info[ann['image_id']]
                segm = ann.get('segmentation', None)
                if isinstance(segm, list):

                    rles = maskUtils.frPyObjects(segm, img_info['height'],
                                                 img_info['width'])
                    rle = maskUtils.merge(rles)
                elif isinstance(segm['counts'], list):
                    # uncompressed RLE
                    rle = maskUtils.frPyObjects(segm, img_info['height'],
                                                img_info['width'])
                else:
                    # rle
                    rle = segm
                mask = maskUtils.decode(rle)
                if mask.sum() > 0:
                    ys, xs = np.nonzero(mask)
                    point_idx = np.random.randint(len(xs))
                    x1 = int(xs[point_idx])
                    y1 = int(ys[point_idx])
                    ann['point'] = [x1, y1, x1, y1]
                else:
                    x1, y1, w, h = ann['bbox']
                    x1 = np.random.uniform(x1, x1 + w)
                    y1 = np.random.uniform(y1, y1 + h)
                    ann['point'] = [x1, y1, x1, y1]

                save_json['annotations'].append(ann)
        mmcv.mkdir_or_exist('./point_ann/')
        ann_name = self.ann_file.split('/')[-1]
        with open(f'./point_ann/{ann_name}', 'w') as f:
            json.dump(save_json, f)


if __name__ == '__main__':

    args = sys.argv
    if len(args) > 1:
        ann_file = args[1]

    point_generator = PointGenerator(ann_file=ann_file)
    point_generator.generate_points()
