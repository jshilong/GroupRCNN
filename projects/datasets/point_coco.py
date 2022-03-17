# --------------------------------------------------------
# Group R-CNN
# Copyright (c) OpenMMLab. All rights reserved.
# Written by Shilong Zhang
# --------------------------------------------------------

import itertools
import json
import logging
import os.path as osp
import warnings
from collections import OrderedDict, defaultdict
from contextlib import contextmanager

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.api_wrappers import COCOeval
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.pipelines import Compose
from mmdet.utils import get_root_logger
from terminaltables import AsciiTable


@contextmanager
def local_numpy_seed(seed=None):
    """Run numpy codes with a local random seed.

    If seed is None, the default random state will be used.
    """
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


INF = 1e8


@DATASETS.register_module()
class PointCocoDataset(CocoDataset):
    def __init__(self,
                 *args,
                 seed=0,
                 full_ann_ratio=0.1,
                 student_thr=0,
                 predictions_path=None,
                 need_points=True,
                 **kwargs):
        """Dataset of COCODataset with point annotations.

        Args:
            seed (int): The seed to split the dataset. Default: 0.
            full_ann_ratio (float): The ratio of images with bbox annotations
                used in traning. Default 0.1.
            student_thr (float): The threshold of predictions. Default: 0.
            predictions_path (str): Path of json that contains predictions
                of Group R-CNN. Defaults to None.
            need_points (bool): Control whether return the point annotations.
                Defaults to True.
        """
        self.student_thr = student_thr
        self.seed = seed

        # `predictions_path` Only be used when finetune the student
        # model, we replace the
        # corresponding bbox annotation with the predictions of
        # Group R-CNN
        self.predictions_path = predictions_path

        self.need_points = need_points
        if self.predictions_path:
            with open(self.predictions_path, 'r') as f:
                student_dataset = json.load(f)

            self.student_anns = defaultdict(list)
            for item in student_dataset:
                self.student_anns[item['image_id']].append(item)

        self.full_ann_ratio = full_ann_ratio

        self.super_init(*args, **kwargs)
        if self.test_mode:
            self.full_ann_ratio = 0
        self.split_data(self.data_infos)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def super_init(self,
                   ann_file,
                   pipeline,
                   classes=None,
                   data_root=None,
                   img_prefix='',
                   seg_prefix=None,
                   proposal_file=None,
                   test_mode=False,
                   filter_empty_gt=True,
                   **kwargs):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def split_data(self, data_infos):
        """Split the dataset to two part,with bbox annotations and Point
        Annotations.

        Args:
            data_infos (list[dict]): List of image infos.
        """
        self._total_length = len(data_infos)
        self._bbox_length = int(self.full_ann_ratio * self._total_length)
        self._point_length = self._total_length - self._bbox_length

        for img_info in self.data_infos:
            img_info['with_bbox_ann'] = False

        with local_numpy_seed(self.seed):
            self.bbox_ann_img_idxs = np.random.choice(self._total_length,
                                                      size=self._bbox_length,
                                                      replace=False)
        for img_id in self.bbox_ann_img_idxs:
            img_info = self.data_infos[img_id]
            img_info['with_bbox_ann'] = True

    def __len__(self):
        if self.test_mode or self.predictions_path is not None:
            return self._total_length
        else:
            return self._bbox_length

    def get_ann_info(self, idx, is_bbox_ann=False):

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        parsed_anns = self._parse_ann_info(self.data_infos[idx],
                                           ann_info,
                                           is_bbox_ann=is_bbox_ann)
        return parsed_anns

    def get_predictions_ann_info(self, idx):
        """Replace real bbox annotations with predictions. Only be used when
        finetune a Student.

        Args:
            idx (int): the index of data_info/

        Returns:
            dict: A dict contains predictions of Group R-CNN
        """
        img_id = self.data_infos[idx]['id']
        anns = self.student_anns[img_id]

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []

        for ann in anns:
            score = ann['score']
            if score > self.student_thr:
                gt_bboxes.append(ann['bbox'])
                gt_labels.append(self.cat2label[ann['category_id']])
            else:
                # TODO point ignore
                gt_bboxes_ignore.append(ann['bbox'])
                gt_labels_ignore.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            # ccwh to xxyy
            gt_bboxes[:, 2:] = gt_bboxes[:, 2:] + gt_bboxes[:, :2]
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if len(gt_bboxes) == 0:
            return None

        ann_dict = dict(bboxes=gt_bboxes, labels=gt_labels)

        return ann_dict

    def prepare_img_with_predictions(self, idx):
        """Get annotations of image when finetune a student. We replace
        original bbox annotations with predictions of Group R-CNN to if the
        images is not in division with bbox annotations.

        Args:
            idx (int): The index of corresponding image.

        Returns:
            dict: dict contains annotations.
        """
        if idx in self.bbox_ann_img_idxs:
            img_info = self.data_infos[idx]
            ann_info = self.get_ann_info(idx)
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results)
        else:
            # use predictions
            img_info = self.data_infos[idx]
            ann_info = self.get_predictions_ann_info(idx)
            # empty image
            if ann_info is None:
                return self.prepare_img_with_predictions((idx + 1) % len(self))
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results)

        return self.pipeline(results)

    def prepare_train_img(self, idx, is_bbox_ann=False):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx, is_bbox_ann=is_bbox_ann)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        # load point annotation when test
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _parse_ann_info(self, img_info, ann_info, is_bbox_ann=False):
        """Parse the annotations to a dict. We random sample a point from bbox
        if the image is in the division with bbox annotations.

        Args:
            img_info (dict): Dict contains the image information.
            ann_info (dict): Dict contains the annotations.
            is_bbox_ann (bool): Whether image is in the division that
                is with bbox annotations.

        Returns:
            dict: A dict contains point annotations and bbox annotations.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_points = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

                # only return bbox annotation when train the student model
                if self.predictions_path is None:
                    # use points in json
                    if not is_bbox_ann:
                        x1, y1 = ann['point'][:2]
                        gt_points.append([x1, y1, x1 + 1, y1 + 1])

                    # random sample points in bbox
                    else:
                        # follow point as query
                        x1 = np.random.uniform(x1 + 0.01 * w, x1 + 0.99 * w)
                        y1 = np.random.uniform(y1 + 0.01 * h, y1 + 0.99 * h)
                        # + 1 for vis, only x1, y1 would be actually used
                        gt_points.append([x1, y1, x1 + 1, y1 + 1])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            if not self.predictions_path:
                gt_points = np.array(gt_points, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            if not self.predictions_path:
                gt_points = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        if not self.predictions_path and self.need_points:
            gt_bboxes = np.concatenate([gt_bboxes, gt_points], axis=0)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
        )

        return ann

    def __getitem__(self, idx):

        # val the model with point ann in val dataset
        if self.test_mode:
            return self.prepare_test_img(idx)

        # train the student model
        if self.predictions_path:
            return self.prepare_img_with_predictions(idx)

        while True:
            bbox_ann_img_idx = self.bbox_ann_img_idxs[idx]
            # random sample points from bbox
            bbox_data = self.prepare_train_img(bbox_ann_img_idx,
                                               is_bbox_ann=True)

            if bbox_data is None:
                idx = (idx + 1) % len(self)
                continue
            else:
                bbox_data['img_metas'].data['mode'] = 'bbox'
                break
        # TODO remove this
        point_datas = []
        point_datas.append(bbox_data)

        return point_datas

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):

        if logger is None:
            logger = get_root_logger()
        if isinstance(results[0], tuple):
            main_results = [item[0] for item in results]
            rpn_results = [item[1] for item in results]
            semi_results = [item[2] for item in results]
            semi = True
        else:
            main_results = results
            semi_results = None
            rpn_results = None
            semi = False
        eval_results = OrderedDict()
        prefix = ['rpn', 'nms_rpn_topk', 'rcnn']
        for i, results in enumerate([main_results, rpn_results, semi_results]):
            if i == 1:
                continue
            metrics = metric if isinstance(metric, list) else [metric]
            allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
            for metric in metrics:
                if metric not in allowed_metrics:
                    raise KeyError(f'metric {metric} is not supported')
            if iou_thrs is None:
                iou_thrs = np.linspace(.5,
                                       0.95,
                                       int(np.round((0.95 - .5) / .05)) + 1,
                                       endpoint=True)
            if metric_items is not None:
                if not isinstance(metric_items, list):
                    metric_items = [metric_items]

            result_files, tmp_dir = self.format_results(
                results, jsonfile_prefix)

            cocoGt = self.coco
            for metric in metrics:
                msg = f'Evaluating {metric}...'
                if logger is None:
                    msg = '\n' + msg
                print_log(msg, logger=logger)

                if metric == 'proposal_fast':
                    ar = self.fast_eval_recall(results,
                                               proposal_nums,
                                               iou_thrs,
                                               logger='silent')
                    log_msg = []
                    for i, num in enumerate(proposal_nums):
                        eval_results[f'AR@{num}'] = ar[i]
                        log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                    log_msg = ''.join(log_msg)
                    print_log(log_msg, logger=logger)
                    continue

                iou_type = 'bbox' if metric == 'proposal' else metric
                if metric not in result_files:
                    raise KeyError(f'{metric} is not in results')
                try:
                    predictions = mmcv.load(result_files[metric])
                    if iou_type == 'segm':

                        for x in predictions:
                            x.pop('bbox')
                        warnings.simplefilter('once')
                        warnings.warn(
                            'The key "bbox" is deleted for more '
                            'accurate mask AP '
                            'of small/medium/large instances '
                            'since v2.12.0. This '
                            'does not change the '
                            'overall mAP calculation.', UserWarning)
                    cocoDt = cocoGt.loadRes(predictions)
                except IndexError:
                    print_log(
                        'The testing results of the whole dataset is empty.',
                        logger=logger,
                        level=logging.ERROR)
                    break

                cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
                cocoEval.params.catIds = self.cat_ids
                cocoEval.params.imgIds = self.img_ids
                cocoEval.params.maxDets = list(proposal_nums)
                cocoEval.params.iouThrs = iou_thrs
                # mapping of cocoEval.stats
                coco_metric_names = {
                    'mAP': 0,
                    'mAP_50': 1,
                    'mAP_75': 2,
                    'mAP_s': 3,
                    'mAP_m': 4,
                    'mAP_l': 5,
                    'AR@100': 6,
                    'AR@300': 7,
                    'AR@1000': 8,
                    'AR_s@1000': 9,
                    'AR_m@1000': 10,
                    'AR_l@1000': 11
                }
                if metric_items is not None:
                    for metric_item in metric_items:
                        if metric_item not in coco_metric_names:
                            raise KeyError(
                                f'metric item {metric_item} is not supported')

                if metric == 'proposal':
                    cocoEval.params.useCats = 0
                    cocoEval.evaluate()
                    cocoEval.accumulate()
                    cocoEval.summarize()
                    if metric_items is None:
                        metric_items = [
                            'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                            'AR_m@1000', 'AR_l@1000'
                        ]

                    for item in metric_items:
                        val = float(
                            f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                        eval_results[item] = val
                else:
                    cocoEval.evaluate()
                    cocoEval.accumulate()
                    cocoEval.summarize()
                    if classwise:  # Compute per-category AP
                        # Compute per-category AP
                        # from https://github.com/facebookresearch/detectron2/
                        precisions = cocoEval.eval['precision']
                        # precision: (iou, recall, cls, area range, max dets)
                        assert len(self.cat_ids) == precisions.shape[2]

                        results_per_category = []
                        for idx, catId in enumerate(self.cat_ids):
                            # area range index 0: all area ranges
                            # max dets index -1: typically 100 per image
                            nm = self.coco.loadCats(catId)[0]
                            precision = precisions[:, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            results_per_category.append(
                                (f'{nm["name"]}', f'{float(ap):0.3f}'))

                        num_columns = min(6, len(results_per_category) * 2)
                        results_flatten = list(
                            itertools.chain(*results_per_category))
                        headers = ['category', 'AP'] * (num_columns // 2)
                        results_2d = itertools.zip_longest(*[
                            results_flatten[i::num_columns]
                            for i in range(num_columns)
                        ])
                        table_data = [headers]
                        table_data += [result for result in results_2d]
                        table = AsciiTable(table_data)
                        print_log('\n' + table.table, logger=logger)

                    if metric_items is None:
                        metric_items = [
                            'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m',
                            'mAP_l'
                        ]

                    for metric_item in metric_items:
                        key = f'{prefix[i]}_{metric}_{metric_item}'
                        val = float(
                            f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'  # noqa
                        )
                        eval_results[key] = val
                    ap = cocoEval.stats[:6]
                    eval_results[f'{prefix[i]}_{metric}_mAP_copypaste'] = (
                        f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                        f'{ap[4]:.3f} {ap[5]:.3f}')
            if not semi:
                break
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
