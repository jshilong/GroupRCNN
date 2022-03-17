# --------------------------------------------------------
# Group R-CNN
# Copyright (c) OpenMMLab. All rights reserved.
# Written by Shilong Zhang
# --------------------------------------------------------

import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomCrop


@PIPELINES.register_module()
class PointRandomCrop(RandomCrop):
    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps. The difference with :Class:RandomCrop is this class
        add the process the point annotation.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        with_bbox_ann = results['img_info']['with_bbox_ann']
        # crop bboxes accordingly and clip to the image boundary

        num_gt = len(results['gt_labels'])
        # TODO check this aug
        for key in ['gt_bboxes']:
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            temp_bboxes = results[key] - bbox_offset

            if with_bbox_ann:
                bboxes = temp_bboxes[:num_gt]
                points = temp_bboxes[num_gt:]
                if self.bbox_clip_border:
                    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                    points[:, 0::2] = np.clip(points[:, 0::2], 0, img_shape[1])
                    points[:, 1::2] = np.clip(points[:, 1::2], 0, img_shape[0])

                valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] >
                                                              bboxes[:, 1])
                # judge is there points in bbox
                if len(points):
                    points = points[valid_inds]
                # If the crop does not contain any gt-bbox area and
                # allow_negative_crop is False, skip this image.
                if (key == 'gt_bboxes' and not valid_inds.any()
                        and not allow_negative_crop):
                    return None
                results[key] = np.concatenate([bboxes[valid_inds, :], points],
                                              axis=0)
                # label fields. e.g. gt_labels and gt_labels_ignore
                label_key = self.bbox2label.get(key)
                if label_key in results:
                    results[label_key] = results[label_key][valid_inds]

            else:
                bboxes = temp_bboxes[:num_gt]
                # point here would be dep
                # we will sample point in model instead
                # of dataset if there is bbox annotation
                points = temp_bboxes[num_gt:]
                valid_inds = (0 <= points[:, 0] ) &  ( points[:, 0]< img_shape[1]) \
                             & (0 <= points[:, 1]) & (points[:, 1]< img_shape[0]) # noqa
                if self.bbox_clip_border:
                    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                    points[:, 0::2] = np.clip(points[:, 0::2], 0, img_shape[1])
                    points[:, 1::2] = np.clip(points[:, 1::2], 0, img_shape[0])

                points = points[valid_inds]
                # If the crop does not contain any gt-bbox area and
                # allow_negative_crop is False, skip this image.
                if (key == 'gt_bboxes' and not valid_inds.any()
                        and not allow_negative_crop):
                    return None
                results[key] = np.concatenate([bboxes[valid_inds, :], points],
                                              axis=0)
                # label fields. e.g. gt_labels and gt_labels_ignore
                label_key = self.bbox2label.get(key)
                if label_key in results:
                    results[label_key] = results[label_key][valid_inds]

            return results
