# --------------------------------------------------------
# Group R-CNN
# Copyright (c) OpenMMLab. All rights reserved.
# Written by Shilong Zhang
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import batched_nms
from mmdet.core import bbox2result, select_single_mlvl
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector


@DETECTORS.register_module()
class GroupRCNN(TwoStageDetector):
    def __init__(self,
                 *args,
                 pre_topk=3,
                 rpn_nms_topk=50,
                 rpn_iou=0.7,
                 roi_iou_threshold=0.5,
                 num_projection_convs=1,
                 **kwargs):
        """Group R-CNN.

        Args:
            pre_topk (int): Select nearest `pre_topk` points to form a
                group. Default to 3.
            rpn_nms_topk (int): The number of bboxes reserved each group.
                Default to 50.
            rpn_iou (float): The IOU Threshold when do the nms of RPN
                proposals. Default to 0.7.
            roi_iou_threshold (float): The IOU Threshold when do the nms of RPN
                proposals. Default to 0.5.
            num_projection_convs (int): The number of projection convs. Default
                to 1.
        """
        self.pre_topk = pre_topk
        self.rpn_iou = rpn_iou
        self.rpn_nms_topk = rpn_nms_topk
        super(GroupRCNN, self).__init__(*args, **kwargs)
        self.roi_head.iou_threshold = roi_iou_threshold
        self.num_projection_convs = num_projection_convs
        self.projection_convs = nn.ModuleList()
        for _ in range(self.num_projection_convs):
            self.projection_convs.append(
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=3,
                          padding=1))

    def splte_preds(self, mlvl_pred, bbox_flag):
        point_flag = ~bbox_flag

        mlvl_normanl_pred = [item[bbox_flag] for item in mlvl_pred]
        mlvl_semi_pred = [item[point_flag] for item in mlvl_pred]

        return mlvl_normanl_pred, mlvl_semi_pred

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        gt_bboxes, gt_points = self.process_gts(gt_bboxes, gt_labels)

        x = self.extract_feat(img)

        losses = dict()
        feat_sizes = [item.size()[-2:] for item in x]
        mlvl_points = self.gen_points(feat_sizes,
                                      dtype=x[0].dtype,
                                      device=x[0].device)

        # used in roi_head
        rela_coods_list = self.get_relative_coordinate(mlvl_points, gt_points)
        mlti_assign_results = self.point_assign(mlvl_points, gt_points)
        rpn_losses, results_list = self.rpn_forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            assign_results=mlti_assign_results)
        losses.update(rpn_losses)

        pred_bboxes, pred_scores = self._rpn_post_process(results_list,
                                                          gt_labels=gt_labels)

        detached_x = [item.detach() for item in x]
        for conv in self.projection_convs:
            detached_x = [F.relu(conv(item)) for item in detached_x]

        roi_losses = self.roi_head.forward_train(
            detached_x,
            img_metas,
            pred_bboxes,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            rela_coods_list=rela_coods_list,
            gt_points=gt_points,
            **kwargs)
        losses.update(roi_losses)
        return losses

    def _rpn_post_process(
        self,
        pred_bboxes,
        gt_labels,
    ):
        """Do nms cross group and reserved `pre_topk` for each Group.

        Args:
            pred_bboxes (list[Tensor]): Regression predictions
                from RPN. Each has shape (group_size, num_gts, 4)
            gt_labels: (list[Tensor]): Gt_labels for multiple images.
                Each has shape (num_gts,)

        Returns:
            tuple[tensor]:

                - temp_all_pred_bboxes (list[tensor]): Bboxes
                  after post process. Each has shape
                  (rpn_nms_topk, num_gts, 4)
                - all_pred_scores (list[tensor]): Scores of
                  each bboxes. Each tensor in the list has shape
                  (rpn_nms_topk, num_gts)
        """
        temp_all_pred_bboxes = []
        all_pred_scores = []
        for img_id, (bboxes, scores) in enumerate(pred_bboxes):
            bag_size = bboxes.shape[0]
            repeat_label = gt_labels[img_id][None].repeat(bag_size, 1)

            scores = torch.gather(scores, 2, repeat_label[...,
                                                          None]).squeeze(-1)
            num_gt = bboxes.shape[1]

            if num_gt == 0:
                temp_all_pred_bboxes.append(scores.new_zeros(bag_size, 0, 5))
                all_pred_scores.append(scores.new_zeros(bag_size, 0))
                continue
            dets_with_score, keep = batched_nms(
                bboxes.view(-1, 4), scores.view(-1), repeat_label.view(-1),
                dict(max_num=1000, iou_threshold=self.rpn_iou))

            num_pred = len(keep)
            gt_index = keep % num_gt
            arrange_gt_index = torch.arange(num_gt, device=keep.device)[:,
                                                                        None]
            # num_gt x num_pred
            keep_matrix = gt_index == arrange_gt_index
            temp_index = torch.arange(-num_pred,
                                      end=0,
                                      step=1,
                                      device=keep.device)
            keep_matrix = keep_matrix * temp_index
            rpn_nms_topk = min(num_pred, self.rpn_nms_topk)
            value_, index = keep_matrix.topk(rpn_nms_topk,
                                             dim=-1,
                                             largest=False)
            index = index.view(-1)
            dets_with_score = dets_with_score[index]

            num_pad = self.rpn_nms_topk - rpn_nms_topk
            padding = dets_with_score.new_zeros(num_gt, num_pad, 5)

            dets_with_score = dets_with_score.view(num_gt, rpn_nms_topk, 5)
            dets_with_score = torch.cat([dets_with_score, padding], dim=1)

            dets = dets_with_score[..., :4]
            det_scores = dets_with_score[..., 4]
            # topk * num_gt * 4
            dets = dets.permute(1, 0, 2)
            det_scores = det_scores.permute(1, 0)
            temp_all_pred_bboxes.append(dets.contiguous())
            all_pred_scores.append(det_scores.contiguous())

        return temp_all_pred_bboxes, all_pred_scores

    def process_gts(self, gt_bboxes=None, gt_labels=None):
        """Split point annotations from `gt_bboxes`. We concatenate point
        positions to bbox annotations in dataset.

        Args:
            gt_bboxes (list[Tensor]): Bbox annotations
                concatenated with point annotations of \
                multiple images.
                Each tensor has shape (2 * num_intances, 4)
            gt_labels (list[Tensor]): Label of multiple
                images. Each has shape (num_instances,)

        Returns:
            tuple[List[Tensor]]: Bbox annotations and
                point annotations.
        """
        gt_points = []
        new_gt_bboxes = []
        for img_id in range(len(gt_bboxes)):
            num_gt = len(gt_labels[img_id])
            new_gt_bboxes.append(gt_bboxes[img_id][:num_gt])
            gt_points.append(gt_bboxes[img_id][num_gt:][:, :2])
        return new_gt_bboxes, gt_points

    def get_relative_coordinate(self, mlvl_points, points_list):
        """Calculate relative coordinates for each instance.

        Args:
            mlvl_points (list[Tensor]): Coordinates for multiple
                level of FPN. Each has shape (h, w, 2).
            points_list (list[Tensor]): Point annotations for
                multiple images. Each has shape (num_instances, 2)

        Returns:
            list[Tensor]: Relative coordinates for each instances.
                Each has shape (num_instances, h ,w, 2).
        """
        real_coord_list = []
        for img_id, single_img_points in enumerate(points_list):
            mlvl_real_coord = []
            gt_points = points_list[img_id]
            for level in range(len(self.strides)):
                feat_points = mlvl_points[level]
                real_coods = gt_points[:, None, :] - feat_points
                if isinstance(self.strides[level], int):
                    temp_stride = self.strides[level]
                else:
                    temp_stride = self.strides[level][0]
                real_coods = real_coods.float() / temp_stride
                # num_gt * h * w * 2
                mlvl_real_coord.append(real_coods)
            real_coord_list.append(mlvl_real_coord)
        return real_coord_list

    def rpn_forward_train(self,
                          x,
                          img_metas,
                          gt_bboxes,
                          gt_labels=None,
                          gt_bboxes_ignore=None,
                          assign_results=None,
                          **kwargs):
        outs = self.rpn_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        rpn_losses = self.rpn_head.loss(*loss_inputs,
                                        gt_bboxes_ignore=gt_bboxes_ignore)

        results_list = self._rpn_get_bboxes(outs, img_metas, assign_results,
                                            gt_labels)

        return rpn_losses, results_list

    def _rpn_get_bboxes(self,
                        outs,
                        img_metas=None,
                        assign_results=None,
                        gt_labels=None):
        with torch.no_grad():
            if len(outs) == 2:
                cls_scores, bbox_preds = outs
                with_score_factors = False
            else:
                cls_scores, bbox_preds, score_factors = outs
                with_score_factors = True
                assert len(cls_scores) == len(score_factors)
            num_levels = len(cls_scores)
            results_list = []
            for img_id in range(len(img_metas)):
                img_meta = img_metas[img_id]
                gt_label = gt_labels[img_id]
                assign_result = assign_results[img_id]

                mlvl_cls_score = select_single_mlvl(cls_scores, img_id)
                mlvl_bbox_pred = select_single_mlvl(bbox_preds, img_id)
                if with_score_factors:
                    mlvl_score_factor = select_single_mlvl(
                        score_factors, img_id)
                else:
                    mlvl_score_factor = [None for _ in range(num_levels)]
                results = self._get_dummy_bboxes_single(
                    mlvl_cls_score,
                    mlvl_bbox_pred,
                    mlvl_score_factor,
                    img_meta,
                    gt_label=gt_label,
                    assign_result=assign_result,
                    img_id=img_id,
                )
                results_list.append(results)
        return results_list

    def repeat_index(self, asssign_results, gt_labels):
        """Each point has several anchors.

        We repeat the index to the indices of corresponding bboxes
        """
        num_base_priors = self.rpn_head.num_base_priors
        reapeated_indexs = []
        if num_base_priors > 1:
            # topk * num_gt
            for single_lvl_results in asssign_results:
                temp_list = [
                    single_lvl_results * num_base_priors + i
                    for i in range(num_base_priors)
                ]
                repeated_sinlge_lvl_results = torch.cat(temp_list, dim=0)
                reapeated_indexs.append(repeated_sinlge_lvl_results.view(-1))
        else:
            reapeated_indexs = [item.view(-1) for item in asssign_results]
        repeat_labels = gt_labels[None].repeat(self.pre_topk * num_base_priors,
                                               1)
        repeat_labels = repeat_labels.view(-1)

        return reapeated_indexs, repeat_labels

    def _get_dummy_bboxes_single(self,
                                 mlvl_cls_score,
                                 mlvl_bbox_pred,
                                 mlvl_score_factor,
                                 img_meta,
                                 gt_label=None,
                                 assign_result=None,
                                 **kwargs):
        img_shape = img_meta['img_shape']
        num_gt = len(gt_label)
        assert num_gt == assign_result[0].shape[1]
        # avoid empty tensor
        shape_0 = self.pre_topk * self.rpn_head.num_base_priors
        assign_result, repeat_label = self.repeat_index(
            assign_result, gt_label)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_priors = []

        for level_idx, (cls_score, bbox_pred, score_factor,
                        single_lvl_pos_index) in enumerate(
                            zip(mlvl_cls_score, mlvl_bbox_pred,
                                mlvl_score_factor, assign_result)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            featmap_size_hw = cls_score.shape[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.rpn_head.cls_out_channels)
            if self.rpn_head.loss_cls.use_sigmoid:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            priors = self.rpn_head.prior_generator.sparse_priors(
                single_lvl_pos_index, featmap_size_hw, level_idx, scores.dtype,
                scores.device)

            bbox_pred = bbox_pred[single_lvl_pos_index, :]
            scores = scores[single_lvl_pos_index, :]
            bboxes = self.rpn_head.bbox_coder.decode(priors,
                                                     bbox_pred,
                                                     max_shape=img_shape)
            mlvl_bboxes.append(bboxes.view(shape_0, num_gt, 4))
            mlvl_priors.append(priors.view(shape_0, num_gt, 4))
            mlvl_scores.append(
                scores.view(shape_0, num_gt, self.rpn_head.cls_out_channels))

        mlgt_bboxes = torch.cat(mlvl_bboxes, dim=0)
        mlgt_scores = torch.cat(mlvl_scores, dim=0)
        returns_list = [mlgt_bboxes, mlgt_scores]
        return returns_list

    def gen_points(
        self,
        featmap_sizes,
        dtype,
        device,
        flatten=True,
    ):
        """Generate coordinates for FPN.

        Args:
            featmap_sizes (list[tensor]): (h ,w) of each level
                of FPN.
            dtype (torch.dtype): Type of generated coordinates.
            device (torch.device): Device of generated coordinates.
            flatten (bool): Whether to flatten the (h, w) dimensions.

        Returns:
            list[tensor]: Coordinates of each level.
        """

        self.strides = self.rpn_head.prior_generator.strides

        def _get_points_single(featmap_size,
                               stride,
                               dtype,
                               device,
                               flatten=False,
                               offset=0.0):
            """Get points of a single scale level."""
            if isinstance(stride, (tuple, list)):
                stride = stride[0]
            h, w = featmap_size
            # First create Range with the default dtype, than convert to
            # target `dtype` for onnx exporting.
            x_range = torch.arange(w, device=device).to(dtype)
            y_range = torch.arange(h, device=device).to(dtype)
            y, x = torch.meshgrid(y_range, x_range)
            if flatten:
                y = y.flatten()
                x = x.flatten()

            points = torch.stack(
                (x * stride, y * stride), dim=-1) + stride * 0.0
            return points

        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                _get_points_single(featmap_sizes[i], self.strides[i], dtype,
                                   device, flatten))

        return mlvl_points

    def point_assign(self, mlvl_points, gt_points):
        """Select the nearest `pre_topk` points for each point annotation.

        Args:
            mlvl_points (list[tensor]): Coordinates of each FPN
                level.
            gt_points (list[tensor]): Points annotations for
                multiple images.

        Returns:
            list[list[tensor]]: The outer list indicate the
                images. Inter list indicate the FPN level.
                And the tensor id the index of selected
                feature points. Each tensor has shape
                (pre_topk, num_instances)
        """
        def nearest_k(mlvl_points, gt_points, pre_topk=3):

            mlvl_prior_index = []
            for points in mlvl_points:
                distances = (points[:, None, :] -
                             gt_points[None, :, :]).pow(2).sum(-1)
                min_pre_topk = min(len(distances), pre_topk)
                _, topk_idxs_per_level = distances.topk(min_pre_topk,
                                                        dim=0,
                                                        largest=False)
                mlvl_prior_index.append(topk_idxs_per_level)
            return mlvl_prior_index

        mlti_img_assign = []
        for single_img_gt_points in gt_points:
            mlti_img_assign.append(
                nearest_k(mlvl_points, single_img_gt_points, self.pre_topk))
        return mlti_img_assign

    def simple_test(self,
                    img,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'

        gt_labels = kwargs.get('gt_labels', [])
        # remove aug
        gt_labels = [item[0] for item in gt_labels]
        gt_bboxes = kwargs.get('gt_bboxes', [])
        gt_bboxes = [item[0] for item in gt_bboxes]
        gt_bboxes, gt_points = self.process_gts(gt_labels=gt_labels,
                                                gt_bboxes=gt_bboxes)

        x = self.extract_feat(img)

        outs = self.rpn_head.forward(x)
        rpn_results_list = self.rpn_head.get_bboxes(*outs,
                                                    img_metas=img_metas,
                                                    rescale=True)

        feat_sizes = [item.size()[-2:] for item in outs[0]]
        mlvl_points = self.gen_points(feat_sizes,
                                      dtype=outs[0][0].dtype,
                                      device=outs[0][0].device)
        rela_coods_list = self.get_relative_coordinate(mlvl_points, gt_points)

        mlti_assign_results = self.point_assign(mlvl_points, gt_points)

        all_pred_results = self._rpn_get_bboxes(
            outs,
            img_metas=img_metas,
            assign_results=mlti_assign_results,
            gt_labels=gt_labels)
        all_pred_bboxes, all_pred_scores = self._rpn_post_process(
            all_pred_results, gt_labels=gt_labels)

        group_resutls_list = []
        for pred_bboxes, pred_scores, pred_label, img_meta \
                in zip(all_pred_bboxes, all_pred_scores, gt_labels, img_metas):
            pred_bboxes = torch.cat([pred_bboxes, pred_scores[..., None]],
                                    dim=-1)
            bag_size = len(pred_bboxes)
            pred_label = pred_label[None].repeat(bag_size, 1)
            pred_bboxes = pred_bboxes.view(-1, 5)
            pred_label = pred_label.view(-1)
            scale_factors = pred_bboxes.new_tensor(img_meta['scale_factor'])
            pred_bboxes[:, :4] = pred_bboxes[:, :4] / scale_factors
            group_resutls_list.append((pred_bboxes, pred_label))

        if len(gt_labels[0]) > 0:

            for conv in self.projection_convs:
                x = [F.relu(conv(item)) for item in x]

            extra_gts, scores, gt_labels = self.roi_head.simple_test(
                x,
                all_pred_bboxes,
                img_metas,
                rela_coods_list=rela_coods_list,
                labels=gt_labels,
                all_pred_scores=all_pred_scores,
                gt_points=gt_points,
            )
            roi_results_list = []
            for img_id, (bboxes, score, img_meta) in enumerate(
                    zip(extra_gts, scores, img_metas)):
                scale_factors = bboxes.new_tensor(img_meta['scale_factor'])
                bboxes = bboxes / scale_factors
                roi_results_list.append((torch.cat([bboxes, score[:, None]],
                                                   dim=-1), gt_labels[img_id]))
        else:
            roi_results_list = [(torch.zeros(0, 5), gt_labels[0])]

        return self.encode_results(rpn_results_list, group_resutls_list,
                                   roi_results_list)

    def encode_results(self, rpn_results_list, group_resutls_list,
                       roi_results_list):

        main_results = [
            bbox2result(det_bboxes, det_labels, 80)
            for det_bboxes, det_labels in rpn_results_list
        ]
        rpn_results = [
            bbox2result(det_bboxes, det_labels, 80)
            for det_bboxes, det_labels in group_resutls_list
        ]

        semi_results = [
            bbox2result(det_bboxes, det_labels, 80)
            for det_bboxes, det_labels in roi_results_list
        ]

        results = [(main_results[img_id], rpn_results[img_id],
                    semi_results[img_id])
                   for img_id in range(len(main_results))]

        return results
