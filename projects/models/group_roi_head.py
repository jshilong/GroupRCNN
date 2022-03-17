# --------------------------------------------------------
# Group R-CNN
# Copyright (c) OpenMMLab. All rights reserved.
# Written by Shilong Zhang
# --------------------------------------------------------
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32
from mmdet.core import bbox2roi, bbox_overlaps
from mmdet.models.builder import HEADS
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class GroupRoIHead(CascadeRoIHead):
    def __init__(self, *args, pos_iou_thrs=[0.5, 0.6, 0.7], **kwargs):
        # Used to assign the label
        self.pos_iou_thrs = pos_iou_thrs
        super(GroupRoIHead, self).__init__(*args, **kwargs)
        self._init_dy_groupconv()

    def _init_dy_groupconv(self):
        # fusing the relative coordinates feature with norm proposal pooling
        # features
        self.compress_feat = nn.Conv2d(258, 256, 3, stride=1, padding=1)
        self.cls_embedding = nn.Embedding(80, 256)
        # 1 is the kernel size
        self.generate_params = nn.ModuleList([
            nn.Linear(256, 1 * 1 * 256 * 256) for _ in range(self.num_stages)
        ])
        self.avg_pool = nn.AvgPool2d((7, 7))
        # fusing the mean roi features and category embedding
        self.compress = nn.Linear(256 + 256, 256)
        self.group_norm = nn.GroupNorm(32, 256)

    def _bbox_forward(self,
                      stage,
                      x,
                      rois,
                      coord_feats,
                      group_size=None,
                      rois_per_image=None,
                      gt_labels=None,
                      **kwargs):
        """Box head forward function used in both training and testing.

        Args:
            stage (int): The index of stage.
            x (list[Tensor]): FPN Features.
            rois (Tensor): Has shape (num_proposals, 5).
            coord_feats (Tensor): Pooling feature from relative coordinates
                feature map. Has shape (num_proposals, 2, 7, 7).
            group_size (list[int]): Size of instance group. Default to None.
            rois_per_image (list[int]): Number of proposals of each image.
                Default to None.
            gt_labels (list[Tensor]): Gt labels of multiple images.
                Default to None.

        Returns:
            dict: dict of model predictions.
        """
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        bbox_feats = torch.cat([bbox_feats, coord_feats], dim=1)
        # compress from 258 to 256
        bbox_feats = self.compress_feat(bbox_feats)
        params = \
            self.generate_conv_params(bbox_feats,
                                      rois_per_image,
                                      group_size,
                                      gt_labels,
                                      stage=stage)
        bag_size = group_size[0]
        num_all_gt = int(bbox_feats.size(0) / bag_size)
        bbox_feats = bbox_feats.view(bag_size, num_all_gt * 256, 7, 7)
        bbox_feats = F.conv2d(bbox_feats,
                              params,
                              stride=1,
                              padding=0,
                              groups=num_all_gt)
        bbox_feats = bbox_feats.view(bag_size * num_all_gt, 256, 7, 7)
        # stable the training
        bbox_feats = self.group_norm(bbox_feats)

        bbox_feats = F.relu(bbox_feats)
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(cls_score=cls_score,
                            bbox_pred=bbox_pred,
                            bbox_feats=bbox_feats)
        return bbox_results

    def generate_conv_params(
        self,
        rois_feats,
        rois_per_image,
        group_size,
        labels,
        stage=0,
    ):
        """Generate parameters for dynamic group conv.

        Args:
            rois_feats (tensor): Pooling feature from FPN. Has shape
                 (num_proposals, 256, 7 , 7).
            rois_per_image (list[int]): Number of proposals of each
                image.
            group_size (list[int]): Instance group size of each image.
            labels (list[tensor]):  Gt labels for multiple images.
            stage (int): Index of stage.

        Returns:
            Tensor: Parameters for dynamic group conv. Has shape
                (num_gts * 256, 256, 1, 1), which arrange as
                (C_in, C_out, kernel_size, Kernel_size).
        """
        start = 0
        param_list = []
        ori_pool_rois_feats = self.avg_pool(rois_feats).squeeze()

        for img_id in range(len(labels)):
            num_rois = rois_per_image[img_id]
            end = num_rois + start
            pool_rois_feats = ori_pool_rois_feats[start:end]
            start = end
            bag_size = group_size[img_id]
            label = labels[img_id]
            num_gt = len(label)
            bag_embeds = self.cls_embedding.weight[label]
            pool_rois_feats = pool_rois_feats.view(bag_size, num_gt, 256)
            pool_rois_feats = pool_rois_feats.mean(0)
            pool_rois_feats = torch.cat([bag_embeds, pool_rois_feats], dim=-1)
            pool_rois_feats = self.compress(pool_rois_feats)

            params = self.generate_params[stage](pool_rois_feats)
            # use group conv
            conv_weight = params.view(num_gt, 256, 256, 1, 1)
            conv_weight = conv_weight.reshape(num_gt * 256, 256, 1, 1)
            # num_gt *
            param_list.append(conv_weight)
        params = torch.cat(param_list, dim=0)

        return params

    def _first_coord_pooling(self, coord_feats, proposal_list):
        """Pooling relative coordinates for first stage."""
        rois_list = []
        start_index = 0

        for img_id, bag_bboxes in enumerate(proposal_list):
            bag_size, num_gt, _ = bag_bboxes.size()
            roi_index = torch.arange(start_index,
                                     start_index + num_gt,
                                     device=bag_bboxes.device)
            roi_index = roi_index[None, :, None].repeat(bag_size, 1, 1).float()
            bag_bboxes = torch.cat([roi_index, bag_bboxes], dim=-1)
            bag_rois = bag_bboxes.view(-1, 5)
            rois_list.append(bag_rois)
            start_index += num_gt

        rois = torch.cat(rois_list, 0).contiguous()
        self.roi_index = rois[:, :1]
        # keep same during three stages
        self.cood_roi_extractor = copy.deepcopy(self.bbox_roi_extractor[0])
        self.cood_roi_extractor.out_channels = 2
        self.coord_feats = coord_feats[:self.cood_roi_extractor.num_inputs]

        coord_feats = self.cood_roi_extractor(
            coord_feats[:self.cood_roi_extractor.num_inputs], rois)

        return coord_feats

    def _not_first_coord_pooling(self, rois):
        """Pooling relative coordinates for second and third stage."""
        rois = torch.cat([self.roi_index, rois], dim=-1)
        coord_feats = self.cood_roi_extractor(
            self.coord_feats[:self.cood_roi_extractor.num_inputs], rois)
        return coord_feats

    def instance_assign(self, stage, anchors, gt_bboxes, gt_labels,
                        group_size):

        repeat_gts = []
        repeat_labels = []
        num_gts = 0
        group_size_each_gt = []
        for gt, label, bag_size in zip(gt_bboxes, gt_labels, group_size):
            num_gts += len(gt)
            gt = gt[None, :, :].repeat(bag_size, 1, 1)
            label = label[None, :].repeat(bag_size, 1)
            repeat_gts.append(gt.view(-1, 4))
            repeat_labels.append(label.view(-1))
            group_size_each_gt.extend(gt.size(1) * [bag_size])

        repeat_gts = torch.cat(repeat_gts, dim=0)
        repeat_labels = torch.cat(repeat_labels, dim=0)

        self.repeat_labels = repeat_labels

        match_quality_matrix = bbox_overlaps(anchors,
                                             repeat_gts,
                                             is_aligned=True)

        pos_mask = match_quality_matrix > self.pos_iou_thrs[stage]
        targets_weight = match_quality_matrix.new_ones(len(pos_mask))

        bbox_targets = self.bbox_head[stage].bbox_coder.encode(
            anchors, repeat_gts)
        all_labels = torch.ones_like(
            repeat_labels) * self.bbox_head[0].num_classes
        all_labels[pos_mask] = repeat_labels[pos_mask]

        pos_bbox_targets = bbox_targets[pos_mask]

        return pos_mask, pos_bbox_targets, all_labels, targets_weight

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             stage,
             cls_score,
             bbox_pred,
             rois,
             bbox_targets,
             labels,
             pos_mask,
             reduction_override=None):

        label_weights = torch.ones_like(labels)
        bbox_weights = torch.ones_like(bbox_targets)
        losses = dict()
        avg_factor = max(pos_mask.sum(), pos_mask.new_ones(1).sum())

        loss_cls_ = self.bbox_head[stage].loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=avg_factor,
            reduction_override=reduction_override)

        losses['loss_cls'] = loss_cls_
        losses['acc'] = accuracy(cls_score, labels)
        # will be divided by num_gts of single batch outside
        losses['avg_pos'] = pos_mask.sum()
        pos_inds = pos_mask
        # do not perform bounding box regression for BG anymore.
        if pos_inds.any():
            if self.bbox_head[stage].reg_decoded_bbox:
                # When the regression loss (e.g. `IouLoss`,
                # `GIouLoss`, `DIouLoss`) is applied directly on
                # the decoded bounding boxes, it decodes the
                # already encoded coordinates to absolute format.
                bbox_pred = self.bbox_head[stage].bbox_coder.decode(
                    rois[:, 1:], bbox_pred)
            if self.bbox_head[stage].reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0),
                                               4)[pos_inds.type(torch.bool)]
            else:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    4)[pos_inds.type(torch.bool),
                       labels[pos_inds.type(torch.bool)]]
            losses['loss_bbox'] = self.bbox_head[stage].loss_bbox(
                pos_bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
        else:
            losses['loss_bbox'] = bbox_pred.sum() * 0

        return losses

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      rela_coods_list=None,
                      gt_points=None,
                      **kwargs):
        """Get loss of single iter.

        Args:
            x (list(Tensor)): FPN Feature maps
            img_metas(list[dict]): Meta information for multiple images.
            proposal_list (list[Tensor]): Proposals of each instance
                group in multiple images. Each has shape
                (rpn_nms_topk, num_gts, 4).
            gt_bboxes (list[Tensor]): Gt bboxes for multiple images.
                Each has shape (num_gts, 4).
            gt_labels (list[Tensor]): Gt labels for multiple images.
                Each has shape (num_gts,).
            rela_coods_list (list[list[tensor]]): Relative coordinates for
                FPN in multiple images. Each tensor has shape
                (num_instances, h*w, 2).
            gt_points (list[Tensor]): Gt points for multiple images.
                Each has shape (num_gts, 2).

        Returns:
            dict: losses of RPN and RoIHead.
        """
        # when nms_topk may has different bag size
        group_size = [item.size(0) for item in proposal_list]
        wh_each_level = [item.shape[-2:] for item in x]
        num_img = len(rela_coods_list)
        num_level = len(rela_coods_list[0])
        all_num_gts = 0
        format_rela_coods_list = []
        for img_id in range(num_img):
            real_coods = rela_coods_list[img_id]
            mlvl_coord_list = []
            for level in range(num_level):
                format_coords = real_coods[level]
                num_gt = format_coords.size(0)
                all_num_gts += num_gt
                format_coords = format_coords.view(num_gt,
                                                   *wh_each_level[level],
                                                   2).permute(0, 3, 1, 2)
                mlvl_coord_list.append(format_coords)
            format_rela_coods_list.append(mlvl_coord_list)

        mlvl_concate_coods = []
        for level in range(num_level):
            mlti_img_cood = [
                format_rela_coods_list[img_id][level]
                for img_id in range(num_img)
            ]
            concat_coods = torch.cat(mlti_img_cood, dim=0).contiguous()
            mlvl_concate_coods.append(concat_coods)

        losses = dict()

        rois_per_image = [
            item.size(0) * item.size(1) for item in proposal_list
        ]

        rois = None

        for stage in range(self.num_stages):
            if stage == 0:
                coord_feats = self._first_coord_pooling(
                    mlvl_concate_coods, proposal_list)
                feat_rois = bbox2roi(
                    [item.view(-1, 4).detach() for item in proposal_list])
            else:
                coord_feats = self._not_first_coord_pooling(rois)
                feat_rois = rois.split(rois_per_image, dim=0)
                feat_rois = bbox2roi(
                    [item.view(-1, 4).detach() for item in feat_rois])

            pos_mask, pos_bbox_targets, pos_labels, \
                targets_reweight = self.instance_assign(
                     stage, feat_rois[:, 1:], gt_bboxes, gt_labels, group_size)

            bbox_results = self._bbox_forward(
                stage,
                x,
                feat_rois,
                coord_feats,
                group_size,
                rois_per_image,
                gt_labels=gt_labels,
                gt_points=gt_points,
                img_metas=img_metas,
            )

            single_stage_loss = self.loss(stage, bbox_results['cls_score'],
                                          bbox_results['bbox_pred'], feat_rois,
                                          pos_bbox_targets, pos_labels,
                                          pos_mask)
            single_stage_loss['avg_pos'] = single_stage_loss[
                'avg_pos'] / float(all_num_gts) * 5

            for name, value in single_stage_loss.items():
                losses[f's{stage}.{name}'] = (value *
                                              self.stage_loss_weights[stage]
                                              if 'loss' in name else value)

            # refine bboxes
            if stage < self.num_stages - 1:
                with torch.no_grad():
                    rois = self.bbox_head[stage].bbox_coder.decode(
                        feat_rois[:, 1:],
                        bbox_results['bbox_pred'],
                    )

        return losses

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    rela_coods_list=None,
                    labels=None,
                    gt_points=None,
                    **kwargs):
        """Get predictions of single iter.

        Args:
            x (list(Tensor)): FPN Feature maps
            proposal_list (list[Tensor]): Proposals of each instance
                group in multiple images. Each has shape
                (rpn_nms_topk, num_gts, 4).
            img_metas(list[dict]): Meta information for multiple images.
            rela_coods_list (list[list[tensor]]): Relative coordinates for
                FPN in multiple images. Each tensor has shape
                (num_instances, h*w, 2).
            labels (list[Tensor]): Gt labels for multiple images.
                Each has shape (num_gts,).
            gt_points (list[Tensor]): Gt points for multiple images.
                Each has shape (num_gts, 2).

        Returns:
            Tuple:

                - pred_bboxes (list[tensor]): Bbox prediction
                  of multiple images. Each has shape (num_instances, 4).
                - pred_scores (list[tensor]): Score of each bbox in
                  multiple images. Each has shape (num_instances,).
                - pred_labels (list[tensor]): Label of each bbox in
                  multiple images. Each has shape (num_instances,).
        """
        num_images = len(proposal_list)
        group_size = [item.size(0) for item in proposal_list]
        ms_scores = []
        wh_each_level = [item.shape[-2:] for item in x]
        num_img = len(rela_coods_list)
        num_level = len(rela_coods_list[0])
        format_rela_coods_list = []

        repeat_labels = []
        for label, bag_size in zip(labels, group_size):
            label = label[None, :].repeat(bag_size, 1)
            repeat_labels.append(label.view(-1))
        # used in post process in Group R-CNN
        self.repeat_labels = torch.cat(repeat_labels, dim=0)

        for img_id in range(num_img):
            real_coods = rela_coods_list[img_id]
            mlvl_coord_list = []
            for level in range(num_level):
                format_coords = real_coods[level]
                num_gt = format_coords.size(0)
                format_coords = format_coords.view(num_gt,
                                                   *wh_each_level[level],
                                                   2).permute(0, 3, 1, 2)
                mlvl_coord_list.append(format_coords)
            format_rela_coods_list.append(mlvl_coord_list)
        mlvl_concate_coods = []

        for level in range(num_level):
            mlti_img_cood = [
                format_rela_coods_list[img_id][level]
                for img_id in range(num_img)
            ]
            concat_coods = torch.cat(mlti_img_cood, dim=0).contiguous()
            mlvl_concate_coods.append(concat_coods)

        rois_per_image = [
            item.size(0) * item.size(1) for item in proposal_list
        ]
        for stage in range(self.num_stages):
            self.current_stage = stage
            if stage == 0:
                coord_feats = self._first_coord_pooling(
                    mlvl_concate_coods, proposal_list)
                feat_rois = bbox2roi(
                    [item.view(-1, 4).detach() for item in proposal_list])
            else:
                coord_feats = self._not_first_coord_pooling(
                    torch.cat(proposal_list, dim=0))
                feat_rois = proposal_list
                feat_rois = bbox2roi(
                    [item.view(-1, 4).detach() for item in feat_rois])

            bbox_results = self._bbox_forward(
                stage,
                x,
                feat_rois,
                coord_feats,
                rois_per_image=rois_per_image,
                group_size=group_size,
                gt_labels=labels,
                gt_points=gt_points,
                img_metas=img_metas,
            )

            bbox_preds = bbox_results['bbox_pred']

            if self.bbox_head[-1].loss_cls.use_sigmoid:
                cls_score = bbox_results['cls_score'].sigmoid()
                num_classes = cls_score.size(-1)
            else:
                cls_score = bbox_results['cls_score'].softmax(-1)
                num_classes = cls_score.size(-1) - 1
            cls_score = cls_score[:, :num_classes]

            decode_bboxes = []
            all_scores = []

            for img_id in range(num_images):
                img_shape = img_metas[img_id]['img_shape']
                img_mask = feat_rois[:, 0] == img_id
                temp_rois = feat_rois[img_mask]
                temp_bbox_pred = bbox_preds[img_mask]
                bboxes = self.bbox_head[stage].bbox_coder.decode(
                    temp_rois[..., 1:], temp_bbox_pred, max_shape=img_shape)
                temp_scores = cls_score[img_mask]
                bboxes = bboxes.view(group_size[img_id], -1, 4)
                temp_scores = temp_scores.view(group_size[img_id], -1,
                                               num_classes)
                decode_bboxes.append(bboxes)
                all_scores.append(temp_scores)

            ms_scores.append(all_scores)
            proposal_list = [item.view(-1, 4) for item in decode_bboxes]

        ms_scores = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_images)
        ]

        pred_bboxes = []
        pred_scores = []
        pred_labels = []
        # select bbox from each group
        for img_id in range(num_images):
            all_class_scores = ms_scores[img_id]
            if all_class_scores.numel():
                repeat_label = labels[img_id][None].repeat(
                    group_size[img_id], 1)
                scores = torch.gather(all_class_scores, 2,
                                      repeat_label[..., None]).squeeze(-1)

                num_gt = decode_bboxes[img_id].shape[1]
                dets, keep = batched_nms(
                    decode_bboxes[img_id].view(-1, 4), scores.view(-1),
                    repeat_label.view(-1),
                    dict(max_num=1000, iou_threshold=self.iou_threshold))
                num_pred = len(keep)
                gt_index = keep % num_gt
                arrange_gt_index = torch.arange(num_gt,
                                                device=keep.device)[:, None]
                # num_gt x num_pred
                keep_matrix = gt_index == arrange_gt_index
                temp_index = torch.arange(-num_pred,
                                          end=0,
                                          step=1,
                                          device=keep.device)
                keep_matrix = keep_matrix * temp_index

                value_, index = keep_matrix.min(dim=-1)
                dets = dets[index]
                pred_bboxes.append(dets[:, :4])
                pred_scores.append(dets[:, -1])
                pred_labels.append(labels[img_id])

            else:
                pred_bboxes.append(all_class_scores.new_zeros(0, 4))
                pred_scores.append(all_class_scores.new_zeros(0))
                pred_labels.append(all_class_scores.new_zeros(0))

        return pred_bboxes, pred_scores, pred_labels
