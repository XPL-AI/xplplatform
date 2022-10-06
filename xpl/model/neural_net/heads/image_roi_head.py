import torch
import torch.nn.functional as F
from torch import nn, Tensor

import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.ops import roi_align
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor

from xpl.model.neural_net.xpl_model import XPLModel
from collections import OrderedDict
from typing import Optional, List, Dict, Tuple


class ImageROIHead(XPLModel):
    def init_neural_net(self,
                        box_roi_pool=None,
                        box_head=None,
                        box_predictor=None,
                        box_score_thresh: float = 0.05,
                        box_nms_thresh: float = 0.5,
                        box_detections_per_img: int = 100,
                        box_fg_iou_thresh: float = 0.5,
                        box_bg_iou_thresh: float = 0.5,
                        box_batch_size_per_image: int = 512,
                        box_positive_fraction: float = 0.25,
                        bbox_reg_weights=None
                        ) -> torch.nn.Module:
        num_classes = self.definition['output_channels']
        out_channels = 1280  # Hard coded for mobilenet_v2 as backbone

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        return {'neural_net': RoIHeads(box_roi_pool, box_head, box_predictor,
                                       box_fg_iou_thresh, box_bg_iou_thresh,
                                       box_batch_size_per_image, box_positive_fraction,
                                       bbox_reg_weights,
                                       box_score_thresh, box_nms_thresh, box_detections_per_img)
                }

    def forward(self,
                batch: dict
                ) -> None:
        # Reformat input to fit torch architecture
        shapes = [(256, 256) for x in enumerate(batch[self.head_names[0]])]  # Hard coded for size of 256 256
        features = OrderedDict([('0', batch[self.head_names[1]])])
        proposals = batch[self.head_names[2]]
        gt_boxes = batch[self.head_names[3]]
        gt_labels = batch[self.head_names[4]]
        indices = batch[self.head_names[5]]
        targets = []
        for i in range(gt_labels.size()[0]):
            targets.append(create_detection_target(gt_boxes[i].to(self.device), gt_labels[i].to(self.device), indices[i].to(self.device)))

        # Pass feature map(s), proposals, image shapes and targets through neural net
        detections, sampled_class_logits, sampled_box_regression, sampled_labels, sampled_regression_targets = self.neural_net(
            features, proposals, shapes, targets)
        # Put detection in batch
        batch[self.tail_names[0]] = detections

        # Put tensors needed for computing loss and error in batch
        batch[self.tail_names[1]] = sampled_class_logits
        batch[self.tail_names[2]] = sampled_box_regression
        batch[self.tail_names[3]] = sampled_labels
        batch[self.tail_names[4]] = sampled_regression_targets
        return None


def create_detection_target(bounding_box: Tensor,
                            label: Tensor,
                            idx: Tensor
                            ) -> dict:
    area = torch.tensor(1).to(bounding_box.device)
    iscrowd = torch.tensor(0).to(bounding_box.device)
    target = {}
    target["boxes"] = bounding_box
    target["labels"] = label.reshape(1)
    target["image_id"] = idx
    target["area"] = area
    target["iscrowd"] = iscrowd
    return target


class RoIHeads(nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool: MultiScaleRoIAlign,
                 box_head: TwoMLPHead,
                 box_predictor: FastRCNNPredictor,
                 # Faster R-CNN training
                 fg_iou_thresh: float, bg_iou_thresh: float,
                 batch_size_per_image: int, positive_fraction: float,
                 bbox_reg_weights: float,
                 # Faster R-CNN inference
                 score_thresh: float,
                 nms_thresh: float,
                 detections_per_img: int,
                 ) -> None:
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Optional[Tensor], Optional[Tensor], Optional[List[Tensor]], Optional[List[Tensor]]]
        if targets is not None:
            sampled_proposals, matched_idxs, sampled_labels, sampled_regression_targets = self.select_training_samples(proposals, targets)
            sampled_box_features = self.box_roi_pool(features, sampled_proposals, image_shapes)
            sampled_box_features = self.box_head(sampled_box_features)
            sampled_class_logits, sampled_box_regression = self.box_predictor(sampled_box_features)
        else:
            sampled_class_logits = None
            sampled_box_regression = None
            sampled_labels = None
            sampled_regression_targets = None

        labels = None
        regression_targets = None
        matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

        return result, sampled_class_logits, sampled_box_regression, sampled_labels, sampled_regression_targets
