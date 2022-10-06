from logging import error
import math
#from xpl.sandbox.Flow import losses
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from xpl.measurer.xpl_measurer import XPLMeasurer
class RpnMeasurer(XPLMeasurer):
    def init_measurer(self,
                      definition: dict
                      ) -> None:
        assert 'outputs' in definition, f'{definition=} must contain "outputs"'
        self.__output_name = definition['outputs']

        assert 'output_is_probability' in definition, f'{definition=} must contain "output_is_probability"'
        self.__output_is_probability = definition['output_is_probability']
        assert isinstance(self.__output_is_probability, bool), f'{self.__output_is_probability=} must be a boolean'

        assert 'targets' in definition, f'{definition=} must contain "targets"'
        self.__target_name = definition['targets']

        assert 'num_classes' in definition, f'{definition=} must contain "targets"'
        self.__num_classes = definition['num_classes']
        assert isinstance(self.__num_classes, int), f'{self.__num_classes=} must be a int'

        self.__class_counter = torch.ones(self.__num_classes)

        self.loss_function = rpn_loss
        self.error_function = box_error
        
    def __call__(self,
                 batch: dict,
                 is_train: bool,
                 ) -> torch.Tensor: 
        #Get ground truth from batch
        gt_boxes = batch[self.__target_name[1]]

        #Get proposals and values needed for computing loss from batch
        proposals = batch[self.__output_name[0]]
        objectness = batch[self.__output_name[1]]
        pred_bbox_deltas = batch[self.__output_name[2]]
        labels = batch[self.__output_name[3]]
        regression_targets = batch[self.__output_name[4]]
        sampled_inds = batch[self.__output_name[5]]
        
        #Compute loss and error
        loss_value = sum(self.loss_function(objectness, pred_bbox_deltas, labels, regression_targets, sampled_inds))
        error_value = self.error_function(proposals, gt_boxes)
        return {
            'loss': loss_value,
            'error': error_value,
            'entropy': torch.tensor(0.0) #not implemented
        }

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def box_error(proposals, gt_boxes):
    average_intersection_over_union = 0     
    for i in range(len(proposals)):
        detected_boxes = proposals[i]
        if detected_boxes.nelement() == 0: #If model has not trained enough the output will be a blank box
            best_detected_box = [0,0,1,1]
        else:
            best_detected_box = detected_boxes[0].tolist()            
        ground_truth_box = gt_boxes[i].tolist()[0]
        intersection_over_union = bb_intersection_over_union(best_detected_box, ground_truth_box)
        average_intersection_over_union += intersection_over_union
    average_intersection_over_union /= len(proposals)
    return torch.tensor(1-average_intersection_over_union)

def rpn_loss(objectness, pred_bbox_deltas, labels, regression_targets, sampled_inds):
    objectness = objectness.flatten()

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    box_loss = F.smooth_l1_loss(
        pred_bbox_deltas[sampled_inds],
        regression_targets[sampled_inds],
        beta=1 / 9,
        reduction='sum',
    ) / (sampled_inds.numel())

    objectness_loss = F.binary_cross_entropy_with_logits(
        objectness[sampled_inds], labels[sampled_inds]
    )

    return objectness_loss, box_loss


