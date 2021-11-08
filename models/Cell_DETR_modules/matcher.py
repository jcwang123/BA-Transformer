from typing import Tuple, List

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

import misc


class HungarianMatcher(nn.Module):
    """
    This class implements a hungarian algorithm based matcher for DETR.
    """

    def __init__(self, weight_classification: float = 1.0,
                 weight_bb_l1: float = 1.0,
                 weight_bb_giou: float = 1.0) -> None:
        # Call super constructor
        super(HungarianMatcher, self).__init__()
        # Save parameters
        self.weight_classification = weight_classification
        self.weight_bb_l1 = weight_bb_l1
        self.weight_bb_giou = weight_bb_giou

    def __repr__(self):
        """
        Get representation of the matcher module
        :return: (str) String including information
        """
        return "{}, W classification:{}, W BB L1:{}, W BB gIoU".format(self.__class__.__name__,
                                                                       self.weight_classification, self.weight_bb_l1,
                                                                       self.weight_bb_giou)

    @torch.no_grad()
    def forward(self, prediction_classification: torch.Tensor,
                prediction_bounding_box: torch.Tensor,
                label_classification: Tuple[torch.Tensor],
                label_bounding_box: Tuple[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass computes the permutation produced by the hungarian algorithm.
        :param prediction_classification: (torch.Tensor) Classification prediction (batch size, # queries, classes + 1)
        :param prediction_bounding_box: (torch.Tensor) BB predictions (batch size, # queries, 4)
        :param label_classification: (Tuple[torch.Tensor]) Classification label batched [(instances, classes + 1)]
        :param label_bounding_box: (Tuple[torch.Tensor]) BB label batched [(instances, 4)]
        :return: (torch.Tensor) Permutation of shape (batch size, instances)
        """
        # Save shapes
        batch_size, number_of_queries = prediction_classification.shape[:2]
        # Get number of instances in each training sample
        number_of_instances = [label_bounding_box_instance.shape[0] for label_bounding_box_instance in
                               label_bounding_box]
        # Flatten  to shape [batch size * # queries, classes + 1]
        prediction_classification = prediction_classification.flatten(start_dim=0, end_dim=1)
        # Flatten  to shape [batch size * # queries, 4]
        prediction_bounding_box = prediction_bounding_box.flatten(start_dim=0, end_dim=1)
        # Class label to index
        # Concat labels
        label_classification = torch.cat([instance.argmax(dim=-1) for instance in label_classification], dim=0)
        label_bounding_box = torch.cat([instance for instance in label_bounding_box], dim=0)
        # Compute classification cost
        cost_classification = -prediction_classification[:, label_classification.long()]
        # Compute the L1 cost of bounding boxes
        cost_bounding_boxes_l1 = torch.cdist(prediction_bounding_box, label_bounding_box, p=1)
        # Compute gIoU cost of bounding boxes
        cost_bounding_boxes_giou = -misc.giou_for_matching(
            misc.bounding_box_xcycwh_to_x0y0x1y1(prediction_bounding_box),
            misc.bounding_box_xcycwh_to_x0y0x1y1(label_bounding_box))
        # Construct cost matrix
        cost_matrix = self.weight_classification * cost_classification \
                      + self.weight_bb_l1 * cost_bounding_boxes_l1 \
                      + self.weight_bb_giou * cost_bounding_boxes_giou
        cost_matrix = cost_matrix.view(batch_size, number_of_queries, -1).cpu().clamp(min=-1e20, max=1e20)
        # Get optimal indexes
        indexes = [linear_sum_assignment(cost_vector[index]) for index, cost_vector in
                   enumerate(cost_matrix.split(number_of_instances, dim=-1))]
        # Convert indexes to list of prediction index and label index
        return [(torch.as_tensor(index_prediction, dtype=torch.int), torch.as_tensor(index_label, dtype=torch.int)) for
                index_prediction, index_label in indexes]
