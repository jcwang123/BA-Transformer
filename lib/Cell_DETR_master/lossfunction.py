from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import misc
from matcher import HungarianMatcher


class LovaszHingeLoss(nn.Module):
    """
    This class implements the lovasz hinge loss which is the continuous of the IoU for binary segmentation.
    Source: https://github.com/bermanmaxim/LovaszSoftmax
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(LovaszHingeLoss, self).__init__()

    def _calc_grad(self, label_sorted: torch.Tensor) -> torch.Tensor:
        """
        Method computes the gradients of the sorted and flattened label
        :param label_sorted: (torch.Tensor) Sorted and flattened label of shape [n]
        :return: (torch.Tensor) Gradient tensor
        """
        # Calc sum of labels
        label_sum = label_sorted.sum()
        # Calc intersection
        intersection = label_sum - label_sorted.cumsum(dim=0)
        # Calc union
        union = label_sum + (1 - label_sorted).cumsum(dim=0)
        # Calc iou
        iou = 1.0 - (intersection / union)
        # Calc grad
        iou[1:] = iou[1:] - iou[0:-1]
        return iou

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the dice loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Dice loss value
        """
        # Flatten both tensors
        prediction = prediction.flatten(start_dim=0)
        label = label.flatten(start_dim=0)
        # Get signs of the label
        signs = 2.0 * label - 1.0
        # Get error
        error = 1.0 - prediction * signs
        # Sort errors
        errors_sorted, permutation = torch.sort(error, dim=0, descending=True)
        # Apply permutation to label
        label_sorted = label[permutation]
        # Calc grad of permuted label
        grad = self._calc_grad(label_sorted)
        # Calc final loss
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss


class DiceLoss(nn.Module):
    """
    This class implements the dice loss for multiple instances
    """

    def __init__(self, smooth_factor: float = 1.0) -> None:
        # Call super constructor
        super(DiceLoss, self).__init__()
        # Save parameter
        self.smooth_factor = smooth_factor

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, smooth factor={}".format(self.__class__.__name__, self.smooth_factor)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the dice loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Dice loss value
        """
        # Flatten both tensors
        prediction = prediction.flatten(start_dim=0)
        label = label.flatten(start_dim=0)
        # Calc dice loss
        loss = torch.tensor(1.0, dtype=torch.float32, device=prediction.device) \
               - ((2.0 * torch.sum(torch.mul(prediction, label)) + self.smooth_factor)
                  / (torch.sum(prediction) + torch.sum(label) + self.smooth_factor))
        return loss


class FocalLoss(nn.Module):
    """
    This class implements the segmentation focal loss.
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        """
        Constructor method
        :param alpha: (float) Alpha constant
        :param gamma: (float) Gamma constant (see paper)
        """
        # Call super constructor
        super(FocalLoss, self).__init__()
        # Save parameters
        self.alpha = alpha
        self.gamma = gamma

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, alpha={}, gamma={}".format(self.__class__.__name__, self.alpha, self.gamma)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the binary cross entropy loss of segmentation masks
        :param prediction: (torch.Tensor) Prediction probability
        :param label: (torch.Tensor) Label one-hot encoded
        :return: (torch.Tensor) Loss value
        """
        # Calc binary cross entropy loss as normal
        binary_cross_entropy_loss = -(label * torch.log(prediction.clamp(min=1e-12))
                                      + (1.0 - label) * torch.log((1.0 - prediction).clamp(min=1e-12)))
        # Calc focal loss factor based on the label and the prediction
        focal_factor = prediction * label + (1.0 - prediction) * (1.0 - label)
        # Calc final focal loss
        loss = ((1.0 - focal_factor) ** self.gamma * binary_cross_entropy_loss * self.alpha).sum(dim=1).mean()
        return loss


class LovaszSoftmaxLoss(nn.Module):
    """
    Implementation of the Lovasz-Softmax loss.
    https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(LovaszSoftmaxLoss, self).__init__()

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the dice loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Dice loss value
        """
        # One hot to class num
        _, label = label.max(dim=0)
        # Flatten tensors
        classes, height, width = prediction.size()
        prediction = prediction.permute(1, 2, 0).contiguous().view(-1, classes)
        label = label.view(-1)
        # Allocate tensor for every class loss
        losses = torch.zeros(classes, dtype=torch.float, device=prediction.device)
        # Calc loss for every class
        for c in range(classes):
            # Foreground for c
            fg = (label == c).float()
            # Class prediction
            class_prediction = prediction[:, c]
            # Calc error
            errors = (Variable(fg) - class_prediction).abs()
            # Sort errors
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            # Sort foreground
            perm = perm.data
            fg_sorted = fg[perm]
            # Calc grad
            p = len(fg_sorted)
            gts = fg_sorted.sum()
            intersection = gts - fg_sorted.float().cumsum(0)
            union = gts + (1 - fg_sorted).float().cumsum(0)
            jaccard = 1. - intersection / union
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
            # Calc class loss
            losses[c] = torch.dot(errors_sorted, Variable(jaccard))
        return losses.mean()


class FocalLossMultiClass(nn.Module):
    """
    Implementation of the multi class focal loss.
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        """
        Constructor method
        :param alpha: (float) Alpha constant
        :param gamma: (float) Gamma constant (see paper)
        """
        # Call super constructor
        super(FocalLossMultiClass, self).__init__()
        # Save parameters
        self.alpha = alpha
        self.gamma = gamma

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, alpha={}, gamma={}".format(self.__class__.__name__, self.alpha, self.gamma)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the binary cross entropy loss of segmentation masks
        :param prediction: (torch.Tensor) Prediction probability
        :param label: (torch.Tensor) Label one-hot encoded
        :return: (torch.Tensor) Loss value
        """
        # Calc binary cross entropy loss as normal
        cross_entropy_loss = - (label * torch.log(prediction.clamp(min=1e-12))).sum(dim=0)
        # Calc focal loss factor based on the label and the prediction
        focal_factor = (prediction * label + (1.0 - prediction) * (1.0 - label))
        # Calc final focal loss
        loss = ((1.0 - focal_factor) ** self.gamma * cross_entropy_loss * self.alpha).sum(dim=0).mean()
        return loss


class MultiClassSegmentationLoss(nn.Module):
    """
    Multi class segmentation loss for the case if a softmax is utilized as the final activation.
    """

    def __init__(self, dice_loss: nn.Module = DiceLoss(),
                 focal_loss: nn.Module = FocalLossMultiClass(),
                 lovasz_softmax_loss: nn.Module = LovaszSoftmaxLoss(),
                 w_dice: float = 1.0, w_focal: float = 0.1, w_lovasz_softmax: float = 0.0) -> None:
        # Call super constructor
        super(MultiClassSegmentationLoss, self).__init__()
        # Save parameters
        self.dice_loss = dice_loss
        self.focal_loss = focal_loss
        self.lovasz_softmax_loss = lovasz_softmax_loss
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_lovasz_softmax = w_lovasz_softmax

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, {}, w_focal={}, {}, w_dice={}, " \
               "{}, w_lovasz_softmax={}".format(self.__class__.__name__,
                                                self.dice_loss.__class__.__name__,
                                                self.w_dice,
                                                self.focal_loss.__class__.__name__,
                                                self.w_focal,
                                                self.lovasz_softmax_loss.__class__.__name__,
                                                self.w_lovasz_softmax)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the segmentation loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Loss value
        """
        return self.w_dice * self.dice_loss(prediction, label) \
               + self.w_focal * self.focal_loss(prediction, label) \
               + self.w_lovasz_softmax * self.lovasz_softmax_loss(prediction, label)


class SegmentationLoss(nn.Module):
    """
    This class implement the segmentation loss.
    """

    def __init__(self, dice_loss: nn.Module = DiceLoss(),
                 focal_loss: nn.Module = FocalLoss(),
                 lovasz_hinge_loss: nn.Module = LovaszHingeLoss(),
                 w_dice: float = 1.0, w_focal: float = 0.2, w_lovasz_hinge: float = 0.0) -> None:
        # Call super constructor
        super(SegmentationLoss, self).__init__()
        # Save parameters
        self.dice_loss = dice_loss
        self.focal_loss = focal_loss
        self.lovasz_hinge_loss = lovasz_hinge_loss
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_lovasz_hinge = w_lovasz_hinge

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, {}, w_focal={}, {}, w_dice={}, " \
               "{}, w_lovasz_hinge={}".format(self.__class__.__name__,
                                              self.dice_loss.__class__.__name__,
                                              self.w_dice,
                                              self.focal_loss.__class__.__name__,
                                              self.w_focal,
                                              self.lovasz_hinge_loss.__class__.__name__,
                                              self.w_lovasz_hinge)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the segmentation loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Loss value
        """
        return self.w_dice * self.dice_loss(prediction, label) \
               + self.w_focal * self.focal_loss(prediction, label) \
               + self.w_lovasz_hinge * self.lovasz_hinge_loss(prediction, label)


class BoundingBoxGIoULoss(nn.Module):
    """
    This class implements the generalized bounding box iou proposed in:
    https://giou.stanford.edu/
    This implementation is highly based on the torchvision bb iou implementation and on:
    https://github.com/facebookresearch/detr/blob/be9d447ea3208e91069510643f75dadb7e9d163d/util/box_ops.py
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(BoundingBoxGIoULoss, self).__init__()

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}".format(self.__class__.__name__)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the GIoU
        :param prediction: (torch.Tensor) Bounding box prediction of shape (batch size, instances, 4)
        :param label: (torch.Tensor) Bounding box labels of shape (batch size, instances, 4)
        :return: (torch.Tensor) GIoU loss value
        """
        return 1.0 - misc.giou(bounding_box_1=prediction, bounding_box_2=label).diagonal().mean()


class BoundingBoxLoss(nn.Module):
    """
    This class implements the bounding box loss proposed in:
    https://arxiv.org/abs/2005.12872
    """

    def __init__(self, iou_loss_function: nn.Module = BoundingBoxGIoULoss(),
                 l1_loss_function: nn.Module = nn.L1Loss(reduction="mean"), weight_iou: float = 0.4,
                 weight_l1: float = 0.6) -> None:
        """
        Constructor method
        :param iou_loss_function: (nn.Module) Loss function module of iou loss
        :param l1_loss_function: (nn.Module) Loss function module of l1 loss
        :param weight_iou: (float) Weights factor of the iou loss
        :param weight_l1: (float) Weights factor of the l1 loss
        """
        # Call super constructor
        super(BoundingBoxLoss, self).__init__()
        # Save parameters
        self.iou_loss_function = iou_loss_function
        self.l1_loss_function = l1_loss_function
        self.weight_iou = weight_iou
        self.weight_l1 = weight_l1

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, {}, w_iou={}, {}, w_l1={}".format(self.__class__.__name__,
                                                      self.iou_loss_function.__class__.__name__,
                                                      self.weight_l1, self.l1_loss_function.__class__.__name__,
                                                      self.weight_l1)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the combined loss
        :param prediction: (torch.Tensor) Bounding box prediction of shape (batch size, instances, 4)
        :param label: (torch.Tensor) Bounding box labels of shape (batch size, instances, 4)
        :return: (torch.Tensor) Loss value
        """
        return self.weight_iou * self.iou_loss_function(prediction, label) \
               + self.weight_l1 * self.l1_loss_function(prediction, label)


class ClassificationLoss(nn.Module):
    """
    This class implements a cross entropy classification loss
    """

    def __init__(self, class_weights=torch.tensor([0.5, 0.5, 1.5, 1.5], dtype=torch.float)) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(ClassificationLoss, self).__init__()
        # Save parameter
        self.class_weights = class_weights

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, class weights:{}".format(self.__class__.__name__, self.class_weights)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor, ohem: bool = False) -> torch.Tensor:
        """
        Forward pass computes loss value
        :param prediction: (torch.Tensor) Prediction one hot encoded with shape (batch size, instances, classes + 1)
        :param label: (torch.Tensor) Label one hot encoded with shape (batch size, instances, classes + 1)
        :param ohem: (bool) If true batch size is not reduced for online hard example mining
        :return: (torch.Tensor) Loss value
        """
        # Compute loss
        if ohem:
            return (- label * torch.log(prediction.clamp(min=1e-12))
                    * self.class_weights.to(label.device)).sum(dim=-1).mean(dim=-1)
        return (- label * torch.log(prediction.clamp(min=1e-12))
                * self.class_weights.to(label.device)[:prediction.shape[-1]]).sum(dim=-1).mean()


class InstanceSegmentationLoss(nn.Module):
    """
    This class combines all losses for instance segmentation
    """

    def __init__(self, classification_loss: nn.Module = ClassificationLoss(),
                 bounding_box_loss: nn.Module = BoundingBoxLoss(),
                 segmentation_loss: nn.Module = SegmentationLoss(),
                 matcher: nn.Module = HungarianMatcher(),
                 w_classification: float = 1.0, w_bounding_box: float = 1.0, w_segmentation: float = 1.0,
                 ohem: bool = False, ohem_faction: float = 0.75) -> None:
        """
        Constructor method
        :param classification_loss: (nn.Module) Classification loss function
        :param bounding_box_loss: (nn.Module) Bounding box loss function
        :param segmentation_loss: (nn.Module) Segmentation loss function
        :param matcher: (nn.Module) Matcher module to estimate the best permutation of label prediction
        :param w_classification: (float) Weights factor of the classification loss
        :param w_bounding_box: (float) Weights factor of the bounding box loss
        :param w_segmentation: (float) Weights factor of the segmentation loss
        :param ohem: (bool) True if hard example mining should be utilized
        :param ohem_faction: (float) Fraction of the whole batch size which is returned after ohm
        """
        # Call super constructor
        super(InstanceSegmentationLoss, self).__init__()
        # Save parameters
        self.classification_loss = classification_loss
        self.bounding_box_loss = bounding_box_loss
        self.segmentation_loss = segmentation_loss
        self.matcher = matcher
        self.w_classification = w_classification
        self.w_bounding_box = w_bounding_box
        self.w_segmentation = w_segmentation
        self.ohem = ohem
        self.ohem_faction = ohem_faction

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, Classification Loss:{}, w_classification={}, Bounding Box Loss:{}, w_classification={}, " \
               "Segmentation Loss:{}, w_classification={}, Matcher:{}" \
            .format(self.__class__.__name__,
                    self.classification_loss, self.w_classification,
                    self.bounding_box_loss, self.w_bounding_box,
                    self.segmentation_loss, self.w_segmentation,
                    self.matcher)

    def _construct_full_classification_label(self, label: List[torch.Tensor],
                                             number_of_predictions: int) -> torch.Tensor:
        """
        Method fills a given label with one hot encoded no-object labels
        :param label: (Tuple[torch.Tensor]) Tuple of each batch instance with variable number of instances
        :param number_of_predictions: (int) Number of predictions from the network
        :return: (torch.Tensor) Filled tensor with no-object classes [batch size, # predictions, classes + 1]
        """
        # Init new label
        new_label = torch.zeros(len(label), number_of_predictions, label[0].shape[-1])
        # Set no-object class in new label
        no_object_vector = torch.zeros(number_of_predictions, label[0].shape[-1])
        no_object_vector[:, 0] = 1.0
        new_label[:, :] = no_object_vector
        # New label to device
        new_label = new_label.to(label[0].device)
        # Iterate over all batch instances
        for index, batch_instance in enumerate(label):
            # Add existing label to new label
            new_label[index, :batch_instance.shape[0]] = batch_instance
        return new_label

    def apply_permutation(self, prediction: torch.Tensor, label: List[torch.Tensor],
                          indexes: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Method applies a given permutation to the prediction and the label
        :param prediction: (torch.Tensor) Prediction tensor of shape [batch size, # predictions, ...]
        :param label: (Tuple[torch.Tensor]) Label of shape len([[instances, ...]])= batch size
        :param indexes: (List[Tuple[torch.Tensor, torch.Tensor]])) Permutation indexes for each instance
        :return: (Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]) Prediction and label with permutation
        """
        # Iterate over batch size
        for batch_index in range(len(label)):
            # Apply permutation to label
            label[batch_index] = label[batch_index][indexes[batch_index][1].long()]
            # Apply permutation to prediction
            prediction[batch_index, :] = prediction[batch_index, torch.unique(
                torch.cat([indexes[batch_index][0].long(), torch.arange(0, prediction[batch_index].shape[0]).long()],
                          dim=0), sorted=False).long().flip(dims=(0,))]
        return prediction, label

    def forward(self, prediction_classification: torch.Tensor,
                prediction_bounding_box: torch.Tensor,
                prediction_segmentation: torch.Tensor,
                label_classification: List[torch.Tensor],
                label_bounding_box: List[torch.Tensor],
                label_segmentation: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computes combined loss
        :param prediction_classification: (torch.Tensor) Classification prediction
        :param prediction_bounding_box: (torch.Tensor) Bounding box prediction
        :param prediction_segmentation: (torch.Tensor) Segmentation prediction
        :param label_classification: (List[torch.Tensor]) Classification label
        :param label_bounding_box: (List[torch.Tensor]) Bounding box label
        :param label_segmentation: (List[torch.Tensor]) Segmentation label
        :return: (torch.Tensor) Loss value
        """
        # Get matching indexes
        matching_indexes = self.matcher(prediction_classification, prediction_bounding_box, label_classification,
                                        label_bounding_box)
        # Apply permutation to labels and predictions
        prediction_classification, label_classification = self.apply_permutation(prediction=prediction_classification,
                                                                                 label=label_classification,
                                                                                 indexes=matching_indexes)
        prediction_bounding_box, label_bounding_box = self.apply_permutation(prediction=prediction_bounding_box,
                                                                             label=label_bounding_box,
                                                                             indexes=matching_indexes)
        prediction_segmentation, label_segmentation = self.apply_permutation(prediction=prediction_segmentation,
                                                                             label=label_segmentation,
                                                                             indexes=matching_indexes)
        # Construct full classification label
        label_classification = self._construct_full_classification_label(label=label_classification,
                                                                         number_of_predictions=
                                                                         prediction_classification.shape[1])
        # Calc classification loss
        loss_classification = self.classification_loss(prediction_classification, label_classification, self.ohem)
        # Calc bounding box loss
        loss_bounding_box = torch.zeros(len(label_bounding_box), dtype=torch.float,
                                        device=prediction_segmentation.device)
        for batch_index in range(len(label_bounding_box)):
            # Calc loss for each batch instance
            loss_bounding_box[batch_index] = self.bounding_box_loss(
                misc.bounding_box_xcycwh_to_x0y0x1y1(
                    prediction_bounding_box[batch_index, :label_bounding_box[batch_index].shape[0]]),
                misc.bounding_box_xcycwh_to_x0y0x1y1(label_bounding_box[batch_index]))
        # Calc segmentation loss
        loss_segmentation = torch.zeros(len(label_bounding_box), dtype=torch.float,
                                        device=prediction_segmentation.device)
        for batch_index in range(len(label_segmentation)):
            # Calc loss for each batch instance
            loss_segmentation[batch_index] = self.segmentation_loss(
                prediction_segmentation[batch_index, :label_segmentation[batch_index].shape[0]],
                label_segmentation[batch_index])
        # Perform online hard example mining if utilized
        if self.ohem:
            # Calc full loss for each batch instance
            loss = self.w_classification * loss_classification + self.w_bounding_box * loss_bounding_box \
                   + self.w_segmentation * loss_segmentation
            # Perform arg sort to get highest losses
            sorted_indexes = torch.argsort(loss, descending=True)
            # Get indexes with the highest loss and apply ohem fraction
            sorted_indexes = sorted_indexes[:min(int(self.ohem_faction * len(label_segmentation)), 1)]
            # Get corresponding losses and perform mean reduction
            return self.w_classification * loss_classification[sorted_indexes].mean(), \
                   self.w_bounding_box * loss_bounding_box[sorted_indexes].mean(), \
                   self.w_segmentation * loss_segmentation[sorted_indexes].mean()
        return self.w_classification * loss_classification, self.w_bounding_box * loss_bounding_box.mean(), \
               self.w_segmentation * loss_segmentation.mean()
