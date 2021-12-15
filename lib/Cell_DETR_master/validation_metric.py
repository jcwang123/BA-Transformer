import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score

import misc


class IoU(nn.Module):
    """
    This class implements the IoU for validation. Not gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(IoU, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the IoU score
        :param prediction: (torch.Tensor) Prediction of all shapes
        :param label: (torch.Tensor) Label of all shapes
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) IoU score
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Compute intersection
        intersection = ((prediction + label) == 2.0).sum()
        # Compute union
        union = ((prediction + label) >= 1.0).sum()
        # Compute iou
        return intersection / (union + 1e-10)


class CellIoU(nn.Module):
    """
    This class implements the IoU metric for cell instances.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(CellIoU, self).__init__()
        # Save parameter
        self.threshold = threshold

    def forward(self, prediction: torch.Tensor, label: torch.Tensor, class_label: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Forward pass
        :param prediction: (torch.Tensor) Instance segmentation prediction
        :param label: (torch.Tensor) Instance segmentation label
        :param class_label: (torch.Tensor) Class label of each instance segmentation map
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Mean cell IoU metric
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Get segmentation maps belonging to the cell class
        indexes = np.argwhere(class_label.cpu().numpy() >= 2)[:, 0]
        # Case if no cells are present
        if indexes.shape == (0,):
            return torch.tensor(np.nan)
        prediction = prediction[indexes].sum(dim=0)
        label = label[indexes].sum(dim=0)
        # Compute intersection
        intersection = ((prediction + label) == 2.0).sum(dim=(-2, -1))
        # Compute union
        union = ((prediction + label) >= 1.0).sum(dim=(-2, -1))
        # Compute iou
        return intersection / (union + 1e-10)


class MIoU(nn.Module):
    """
    This class implements the mean IoU for validation. Not gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(MIoU, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the IoU score
        :param prediction: (torch.Tensor) Prediction of shape [..., height, width]
        :param label: (torch.Tensor) Label of shape [..., height, width]
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) IoU score
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Compute intersection
        intersection = ((prediction + label) == 2.0).sum(dim=(-2, -1))
        # Compute union
        union = ((prediction + label) >= 1.0).sum(dim=(-2, -1))
        # Compute iou
        return (intersection / (union + 1e-10)).mean()


class Dice(nn.Module):
    """
    This class implements the dice score for validation. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(Dice, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the dice coefficient
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Dice coefficient
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Compute intersection
        intersection = ((prediction + label) == 2.0).sum()
        # Compute dice score
        return (2 * intersection) / (prediction.sum() + label.sum() + 1e-10)


class ClassificationAccuracy(nn.Module):
    """
    This class implements the classification accuracy computation. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(ClassificationAccuracy, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the accuracy score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Accuracy
        """
        # Calc correct classified elements
        correct_classified_elements = (prediction == label).float().sum()
        # Calc accuracy
        return correct_classified_elements / prediction.numel()


class InstancesAccuracy(nn.Module):
    """
    This class implements the accuracy computation. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(InstancesAccuracy, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the accuracy score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Accuracy
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Calc correct classified elements
        correct_classified_elements = (prediction == label).float().sum()
        # Calc accuracy
        return correct_classified_elements / prediction.numel()


class Accuracy(nn.Module):
    """
    This class implements the accuracy computation. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(Accuracy, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the accuracy score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Accuracy
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Get instance map
        prediction = (prediction
                      * torch.arange(1, prediction.shape[0] + 1, device=prediction.device).view(-1, 1, 1)).sum(dim=0)
        label = (label * torch.arange(1, label.shape[0] + 1, device=label.device).view(-1, 1, 1)).sum(dim=0)
        # Calc correct classified elements
        correct_classified_elements = (prediction == label).float().sum()
        # Calc accuracy
        return correct_classified_elements / prediction.numel()


class Recall(nn.Module):
    """
    This class implements the recall score. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(Recall, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the recall score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Recall score
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Calc true positive elements
        true_positive_elements = (((prediction == 1.0).float() + (label == 1.0)) == 2.0).float()
        # Calc false negative elements
        false_negative_elements = (((prediction == 0.0).float() + (label == 1.0)) == 2.0).float()
        # Calc recall scale
        return true_positive_elements.sum() / ((true_positive_elements + false_negative_elements).sum() + 1e-10)


class Precision(nn.Module):
    """
    This class implements the precision score. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(Precision, self).__init__()
        # Save parameter
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the precision score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Precision score
        """
        # Apply threshold to prediction
        prediction = (prediction > self.threshold).float()
        # Calc true positive elements
        true_positive_elements = (((prediction == 1.0).float() + (label == 1.0)) == 2.0).float()
        # Calc false positive elements
        false_positive_elements = (((prediction == 1.0).float() + (label == 0.0)) == 2.0).float()
        # Calc precision
        return true_positive_elements.sum() / ((true_positive_elements + false_positive_elements).sum() + 1e-10)


class F1(nn.Module):
    """
    This class implements the F1 score. No gradients supported.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        # Call super constructor
        super(F1, self).__init__()
        # Init recall and precision module
        self.recall = Recall(threshold=threshold)
        self.precision = Precision(threshold=threshold)

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the F1 score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) F1 score
        """
        # Calc recall
        recall = self.recall(prediction, label)
        # Calc precision
        precision = self.precision(prediction, label)
        # Calc F1 score
        return (2.0 * recall * precision) / (recall + precision + 1e-10)


class BoundingBoxIoU(nn.Module):
    """
    This class implements the bounding box IoU.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(BoundingBoxIoU, self).__init__()

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the bounding box iou score
        :param prediction: (torch.Tensor) Bounding box predictions [batch size, instances, 4 (x0, y0, x1, y1)]
        :param label: (torch.Tensor) Bounding box labels in the format [batch size, instances, 4 (x0, y0, x1, y1)]
        :return: (torch.Tensor) Bounding box iou
        """
        return misc.giou(bounding_box_1=prediction, bounding_box_2=label, return_iou=True)[1].diagonal().mean()


class BoundingBoxGIoU(nn.Module):
    """
    This class implements the bounding box IoU.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(BoundingBoxGIoU, self).__init__()

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the bounding box iou score
        :param prediction: (torch.Tensor) Bounding box predictions [batch size, instances, 4 (x0, y0, x1, y1)]
        :param label: (torch.Tensor) Bounding box labels in the format [batch size, instances, 4 (x0, y0, x1, y1)]
        :return: (torch.Tensor) Bounding box iou
        """
        return misc.giou(bounding_box_1=prediction, bounding_box_2=label).diagonal().mean()


class MeanAveragePrecision(nn.Module):
    """
    This class implements the mean average precision metric for instance segmentation.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(MeanAveragePrecision, self).__init__()

    @torch.no_grad()
    def forward(self, prediction: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass computes the accuracy score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Accuracy
        """
        # Flatten tensors and convert to numpy
        prediction_flatten = prediction.detach().cpu().view(-1).numpy()
        label_flatten = label.detach().cpu().view(-1).numpy()
        # Calc accuracy
        return torch.tensor(average_precision_score(label_flatten, prediction_flatten, average="macro"),
                            dtype=torch.float, device=label.device)
