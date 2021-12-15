from typing import Callable, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np

import misc
import augmentation


class CellInstanceSegmentation(Dataset):
    """
    This dataset implements the cell instance segmentation dataset for the DETR model.
    Dataset source: https://github.com/ChristophReich1996/BCS_Data/tree/master/Cell_Instance_Segmentation_Regular_Traps
    """

    def __init__(self, path: str = "../../BCS_Data/Cell_Instance_Segmentation_Regular_Traps/train",
                 normalize: bool = True,
                 normalization_function: Callable[[torch.Tensor], torch.Tensor] = misc.normalize,
                 augmentation: Tuple[augmentation.Augmentation, ...] = (
                         augmentation.VerticalFlip(), augmentation.NoiseInjection(), augmentation.ElasticDeformation()),
                 augmentation_p: float = 0.5, return_absolute_bounding_box: bool = False,
                 downscale: bool = True, downscale_shape: Tuple[int, int] = (128, 128),
                 two_classes: bool = True) -> None:
        """
        Constructor method
        :param path: (str) Path to dataset
        :param normalize: (bool) If true normalization_function is applied
        :param normalization_function: (Callable[[torch.Tensor], torch.Tensor]) Normalization function
        :param augmentation: (Tuple[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]) Tuple of
        augmentation functions to be applied
        :param augmentation_p: (float) Probability that an augmentation is utilized
        :param downscale: (bool) If true images and segmentation maps will be downscaled to a size of 256 X 256
        :param downscale_shape: (Tuple[int, int]) Target shape is downscale is utilized
        :param return_absolute_bounding_box: (Bool) If true the absolute bb is returned else the relative bb is returned
        :param two_classes: (bool) If true only two classes, trap and cell, will be utilized
        """
        # Save parameters
        self.normalize = normalize
        self.normalization_function = normalization_function
        self.augmentation = augmentation
        self.augmentation_p = augmentation_p
        self.return_absolute_bounding_box = return_absolute_bounding_box
        self.downscale = downscale
        self.downscale_shape = downscale_shape
        self.two_class = two_classes
        # Get paths of input images
        self.inputs = []
        for file in sorted(os.listdir(os.path.join(path, "inputs"))):
            self.inputs.append(os.path.join(path, "inputs", file))
        # Get paths of instances
        self.instances = []
        for file in sorted(os.listdir(os.path.join(path, "instances"))):
            self.instances.append(os.path.join(path, "instances", file))
        # Get paths of class labels
        self.class_labels = []
        for file in sorted(os.listdir(os.path.join(path, "classes"))):
            self.class_labels.append(os.path.join(path, "classes", file))
        # Get paths of bounding boxes
        self.bounding_boxes = []
        for file in sorted(os.listdir(os.path.join(path, "bounding_boxes"))):
            self.bounding_boxes.append(os.path.join(path, "bounding_boxes", file))

    def __len__(self) -> int:
        """
        Method returns the length of the dataset
        :return: (int) Length of the dataset
        """
        return len(self.inputs)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item method
        :param item: (int) Item to be returned of the dataset
        :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) Tuple including input image,
        bounding box, class label and instances.
        """
        # Load data
        input = torch.load(self.inputs[item]).unsqueeze(dim=0)
        instances = torch.load(self.instances[item])
        bounding_boxes = torch.load(self.bounding_boxes[item])
        class_labels = torch.load(self.class_labels[item])
        # Encode class labels as one-hot
        if self.two_class:
            class_labels = misc.to_one_hot(class_labels.clamp(max=2.0), num_classes=2 + 1)
        else:
            class_labels = misc.to_one_hot(class_labels, num_classes=3 + 1)
        # Normalize input if utilized
        if self.normalize:
            input = self.normalization_function(input)
        # Apply augmentation if needed
        if np.random.random() < self.augmentation_p and self.augmentation is not None:
            # Get augmentation
            augmentation_to_be_applied = np.random.choice(self.augmentation)
            # Apply augmentation
            if augmentation_to_be_applied.need_labels():
                input, instances, bounding_boxes = augmentation_to_be_applied(input, instances, bounding_boxes)
            else:
                input = augmentation_to_be_applied(input)
        # Downscale data to 256 x 256 if utilized
        if self.downscale:
            # Apply height and width
            bounding_boxes[..., [0, 2]] = bounding_boxes[..., [0, 2]] * (self.downscale_shape[0] / input.shape[-1])
            bounding_boxes[..., [1, 3]] = bounding_boxes[..., [1, 3]] * (self.downscale_shape[1] / input.shape[-2])
            input = F.interpolate(input=input.unsqueeze(dim=0),
                                  size=self.downscale_shape, mode="bicubic", align_corners=False)[0]
            instances = (F.interpolate(input=instances.unsqueeze(dim=0),
                                       size=self.downscale_shape, mode="bilinear", align_corners=False)[
                             0] > 0.75).float()
        # Convert absolute bounding box to relative bounding box of utilized
        if not self.return_absolute_bounding_box:
            bounding_boxes = misc.absolute_bounding_box_to_relative(bounding_boxes=bounding_boxes,
                                                                    height=input.shape[1], width=input.shape[2])
        return input, instances, misc.bounding_box_x0y0x1y1_to_xcycwh(bounding_boxes), class_labels


def collate_function_cell_instance_segmentation(
        batch: List[Tuple[torch.Tensor]]) -> \
        Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Collate function of instance segmentation dataset.
    :param batch: (Tuple[Iterable[torch.Tensor], Iterable[torch.Tensor], Iterable[torch.Tensor], Iterable[torch.Tensor]])
    Batch of input data, instances maps, bounding boxes and class labels
    :return: (Tuple[torch.Tensor, Iterable[torch.Tensor], Iterable[torch.Tensor], Iterable[torch.Tensor]]) Batched input
    data, instances, bounding boxes and class labels are stored in a list due to the different instances.
    """
    return torch.stack([input_samples[0] for input_samples in batch], dim=0), \
           [input_samples[1] for input_samples in batch], \
           [input_samples[2] for input_samples in batch], \
           [input_samples[3] for input_samples in batch]


