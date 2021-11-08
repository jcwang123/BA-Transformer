from typing import Tuple

import torch
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import numpy as np


class Augmentation(object):
    """
    Super class for all augmentations.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        pass

    def need_labels(self) -> None:
        """
        Method should return if the labels are needed for the augmentation
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> None:
        """
        Call method is used to apply the augmentation
        :param args: Will be ignored
        :param kwargs: Will be ignored
        """
        raise NotImplementedError()


class VerticalFlip(Augmentation):
    """
    This class implements vertical flipping for instance segmentation.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(VerticalFlip, self).__init__()

    def need_labels(self) -> bool:
        """
        Method returns that the labels are needed for the augmentation
        :return: (Bool) True will be returned
        """
        return True

    def __call__(self, input: torch.tensor, instances: torch.tensor,
                 bounding_boxes: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Flipping augmentation (only horizontal)
        :param image: (torch.Tensor) Input image of shape [channels, height, width]
        :param instances: (torch.Tenor) Instances segmentation maps of shape [instances, height, width]
        :param bounding_boxes: (torch.Tensor) Bounding boxes of shape [instances, 4 (x1, y1, x2, y2)]
        :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) Input flipped, instances flipped & BBs flipped
        """
        # Flip input
        input_flipped = input.flip(dims=(2,))
        # Flip instances
        instances_flipped = instances.flip(dims=(2,))
        # Flip bounding boxes
        image_center = torch.tensor((input.shape[2] // 2, input.shape[1] // 2))
        bounding_boxes[:, [0, 2]] += 2 * (image_center - bounding_boxes[:, [0, 2]])
        bounding_boxes_w = torch.abs(bounding_boxes[:, 0] - bounding_boxes[:, 2])
        bounding_boxes[:, 0] -= bounding_boxes_w
        bounding_boxes[:, 2] += bounding_boxes_w
        return input_flipped, instances_flipped, bounding_boxes


class ElasticDeformation(Augmentation):
    """
    This class implement random elastic deformation of a given input image
    """

    def __init__(self, alpha: float = 125, sigma: float = 20) -> None:
        """
        Constructor method
        :param alpha: (float) Alpha coefficient which represents the scaling
        :param sigma: (float) Sigma coefficient which represents the elastic factor
        """
        # Call super constructor
        super(ElasticDeformation, self).__init__()
        # Save parameters
        self.alpha = alpha
        self.sigma = sigma

    def need_labels(self) -> bool:
        """
        Method returns that the labels are needed for the augmentation
        :return: (Bool) True will be returned
        """
        return False

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Method applies the random elastic deformation
        :param image: (torch.Tensor) Input image
        :return: (torch.Tensor) Transformed input image
        """
        # Convert torch tensor to numpy array for scipy
        image = image.numpy()
        # Save basic shape
        shape = image.shape[1:]
        # Sample offsets
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        # Perform deformation
        for index in range(image.shape[0]):
            image[index] = map_coordinates(image[index], indices, order=1).reshape(shape)
        return torch.from_numpy(image)


class NoiseInjection(Augmentation):
    """
    This class implements vertical flipping for instance segmentation.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.25) -> None:
        """
        Constructor method
        :param mean: (Optional[float]) Mean of the gaussian noise
        :param std: (Optional[float]) Standard deviation of the gaussian noise
        """
        # Call super constructor
        super(NoiseInjection, self).__init__()
        # Save parameter
        self.mean = mean
        self.std = std

    def need_labels(self) -> bool:
        """
        Method returns that the labels are needed for the augmentation
        :return: (Bool) False will be returned
        """
        return False

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """
        Method injects gaussian noise to the given input image
        :param image: (torch.Tensor) Input image
        :return: (torch.Tensor) Transformed input image
        """
        # Get noise
        noise = self.mean + torch.randn_like(input) * self.std
        # Apply nose to image
        input = input + noise
        return input
