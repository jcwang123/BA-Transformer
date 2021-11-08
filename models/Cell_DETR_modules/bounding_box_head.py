from typing import Tuple, Type

import torch
import torch.nn as nn


class BoundingBoxHead(nn.Module):
    """
    This class implements the feed forward bounding box head as proposed in:
    https://arxiv.org/abs/2005.12872
    """

    def __init__(self, features: Tuple[Tuple[int, int]] = ((256, 64), (64, 16), (16, 4)),
                 activation: Type = nn.PReLU) -> None:
        """
        Constructor method
        :param features: (Tuple[Tuple[int, int]]) Number of input and output features in each layer
        :param activation: (Type) Activation function to be utilized
        """
        # Call super constructor
        super(BoundingBoxHead, self).__init__()
        # Init layers
        self.layers = []
        for index, feature in enumerate(features):
            if index < len(features) - 1:
                self.layers.extend([nn.Linear(in_features=feature[0], out_features=feature[1]), activation()])
            else:
                self.layers.append(nn.Linear(in_features=feature[0], out_features=feature[1]))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of shape (batch size, instances, features)
        :return: (torch.Tensor) Output tensor of shape (batch size, instances, classes + 1 (no object))
        """
        return self.layers(input)
