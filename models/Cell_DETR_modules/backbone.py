from typing import Tuple, Type, List

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """
    This class implements a simple Res-Net block with two convolutions, each followed by a normalization step and an
    activation function, and a residual mapping.
    """

    def __init__(self, in_channels: int, out_channels: int, convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d, activation: Type = nn.PReLU,
                 pooling: Type = nn.AvgPool2d) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param convolution: (Type) Type of convolution to be utilized
        :param normalization: (Type) Type of normalization to be utilized
        :param activation: (Type) Type of activation function to be utilized
        :param pooling: (Type) Type of pooling operation to be utilized
        """
        # Call super constructor
        super(ResNetBlock, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=in_channels, affine=True, track_running_stats=True),
            activation(),
            convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=out_channels, affine=True, track_running_stats=True),
            activation()
        )
        # Init residual mapping
        self.residual_mapping = convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            stride=(1, 1), padding=(0, 0),
                                            bias=True) if in_channels != out_channels else nn.Identity()
        # Init pooling
        self.pooling = pooling(kernel_size=(2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        :param input: (torch.Tensor) Input tensor of shape (batch size, input channels, height, width)
        :return: (torch.Tensor) Output tensor of shape (batch size, output channels, height // 2, width // 2)
        """
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform residual mapping
        output = output + self.residual_mapping(input)
        # Perform pooling
        output = self.pooling(output)
        return output


class StandardBlock(nn.Module):
    """
    This class implements a standard convolution block including two convolutions, each followed by a normalization and
    an activation function.
    """

    def __init__(self, in_channels: int, out_channels: int, convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d, activation: Type = nn.PReLU,
                 pooling: Type = nn.AvgPool2d) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param convolution: (Type) Type of convolution to be utilized
        :param normalization: (Type) Type of normalization to be utilized
        :param activation: (Type) Type of activation function to be utilized
        :param pooling: (Type) Type of pooling operation to be utilized
        """
        # Call super constructor
        super(StandardBlock, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=in_channels, affine=True, track_running_stats=True),
            activation(),
            convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=out_channels, affine=True, track_running_stats=True),
            activation()
        )
        # Init pooling
        self.pooling = pooling(kernel_size=(2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        :param input: (torch.Tensor) Input tensor of shape (batch size, input channels, height, width)
        :return: (torch.Tensor) Output tensor of shape (batch size, output channels, height // 2, width // 2)
        """
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform pooling
        output = self.pooling(output)
        return output


class DenseNetBlock(nn.Module):
    """
    This class implements a Dense-Net block including two convolutions, each followed by a normalization and
    an activation function, and skip connections for each convolution
    """

    def __init__(self, in_channels: int, out_channels: int, convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d, activation: Type = nn.PReLU,
                 pooling: Type = nn.AvgPool2d) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param convolution: (Type) Type of convolution to be utilized
        :param normalization: (Type) Type of normalization to be utilized
        :param activation: (Type) Type of activation function to be utilized
        :param pooling: (Type) Type of pooling operation to be utilized
        """
        # Call super constructor
        super(DenseNetBlock, self).__init__()
        # Calc convolution filters
        filters, additional_filters = divmod(out_channels - in_channels, 2)
        # Init fist mapping
        self.first_mapping = nn.Sequential(
            convolution(in_channels=in_channels, out_channels=filters, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), bias=True),
            normalization(num_features=filters, affine=True, track_running_stats=True),
            activation()
        )
        # Init second mapping
        self.second_mapping = nn.Sequential(
            convolution(in_channels=in_channels + filters, out_channels=filters + additional_filters,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            normalization(num_features=filters + additional_filters, affine=True, track_running_stats=True),
            activation()
        )
        # Init pooling
        self.pooling = pooling(kernel_size=(2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        :param input: (torch.Tensor) Input tensor of shape (batch size, input channels, height, width)
        :return: (torch.Tensor) Output tensor of shape (batch size, output channels, height // 2, width // 2)
        """
        # Perform main mapping
        output = torch.cat((input, self.first_mapping(input)), dim=1)
        # Perform main mapping
        output = torch.cat((output, self.second_mapping(output)), dim=1)
        # Perform pooling
        output = self.pooling(output)
        return output


class Backbone(nn.Module):
    """
    This class implements the backbone network.
    """

    def __init__(self, channels: Tuple[Tuple[int, int], ...] = ((1, 16), (16, 32), (32, 64), (64, 128), (128, 256)),
                 block: Type = StandardBlock, convolution: Type = nn.Conv2d, normalization: Type = nn.InstanceNorm2d,
                 activation: Type = nn.PReLU, pooling: Type = nn.AvgPool2d) -> None:
        """
        Constructor method
        :param channels: (Tuple[Tuple[int, int]]) In and output channels of each block
        :param block: (Type) Basic block to be used
        :param convolution: (Type) Type of convolution to be utilized
        :param normalization: (Type) Type of normalization to be utilized
        :param activation: (Type) Type of activation function to be utilized
        :param pooling: (Type) Type of pooling operation to be utilized
        """
        # Call super constructor
        super(Backbone, self).__init__()
        # Init input convolution
        self.input_convolution = nn.Sequential(convolution(in_channels=channels[0][0], out_channels=channels[0][1],
                                                           kernel_size=(7, 7), stride=(1, 1), padding=(3, 3),
                                                           bias=True),
                                               pooling(kernel_size=(2, 2)))
        # Init blocks
        self.blocks = nn.ModuleList([
            block(in_channels=channel[0], out_channels=channel[1], convolution=convolution, normalization=normalization,
                  activation=activation, pooling=pooling) for channel in channels])
        # Init weights
        for module in self.modules():
            # Case if module is convolution
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)
            # Deformable convolution is already initialized in the right way
            # Init PReLU
            elif isinstance(module, nn.PReLU):
                nn.init.constant_(module.weight, 0.2)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass
        :param input: (torch.Tensor) Input image of shape (batch size, input channels, height, width)
        :return: (torch.Tensor) Output tensor (batch size, output channels, height // 2 ^ depth, width // 2 ^ depth) and
        features of each stage of the backbone network
        """
        # Init list to store feature maps
        feature_maps = []
        # Forward pass of all blocks
        for index, block in enumerate(self.blocks):
            if index == 0:
                input = block(input) + self.input_convolution(input)
                feature_maps.append(input)
            else:
                input = block(input)
                feature_maps.append(input)
        return input, feature_maps
