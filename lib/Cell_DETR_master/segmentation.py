from typing import Type, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# from pixel_adaptive_convolution.pac import PacConv2d


class MultiHeadAttention(nn.Module):
    """
    This class implements a multi head attention module like proposed in:
    https://arxiv.org/abs/2005.12872
    """
    def __init__(self,
                 query_dimension: int = 64,
                 hidden_features: int = 64,
                 number_of_heads: int = 16,
                 dropout: float = 0.0) -> None:
        """
        Constructor method
        :param query_dimension: (int) Dimension of query tensor
        :param hidden_features: (int) Number of hidden features in detr
        :param number_of_heads: (int) Number of prediction heads
        :param dropout: (float) Dropout factor to be utilized
        """
        # Call super constructor
        super(MultiHeadAttention, self).__init__()
        # Save parameters
        self.hidden_features = hidden_features
        self.number_of_heads = number_of_heads
        self.dropout = dropout
        # Init layer
        self.layer_box_embedding = nn.Linear(in_features=query_dimension,
                                             out_features=hidden_features,
                                             bias=True)
        # Init convolution layer
        self.layer_image_encoding = nn.Conv2d(in_channels=query_dimension,
                                              out_channels=hidden_features,
                                              kernel_size=(1, 1),
                                              stride=(1, 1),
                                              padding=(0, 0),
                                              bias=True)
        # Init normalization factor
        self.normalization_factor = torch.tensor(self.hidden_features /
                                                 self.number_of_heads,
                                                 dtype=torch.float).sqrt()

        # Linear
        #self.linear = nn.Linear(in_features=number_of_heads, out_features=1)

    def forward(self, input_box_embeddings: torch.Tensor,
                input_image_encoding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input_box_embeddings: (torch.Tensor) Bounding box embeddings
        :param input_image_encoding: (torch.Tensor) Encoded image of the transformer encoder
        :return: (torch.Tensor) Attention maps of shape (batch size, n, m, height, width)
        """
        # Map box embeddings
        output_box_embeddings = self.layer_box_embedding(input_box_embeddings)
        # Map image features
        output_image_encoding = self.layer_image_encoding(input_image_encoding)
        # Reshape output box embeddings
        output_box_embeddings = output_box_embeddings.view(
            output_box_embeddings.shape[0], output_box_embeddings.shape[1],
            self.number_of_heads, self.hidden_features // self.number_of_heads)
        # Reshape output image encoding
        output_image_encoding = output_image_encoding.view(
            output_image_encoding.shape[0], self.number_of_heads,
            self.hidden_features // self.number_of_heads,
            output_image_encoding.shape[-2], output_image_encoding.shape[-1])
        # Combine tensors and normalize
        output = torch.einsum(
            "bqnc,bnchw->bqnhw",
            output_box_embeddings * self.normalization_factor,
            output_image_encoding)
        # Apply softmax
        output = F.softmax(output.flatten(start_dim=2), dim=-1).view_as(output)

        # Linear: to generate one map
        #b, _, _, h, w = output.shape
        #output = torch.sigmoid(self.linear(output.flatten(start_dim=3).permute(0,1,3,2))).view(b,1,h,w)

        # Perform dropout if utilized
        if self.dropout > 0.0:
            output = F.dropout(input=output,
                               p=self.dropout,
                               training=self.training)


#         print("MultiHead Attention",output.shape)
        return output.contiguous()


class ResFeaturePyramidBlock(nn.Module):
    """
    This class implements a residual feature pyramid block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 feature_channels: int,
                 convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d,
                 activation: Type = nn.PReLU,
                 dropout: float = 0.0) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param feature_channels: (int) Number of channels present in the feature map
        :param convolution: (Type) Type of convolution to be utilized
        :param normalization: (Type) Type of normalization to be used
        :param activation: (Type) Type of activation function to be utilized
        :param dropout: (float) Dropout factor to be applied after upsampling is performed
        """
        # Call super constructor
        super(ResFeaturePyramidBlock, self).__init__()
        # Save parameter
        self.dropout = dropout
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels,
                        out_channels=out_channels // 2,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        bias=True),
            normalization(num_features=out_channels // 2,
                          affine=True,
                          track_running_stats=True), activation(),
            convolution(in_channels=out_channels // 2,
                        out_channels=out_channels // 2,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        bias=True),
            normalization(num_features=out_channels // 2,
                          affine=True,
                          track_running_stats=True), activation())
        # Init residual mapping
        self.residual_mapping = convolution(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True) if in_channels != out_channels // 2 else nn.Identity()
        # Init upsampling
        self.upsampling = nn.Upsample(scale_factor=(2, 2),
                                      mode='bicubic',
                                      align_corners=False)
        # Init feature mapping
        self.feature_mapping = convolution(in_channels=feature_channels,
                                           out_channels=out_channels // 2,
                                           kernel_size=(1, 1),
                                           stride=(1, 1),
                                           padding=(0, 0),
                                           bias=True)

    def forward(self, input: torch.Tensor,
                feature: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of shape (batch size * number of heads, in channels, height, width)
        :param feature: (torch.Tensor) Feature tensor of backbone of shape (batch size, channels, height, width)
        :return: (torch.Tensor) Output tensor (batch size * number of heads, out channels, height * 2, width * 2)
        """
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform residual mapping
        output = output + self.residual_mapping(input)
        # Perform upsampling
        output = self.upsampling(output)
        # Perform dropout if utilized
        if self.dropout > 0.0:
            output = F.dropout(output, p=self.dropout, training=self.training)
        # Add mapped feature
        output = torch.cat(
            (output, self.feature_mapping(feature).unsqueeze(dim=1).repeat(
                1, int(output.shape[0] / feature.shape[0]), 1, 1, 1).flatten(
                    0, 1).contiguous()),
            dim=1)
        return output


class ResPACFeaturePyramidBlock(nn.Module):
    """
    This class implements a residual feature pyramid block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 feature_channels: int,
                 convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d,
                 activation: Type = nn.PReLU,
                 dropout: float = 0.0) -> None:
        # Call super constructor
        super(ResPACFeaturePyramidBlock, self).__init__()
        # Save parameter
        self.dropout = dropout
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        bias=True),
            normalization(num_features=out_channels,
                          affine=True,
                          track_running_stats=True), activation(),
            convolution(in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        bias=True),
            normalization(num_features=out_channels,
                          affine=True,
                          track_running_stats=True), activation())
        # Init residual mapping
        self.residual_mapping = convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True) if in_channels != out_channels else nn.Identity()
        # Init upsampling
        self.upsampling = nn.Upsample(scale_factor=(2, 2),
                                      mode='bicubic',
                                      align_corners=False)
        # Init feature mapping
        self.feature_mapping = convolution(in_channels=feature_channels,
                                           out_channels=out_channels,
                                           kernel_size=(1, 1),
                                           stride=(1, 1),
                                           padding=(0, 0),
                                           bias=True)
        # Init pixel adaptive convolution
        self.pixel_adaptive_convolution = PacConv2d(in_channels=out_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=(5, 5),
                                                    padding=(2, 2),
                                                    stride=(1, 1),
                                                    bias=True,
                                                    normalize_kernel=True)

    def forward(self, input: torch.Tensor,
                feature: torch.Tensor) -> torch.Tensor:
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform residual mapping
        output = output + self.residual_mapping(input)
        # Perform upsampling
        output = self.upsampling(output)
        # Perform dropout if utilized
        if self.dropout > 0.0:
            output = F.dropout(output, p=self.dropout, training=self.training)
        # Perform PAC
        output = self.pixel_adaptive_convolution(
            output,
            self.feature_mapping(feature).unsqueeze(dim=1).repeat(
                1, int(output.shape[0] / feature.shape[0]), 1, 1,
                1).flatten(0, 1).contiguous())
        return output


class FinalBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d,
                 activation: Type = nn.PReLU,
                 number_of_query_positions: int = None) -> None:
        # Call super constructor
        super(FinalBlock, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        bias=True),
            normalization(num_features=in_channels,
                          affine=True,
                          track_running_stats=True), activation(),
            convolution(in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        bias=True),
            normalization(num_features=in_channels,
                          affine=True,
                          track_running_stats=True), activation())
        # Init upsampling
        self.upsampling = nn.Upsample(scale_factor=(2, 2),
                                      mode='bicubic',
                                      align_corners=False)
        # Init final mapping
        self.final_mapping = nn.Sequential(
            convolution(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        bias=True), activation(),
            convolution(in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(0, 0),
                        bias=True))

    def forward(self, input: torch.Tensor, batch_size: int) -> torch.Tensor:
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform residual mapping
        output = output + input
        # Perform upsampling
        output = self.upsampling(output)
        # Perform final mapping
        output = self.final_mapping(output)
        return output.view(batch_size, -1, output.shape[2], output.shape[3])


class FinalBlockReshaped(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d,
                 activation: Type = nn.PReLU,
                 number_of_query_positions: int = 1) -> None:
        # Call super constructor
        super(FinalBlockReshaped, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            convolution(in_channels=in_channels * number_of_query_positions,
                        out_channels=in_channels * number_of_query_positions //
                        2,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        bias=True),
            normalization(num_features=in_channels *
                          number_of_query_positions // 2,
                          affine=True,
                          track_running_stats=True), activation(),
            convolution(
                in_channels=in_channels * number_of_query_positions // 2,
                out_channels=in_channels * number_of_query_positions // 8,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=True),
            normalization(num_features=in_channels *
                          number_of_query_positions // 8,
                          affine=True,
                          track_running_stats=True), activation())
        # Init residual mapping
        self.residual_mapping = convolution(
            in_channels=in_channels * number_of_query_positions,
            out_channels=in_channels * number_of_query_positions // 8,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True)
        # Init upsampling
        self.upsampling = nn.Upsample(scale_factor=(2, 2),
                                      mode='bicubic',
                                      align_corners=False)
        # Init final mapping
        self.final_mapping = nn.Sequential(
            convolution(in_channels=in_channels * number_of_query_positions //
                        8,
                        out_channels=in_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        bias=True), activation(),
            convolution(in_channels=in_channels,
                        out_channels=out_channels * number_of_query_positions,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(0, 0),
                        bias=True))

    def forward(self, input: torch.Tensor, batch_size: int) -> torch.Tensor:
        # Reshape input
        input = input.view(batch_size, -1, input.shape[2], input.shape[3])
        # Perform main mapping
        output = self.main_mapping(input)
        # Perform residual mapping
        output = output + self.residual_mapping(input)
        # Perform upsampling
        output = self.upsampling(output)
        # Perform final mapping
        output = self.final_mapping(output)
        return output


class SegmentationHead(nn.Module):
    def __init__(self,
                 channels: Tuple[Tuple[int, int],
                                 ...] = ((80, 32), (32, 16), (16, 8), (8, 4)),
                 feature_channels: Tuple[int, ...] = (128, 64, 32, 16),
                 convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d,
                 activation: Type = nn.PReLU,
                 block: Type = ResPACFeaturePyramidBlock,
                 dropout: float = 0.0,
                 number_of_query_positions: int = 12,
                 softmax: bool = True) -> None:
        # Call super constructor
        super(SegmentationHead, self).__init__()
        # Init blocks
        self.blocks = nn.ModuleList()
        for channel, feature_channel in zip(channels, feature_channels):
            self.blocks.append(
                block(in_channels=channel[0],
                      out_channels=channel[1],
                      feature_channels=feature_channel,
                      convolution=convolution,
                      normalization=normalization,
                      activation=activation,
                      dropout=dropout))
        # Init final block
        if softmax:
            self.final_block = FinalBlockReshaped(
                in_channels=channels[-1][-1],
                out_channels=1,
                convolution=convolution,
                normalization=normalization,
                activation=activation,
                number_of_query_positions=number_of_query_positions)
        else:
            self.final_block = FinalBlock(
                in_channels=channels[-1][-1],
                out_channels=1,
                convolution=convolution,
                normalization=normalization,
                activation=activation,
                number_of_query_positions=number_of_query_positions)

    def forward(self, features: torch.Tensor,
                segmentation_attention_head: torch.Tensor,
                backbone_features: torch.Tensor) -> torch.Tensor:
        # Construct input to convolutions
        input = torch.cat([
            features.unsqueeze(dim=1).repeat(
                1, segmentation_attention_head.shape[1], 1, 1, 1).flatten(
                    0, 1),
            segmentation_attention_head.flatten(0, 1)
        ],
                          dim=1).contiguous()
        #         input = features_encoded
        # Forward pass of all blocks
        for block, feature in zip(self.blocks, backbone_features):
            input = block(input, feature)


#         print(input.shape) #5x16x64x64
# Forward pass of final block
        output = self.final_block(input, features.shape[0])
        #         print(output.shape) # 5x1x128x128
        return output


class SingleSegmentationHead(nn.Module):
    def __init__(self,
                 channels: Tuple[Tuple[int, int],
                                 ...] = ((80, 32), (32, 16), (16, 8), (8, 4)),
                 feature_channels: Tuple[int, ...] = (128, 64, 32, 16),
                 convolution: Type = nn.Conv2d,
                 normalization: Type = nn.InstanceNorm2d,
                 activation: Type = nn.PReLU,
                 block: Type = ResPACFeaturePyramidBlock,
                 dropout: float = 0.0,
                 number_of_query_positions: int = 12,
                 softmax: bool = True) -> None:
        # Call super constructor
        super(SingleSegmentationHead, self).__init__()
        # Init blocks
        self.blocks = nn.ModuleList()
        for channel, feature_channel in zip(channels, feature_channels):
            self.blocks.append(
                block(in_channels=channel[0],
                      out_channels=channel[1],
                      feature_channels=feature_channel,
                      convolution=convolution,
                      normalization=normalization,
                      activation=activation,
                      dropout=dropout))
        # Init final block
        if softmax:
            self.final_block = FinalBlockReshaped(
                in_channels=channels[-1][-1],
                out_channels=1,
                convolution=convolution,
                normalization=normalization,
                activation=activation,
                number_of_query_positions=number_of_query_positions)
        else:
            self.final_block = FinalBlock(
                in_channels=channels[-1][-1],
                out_channels=1,
                convolution=convolution,
                normalization=normalization,
                activation=activation,
                number_of_query_positions=number_of_query_positions)

    def forward(self, features: torch.Tensor,
                segmentation_attention_head: torch.Tensor,
                backbone_features: torch.Tensor) -> torch.Tensor:
        # Construct input to convolutions
        #         print('Segmentation head')
        #         print(features.size(),segmentation_attention_head.size())
        #         input = features
        input = torch.cat([
            features.unsqueeze(dim=1).repeat(
                1, segmentation_attention_head.shape[1], 1, 1, 1).flatten(
                    0, 1),
            segmentation_attention_head.flatten(0, 1)
        ],
                          dim=1).contiguous()
        #         print(input.size())
        # Forward pass of all blocks
        for block, feature in zip(self.blocks, backbone_features):

            input = block(input, feature)
        # Forward pass of final block
        output = self.final_block(input, features.shape[0])
        #         print(output.size())
        return output
