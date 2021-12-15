from typing import Tuple, Type, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
#from modules.modulated_deform_conv import ModulatedDeformConvPack

from backbone import Backbone, DenseNetBlock, StandardBlock, ResNetBlock
from bounding_box_head import BoundingBoxHead
from segmentation import MultiHeadAttention, SegmentationHead, ResFeaturePyramidBlock, ResPACFeaturePyramidBlock
from transformer import BoundaryAwareTransformer, Transformer

from torch.nn.modules import Conv2d, LeakyReLU

# A
conv = Conv2d
act = LeakyReLU

# B
# conv = ModulatedDeformConvPack
# act = PAU


class CellDETR(nn.Module):
    """
    This class implements a DETR (Facebook AI) like instance segmentation model.
    """
    def __init__(
            self,
            num_classes: int = 3,
            number_of_query_positions: int = 12,
            hidden_features=128,
            backbone_channels: Tuple[Tuple[int, int],
                                     ...] = ((1, 64), (64, 128), (128, 256),
                                             (256, 256)),
            backbone_block: Type = ResNetBlock,
            backbone_convolution: Type = conv,
            backbone_normalization: Type = nn.BatchNorm2d,
            backbone_activation: Type = act,
            backbone_pooling: Type = nn.AvgPool2d,
            bounding_box_head_features: Tuple[Tuple[int, int],
                                              ...] = ((128, 64), (64, 16),
                                                      (16, 4)),
            bounding_box_head_activation: Type = act,
            classification_head_activation: Type = act,
            num_encoder_layers: int = 3,
            num_decoder_layers: int = 2,
            dropout: float = 0.0,
            transformer_attention_heads: int = 8,
            transformer_activation: Type = act,
            transformer_type: Type = BoundaryAwareTransformer,
            segmentation_attention_heads: int = 8,
            segmentation_head_channels: Tuple[Tuple[int, int],
                                              ...] = ((128 + 8, 128),
                                                      (128, 64), (64, 32)),
            segmentation_head_feature_channels: Tuple[int,
                                                      ...] = (256, 128, 64),
            segmentation_head_block: Type = ResPACFeaturePyramidBlock,
            segmentation_head_convolution: Type = conv,
            segmentation_head_normalization: Type = nn.InstanceNorm2d,
            segmentation_head_activation: Type = act,
            segmentation_head_final_activation: Type = nn.Sigmoid) -> None:
        """
        Constructor method
        :param num_classes: (int) Number of classes in the dataset
        :param number_of_query_positions: (int) Number of query positions
        :param hidden_features: (int) Number of hidden features in the transformer module
        :param backbone_channels: (Tuple[Tuple[int, int], ...]) In and output channels of each block in the backbone
        :param backbone_block: (Type) Type of block to be utilized in backbone
        :param backbone_convolution: (Type) Type of convolution to be utilized in the backbone
        :param backbone_normalization: (Type) Type of normalization to be used in the backbone
        :param backbone_activation: (Type) Type of activation function used in the backbone
        :param backbone_pooling: (Type) Type of pooling operation utilized in the backbone
        :param bounding_box_head_features: (Tuple[Tuple[int, int], ...]) In and output features of each layer in BB head
        :param bounding_box_head_activation: (Type) Type of activation function utilized in BB head
        :param classification_head_activation: (Type) Type of activation function utilized in classification head
        :param num_encoder_layers: (int) Number of layers in encoder part of the transformer module
        :param num_decoder_layers: (int) Number of layers in decoder part of the transformer module
        :param dropout: (float) Dropout factor used in transformer module and segmentation head
        :param transformer_attention_heads: (int) Number of attention heads in the transformer module
        :param transformer_activation: (Type) Type of activation function to be utilized in the transformer module
        :param segmentation_attention_heads: (int) Number of attention heads in the 2d multi head attention module
        :param segmentation_head_channels: (Tuple[Tuple[int, int], ...]) Number of in and output channels in seg. head
        :param segmentation_head_feature_channels: (Tuple[int, ...]) Backbone feature channels used in seg. head
        :param segmentation_head_block: (Type) Type of block to be utilized in segmentation head
        :param segmentation_head_convolution: (Type) Type of convolution utilized in segmentation head
        :param segmentation_head_normalization: (Type) Type of normalization used in segmentation head
        :param segmentation_head_activation: (Type) Type of activation used in segmentation head
        :param segmentation_head_final_activation: (Type) Type of activation function to be applied to the output pred
        """
        # Call super constructor
        super(CellDETR, self).__init__()
        # Init backbone
        self.backbone = Backbone(channels=backbone_channels,
                                 block=backbone_block,
                                 convolution=backbone_convolution,
                                 normalization=backbone_normalization,
                                 activation=backbone_activation,
                                 pooling=backbone_pooling)
        # Init convolution mapping to match transformer dims
        self.convolution_mapping = nn.Conv2d(
            in_channels=backbone_channels[-1][-1],
            out_channels=hidden_features,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True)
        # Init query positions
        self.query_positions = nn.Parameter(data=torch.randn(
            number_of_query_positions, hidden_features, dtype=torch.float),
                                            requires_grad=True)
        # Init embeddings
        self.row_embedding = nn.Parameter(data=torch.randn(50,
                                                           hidden_features //
                                                           2,
                                                           dtype=torch.float),
                                          requires_grad=True)
        self.column_embedding = nn.Parameter(data=torch.randn(
            50, hidden_features // 2, dtype=torch.float),
                                             requires_grad=True)
        # Init transformer
        self.transformer = transformer_type(
            d_model=hidden_features,
            nhead=transformer_attention_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            dim_feedforward=4 * hidden_features,
            activation=transformer_activation,
            normalize_before=False)
        # Init bounding box head
        self.bounding_box_head = BoundingBoxHead(
            features=bounding_box_head_features,
            activation=bounding_box_head_activation)
        # Init class head
        self.class_head = nn.Sequential(
            nn.Linear(in_features=hidden_features,
                      out_features=hidden_features // 2,
                      bias=True), classification_head_activation(),
            nn.Linear(in_features=hidden_features // 2,
                      out_features=num_classes + 1,
                      bias=True))
        # Init segmentation attention head
        self.segmentation_attention_head = MultiHeadAttention(
            query_dimension=hidden_features,
            hidden_features=hidden_features,
            number_of_heads=segmentation_attention_heads,
            dropout=dropout)
        # Init segmentation head
        self.segmentation_head = SegmentationHead(
            channels=segmentation_head_channels,
            feature_channels=segmentation_head_feature_channels,
            convolution=segmentation_head_convolution,
            normalization=segmentation_head_normalization,
            activation=segmentation_head_activation,
            block=segmentation_head_block,
            number_of_query_positions=number_of_query_positions,
            softmax=isinstance(segmentation_head_final_activation(),
                               nn.Softmax))
        # Init final segmentation activation
        self.segmentation_final_activation = segmentation_head_final_activation(
            dim=1) if isinstance(
                segmentation_head_final_activation(),
                nn.Softmax) else segmentation_head_final_activation()

    def get_parameters(self,
                       lr_main: float = 1e-04,
                       lr_backbone: float = 1e-05) -> Iterable:
        """
        Method returns all parameters of the model with different learning rates
        :param lr_main: (float) Leaning rate of all parameters which are not included in the backbone
        :param lr_backbone: (float) Leaning rate of the backbone parameters
        :return: (Iterable) Iterable object including the main parameters of the generator network
        """
        return [{
            'params': self.backbone.parameters(),
            'lr': lr_backbone
        }, {
            'params': self.convolution_mapping.parameters(),
            'lr': lr_main
        }, {
            'params': [self.row_embedding],
            'lr': lr_main
        }, {
            'params': [self.column_embedding],
            'lr': lr_main
        }, {
            'params': self.transformer.parameters(),
            'lr': lr_main
        }, {
            'params': self.bounding_box_head.parameters(),
            'lr': lr_main
        }, {
            'params': self.class_head.parameters(),
            'lr': lr_main
        }, {
            'params': self.segmentation_attention_head.parameters(),
            'lr': lr_main
        }, {
            'params': self.segmentation_head.parameters(),
            'lr': lr_main
        }]

    def get_segmentation_head_parameters(self, lr: float = 1e-05) -> Iterable:
        """
        Method returns all parameter of the segmentation head and the 2d multi head attention module
        :param lr: (float) Learning rate to be utilized
        :return: (Iterable) Iterable object including the parameters of the segmentation head
        """
        return [{
            'params': self.segmentation_attention_head.parameters(),
            'lr': lr
        }, {
            'params': self.segmentation_head.parameters(),
            'lr': lr
        }]

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        :param input: (torch.Tensor) Input image of shape (batch size, channels, height, width)
        :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) Class prediction, bounding box predictions and
        segmentation maps
        """
        # Get features from backbone
        features, feature_list = self.backbone(input)
        # Map features to the desired shape
        features = self.convolution_mapping(features)
        # Get height and width of features
        height, width = features.shape[2:]
        # Get batch size
        batch_size = features.shape[0]
        # Make positional embeddings
        positional_embeddings = torch.cat([
            self.column_embedding[:height].unsqueeze(dim=0).repeat(
                height, 1, 1),
            self.row_embedding[:width].unsqueeze(dim=1).repeat(1, width, 1)
        ],
                                          dim=-1).permute(
                                              2, 0, 1).unsqueeze(0).repeat(
                                                  batch_size, 1, 1, 1)
        latent_tensor, features_encoded = self.transformer(
            features, None, self.query_positions, positional_embeddings)
        latent_tensor = latent_tensor.permute(2, 0, 1)
        # Get class prediction
        class_prediction = F.softmax(self.class_head(latent_tensor),
                                     dim=2).clone()
        # Get bounding boxes
        bounding_box_prediction = self.bounding_box_head(latent_tensor)
        # Get bounding box attention masks for segmentation
        bounding_box_attention_masks = self.segmentation_attention_head(
            latent_tensor, features_encoded.contiguous())
        # Get instance segmentation prediction
        instance_segmentation_prediction = self.segmentation_head(
            features.contiguous(), bounding_box_attention_masks.contiguous(),
            feature_list[-2::-1])
        return class_prediction, \
               bounding_box_prediction.sigmoid().clone(), \
               self.segmentation_final_activation(instance_segmentation_prediction).clone()


if __name__ == '__main__':
    # Init model
    detr = CellDETR().cuda()
    # Print number of parameters
    print("DETR # parameters", sum([p.numel() for p in detr.parameters()]))
    # Model into eval mode
    detr.eval()
    # Predict
    class_prediction, bounding_box_prediction, instance_segmentation_prediction = detr(
        torch.randn(2, 1, 128, 128).cuda())
    # Print shapes
    print(class_prediction.shape)
    print(bounding_box_prediction.shape)
    print(instance_segmentation_prediction.shape)
    # Calc pseudo loss and perform backward pass
    loss = class_prediction.sum() + bounding_box_prediction.sum(
    ) + instance_segmentation_prediction.sum()
    loss.backward()
