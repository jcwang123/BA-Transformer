import sys
from typing import Tuple, Type, Iterable
sys.path.insert(0, '/home/chenfei/my_codes/TransformerCode-master/lib/Cell_DETR_master/')
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
# from modules.modulated_deform_conv import ModulatedDeformConvPack
# from pade_activation_unit.utils import PAU
from torch.nn.modules import Conv2d,LeakyReLU

# A
conv = Conv2d
act = LeakyReLU

from backbone import Backbone, DenseNetBlock, StandardBlock, ResNetBlock
from segmentation import MultiHeadAttention, SegmentationHead, SingleSegmentationHead, ResFeaturePyramidBlock, ResPACFeaturePyramidBlock
from transformer import Transformer,TransformerEncoder,TransformerEncoderLayer

class CellDETR(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 number_of_query_positions: int = 1,
                 hidden_features=128,
                 backbone_channels: Tuple[Tuple[int, int], ...] = (
                         (3, 64), (64, 128), (128, 256), (256, 256)),
                 backbone_block: Type = ResNetBlock, backbone_convolution: Type = conv,
                 backbone_normalization: Type = nn.BatchNorm2d, backbone_activation: Type = act,
                 backbone_pooling: Type = nn.AvgPool2d,
                 bounding_box_head_features: Tuple[Tuple[int, int], ...] = ((128, 64), (64, 16), (16, 4)),
                 bounding_box_head_activation: Type = act,
                 classification_head_activation: Type = act,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 2,
                 dropout: float = 0.0,
                 transformer_attention_heads: int = 8,
                 transformer_activation: Type = act,
                 segmentation_attention_heads: int = 8,
                 segmentation_head_channels: Tuple[Tuple[int, int], ...] = (
                         (128+8, 64), (64, 32),(32, 16)),
                 segmentation_head_feature_channels: Tuple[int, ...] = (256, 128, 64),
                 segmentation_head_block: Type = ResPACFeaturePyramidBlock,
                 segmentation_head_convolution: Type = conv,
                 segmentation_head_normalization: Type = nn.InstanceNorm2d,
                 segmentation_head_activation: Type = act,
                 segmentation_head_final_activation: Type = nn.Sigmoid) -> None:

        # Call super constructor
        super(CellDETR, self).__init__()
        # Init backbone
        self.backbone = Backbone(channels=backbone_channels, block=backbone_block, convolution=backbone_convolution,
                                 normalization=backbone_normalization, activation=backbone_activation,
                                 pooling=backbone_pooling)
        # Init convolution mapping to match transformer dims
        self.convolution_mapping = nn.Conv2d(in_channels=backbone_channels[-1][-1], out_channels=hidden_features,
                                             kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        # Init query positions
        self.query_positions = nn.Parameter(
            data=torch.randn(number_of_query_positions, hidden_features, dtype=torch.float),
            requires_grad=True)
        # Init embeddings
        self.row_embedding = nn.Parameter(data=torch.randn(50, hidden_features // 2, dtype=torch.float),
                                          requires_grad=True)
        self.column_embedding = nn.Parameter(data=torch.randn(50, hidden_features // 2, dtype=torch.float),
                                             requires_grad=True)
        # Init transformer
        self.transformer = Transformer(d_model=hidden_features, nhead=transformer_attention_heads,
                                       num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                       dropout=dropout, dim_feedforward=4 * hidden_features,
                                       activation=transformer_activation)
        
        # Init segmentation attention head
        self.segmentation_attention_head = MultiHeadAttention(query_dimension=hidden_features,
                                                              hidden_features=hidden_features,
                                                              number_of_heads=segmentation_attention_heads,
                                                              dropout=dropout)
        # Init segmentation head
        self.segmentation_head = SegmentationHead(channels=segmentation_head_channels,
                                                  feature_channels=segmentation_head_feature_channels,
                                                  convolution=segmentation_head_convolution,
                                                  normalization=segmentation_head_normalization,
                                                  activation=segmentation_head_activation,
                                                  block=segmentation_head_block,
                                                  number_of_query_positions=number_of_query_positions,
                                                  softmax=isinstance(segmentation_head_final_activation(), nn.Softmax))
        # Init final segmentation activation
        self.segmentation_final_activation = segmentation_head_final_activation(dim=1) if isinstance(
            segmentation_head_final_activation(), nn.Softmax) else segmentation_head_final_activation()

        self.point_pre_layer = nn.Conv2d(hidden_features, 1, kernel_size = 1)
        
    def get_parameters(self, lr_main: float = 1e-04, lr_backbone: float = 1e-05) -> Iterable:
        return [{'params': self.backbone.parameters(), 'lr': lr_backbone},
                {'params': self.convolution_mapping.parameters(), 'lr': lr_main},
                {'params': [self.row_embedding], 'lr': lr_main},
                {'params': [self.column_embedding], 'lr': lr_main},
                {'params': self.transformer.parameters(), 'lr': lr_main},
                {'params': self.bounding_box_head.parameters(), 'lr': lr_main},
                {'params': self.class_head.parameters(), 'lr': lr_main},
                {'params': self.segmentation_attention_head.parameters(), 'lr': lr_main},
                {'params': self.segmentation_head.parameters(), 'lr': lr_main}]

    def get_segmentation_head_parameters(self, lr: float = 1e-05) -> Iterable:
        return [{'params': self.segmentation_attention_head.parameters(), 'lr': lr},
                {'params': self.segmentation_head.parameters(), 'lr': lr}]

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, feature_list = self.backbone(input)
        features = self.convolution_mapping(features)
        height, width = features.shape[2:]
        batch_size = features.shape[0]
        positional_embeddings = torch.cat([self.column_embedding[:height].unsqueeze(dim=0).repeat(height, 1, 1),
                                           self.row_embedding[:width].unsqueeze(dim=1).repeat(1, width, 1)],
                                          dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        latent_tensor, features_encoded = self.transformer(features, None, self.query_positions, positional_embeddings)
        latent_tensor = latent_tensor.permute(2, 0, 1)
#         point = self.point_pre_layer(features_encoded)
#         point = torch.sigmoid(point)
#         Feature = point*features_encoded + features_encoded
        bounding_box_attention_masks = self.segmentation_attention_head(
            latent_tensor, features_encoded.contiguous())
        instance_segmentation_prediction = self.segmentation_head(features.contiguous(),
                                              bounding_box_attention_masks.contiguous(),
                                              feature_list[-2::-1])
        return self.segmentation_final_activation(instance_segmentation_prediction).clone()

if __name__ == '__main__':
    # Init model
    detr = CellDETR()
    # Print number of parameters
    print("DETR # parameters", sum([p.numel() for p in detr.parameters()]))
    # Model into eval mode
#     detr.eval()
    image = torch.randn(5,3,512,512)
#     point = torch.randn(5,1,128,128)
    # Predict
    segmentation_prediction = detr(image)
    
    # Print shapes
#     print(segmentation_prediction.shape)
#     print(segmentation_prediction.max(), segmentation_prediction.min())
#     loss = segmentation_prediction.sum()
#     loss.backward()
