import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from lib.Cell_DETR_master.segmentation import MultiHeadAttention
from lib.Cell_DETR_master.transformer import Transformer
from net.resnet import ResNet50_OS16, ResNet18_OS8
from net.ASPP import ASPP, ASPP_Bottleneck


class DeepLabV3(nn.Module):
    def __init__(self,
                 num_classes,
                 num_layers,
                 point_pred,
                 transformer_attention_heads=8,
                 num_encoder_layers=3,
                 num_decoder_layers=2,
                 hidden_features=128,
                 number_of_query_positions=1,
                 segmentation_attention_heads=8,
                 dropout=0):

        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes
        layers = num_layers
        self.point_pred = point_pred

        if layers == 18:
            self.resnet = ResNet18_OS8()
            self.aspp = ASPP(num_classes=self.num_classes)
        elif layers == 50:
            self.resnet = ResNet50_OS16()
            self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

        self.transformer_attention_heads = transformer_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_features = hidden_features
        self.number_of_query_positions = number_of_query_positions
        self.transformer_activation = nn.LeakyReLU
        self.segmentation_attention_heads = segmentation_attention_heads

        in_channels = 2048 if layers == 50 else 512
        self.convolution_mapping = nn.Conv2d(in_channels=in_channels,
                                             out_channels=hidden_features,
                                             kernel_size=(1, 1),
                                             stride=(1, 1),
                                             padding=(0, 0),
                                             bias=True)

        self.query_positions = nn.Parameter(data=torch.randn(
            number_of_query_positions, hidden_features, dtype=torch.float),
                                            requires_grad=True)

        self.row_embedding = nn.Parameter(data=torch.randn(100,
                                                           hidden_features //
                                                           2,
                                                           dtype=torch.float),
                                          requires_grad=True)
        self.column_embedding = nn.Parameter(data=torch.randn(
            100, hidden_features // 2, dtype=torch.float),
                                             requires_grad=True)

        self.transformer = Transformer(d_model=hidden_features,
                                       nhead=transformer_attention_heads,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dropout=dropout,
                                       dim_feedforward=4 * hidden_features,
                                       activation=self.transformer_activation)

        self.trans_out_conv = nn.Conv2d(
            hidden_features + segmentation_attention_heads, in_channels, 1, 1)

        self.segmentation_attention_head = MultiHeadAttention(
            query_dimension=hidden_features,
            hidden_features=hidden_features,
            number_of_heads=segmentation_attention_heads,
            dropout=dropout)
        self.point_pre_layer = nn.Conv2d(hidden_features, 1, kernel_size=1)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]
        feature_map = self.resnet(x)

        features = self.convolution_mapping(feature_map)
        height, width = features.shape[2:]
        batch_size = features.shape[0]
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

        if self.point_pred == 1:
            point = self.point_pre_layer(features_encoded)
            point = torch.sigmoid(point)
            features_encoded = point * features_encoded + features_encoded

        bounding_box_attention_masks = self.segmentation_attention_head(
            latent_tensor, features_encoded.contiguous())

        trans_feature_maps = torch.cat(
            (features_encoded, bounding_box_attention_masks[:, 0]), dim=1)
        trans_feature_maps = self.trans_out_conv(trans_feature_maps)

        output = self.aspp(
            feature_map)  # (shape: (batch_size, num_classes, h/16, w/16))
        output = F.upsample(
            output, size=(h, w),
            mode="bilinear")  # (shape: (batch_size, num_classes, h, w))

        if self.point_pred == 0:
            return output

        elif self.point_pred == 1:
            return output, point


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = DeepLabV3(1, 50, 1)
    d = torch.rand((5, 3, 512, 512))
    o, p = model(d)
    print(o.size())