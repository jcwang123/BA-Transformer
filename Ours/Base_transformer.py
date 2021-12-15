import sys, os

root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(root_path))
sys.path.insert(0, os.path.join(root_path, 'lib'))
sys.path.insert(0, os.path.join(root_path, 'lib/Cell_DETR_master'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from Ours.base import DeepLabV3 as base

from src.BAT_Modules import BoundaryCrossAttention, CrossAttention
from src.BAT_Modules import MultiHeadAttention as Attention_head
from src.transformer import BoundaryAwareTransformer, Transformer


class BAT(nn.Module):
    def __init__(
            self,
            num_classes,
            num_layers,
            point_pred,
            decoder=False,
            transformer_type_index=0,
            hidden_features=128,  # 256
            number_of_query_positions=1,
            segmentation_attention_heads=8):

        super(BAT, self).__init__()

        self.num_classes = num_classes
        self.point_pred = point_pred
        self.transformer_type = "BoundaryAwareTransformer" if transformer_type_index == 0 else "Transformer"
        self.use_decoder = decoder

        self.deeplab = base(num_classes, num_layers)

        in_channels = 2048 if num_layers == 50 else 512

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

        self.transformer = [
            Transformer(d_model=hidden_features),
            BoundaryAwareTransformer(d_model=hidden_features)
        ][point_pred]

        if self.use_decoder:
            self.BCA = BoundaryCrossAttention(hidden_features, 8)

        self.trans_out_conv = nn.Conv2d(in_channels=hidden_features,
                                        out_channels=in_channels,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=(0, 0),
                                        bias=True)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]
        feature_map = self.deeplab.resnet(x)

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

        if self.transformer_type == 'BoundaryAwareTransformer':
            latent_tensor, features_encoded, point_maps = self.transformer(
                features, None, self.query_positions, positional_embeddings)
        else:
            latent_tensor, features_encoded = self.transformer(
                features, None, self.query_positions, positional_embeddings)
            point_maps = []

        latent_tensor = latent_tensor.permute(2, 0, 1)
        # shape:(bs, 1 , 128)

        if self.use_decoder:
            features_encoded, point_dec = self.BCA(features_encoded,
                                                   latent_tensor)
            point_maps.append(point_dec)

        trans_feature_maps = self.trans_out_conv(
            features_encoded.contiguous())  #.contiguous()

        trans_feature_maps = trans_feature_maps + feature_map

        output = self.deeplab.aspp(
            trans_feature_maps
        )  # (shape: (batch_size, num_classes, h/16, w/16))
        output = F.interpolate(
            output, size=(h, w),
            mode="bilinear")  # (shape: (batch_size, num_classes, h, w))

        if self.point_pred == 1:
            return output, point_maps

        return output