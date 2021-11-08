import sys, os

root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(root_path))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.resnet import ResNet50_OS16, ResNet18_OS8
from models.utils.ASPP import ASPP, ASPP_Bottleneck
from models.Base import DeepLabV3 as base

from models.transformers.Cell_DETR import cell_detr_128
from models.transformers.DETR import detr_base_256

class Transformer(nn.Module):
    def __init__(self,
                 num_classes,
                 num_layers,
                 hidden_features=128,
                 number_of_query_positions=1):

        super(Transformer, self).__init__()

        self.num_classes = num_classes
        
        self.deeplab = base(num_classes, num_layers)
        #self.deeplab.load_state_dict(torch.load("/home/chenfei/my_codes/TransformerCode-master/logs/isbi2018/_loss_0_aug_0/arch__trans_0_point_0/fold_0_ver_0/model/best.pkl"))
        #print("load pretained parameters from deeplabv3!")

        in_channels = 2048 if num_layers == 50 else 512

        self.convolution_mapping = nn.Conv2d(in_channels=in_channels,
                                             out_channels=hidden_features,
                                             kernel_size=(1, 1),
                                             stride=(1, 1),
                                             padding=(0, 0),
                                             bias=True)

        self.query_positions = nn.Parameter(data=torch.randn(
            number_of_query_positions, hidden_features, dtype=torch.float), requires_grad=True)

        self.row_embedding = nn.Parameter(data=torch.randn(
            100, hidden_features // 2, dtype=torch.float), requires_grad=True)

        self.column_embedding = nn.Parameter(data=torch.randn(
            100, hidden_features // 2, dtype=torch.float), requires_grad=True)

        #loading Cell_DETR weights
        self.transformer = cell_detr_128(pretrained=False, type_index=1)
        #self.transformer = detr_base_256(pretrained=True, type_index=1)

        self.trans_out_conv = nn.Conv2d(hidden_features, in_channels, 1, 1)

            
    def forward(self, x):
        height = x.size()[2]
        weight = x.size()[3]
        feature_map = self.deeplab.resnet(x)
        b, _, h, w = feature_map.size()

        features = self.convolution_mapping(feature_map)
        positional_embeddings = torch.cat(
            [self.column_embedding[:h].unsqueeze(dim=0).repeat(h, 1, 1),
             self.row_embedding[:w].unsqueeze(dim=1).repeat(1, w, 1)],
             dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
        
        latent_tensor, features_encoded = self.transformer(
            features, None, self.query_positions, positional_embeddings)

        trans_feature_maps = self.trans_out_conv(features_encoded)
        
        trans_feature_maps = trans_feature_maps + feature_map

        output = self.deeplab.aspp(
            trans_feature_maps)  # (shape: (batch_size, num_classes, h/16, w/16))
        output = F.interpolate(
            output, size=(height, weight),
            mode="bilinear")  # (shape: (batch_size, num_classes, h, w))

        return output