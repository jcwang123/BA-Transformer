import os
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from Ours.Cell_DETR_master.transformer import Transformer,TransformerEncoder,TransformerEncoderLayer
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.rand((5,1,128,128)).cpu()
# resnet = models.resnet18().cuda()
# cnn_encoder = nn.Sequential(*list(resnet.children())[:-4])
# features = cnn_encoder(x)

from torch.nn.modules import Conv2d,LeakyReLU
hidden_features=128
query_dimension = 128
dropout = 0.0
num_classes = 3
number_of_heads = 16
num_encoder_layers = 3
num_decoder_layers = 2
number_of_query_positions=12
transformer_attention_heads = 8
segmentation_attention_heads = 8
transformer_activation = nn.LeakyReLU
classification_head_activation = nn.LeakyReLU
normalize_before=False

from backbone import Backbone, DenseNetBlock, StandardBlock, ResNetBlock
from modules.modulated_deform_conv import ModulatedDeformConvPack
from pade_activation_unit.utils import PAU
backbone_channels = ((1, 64), (64, 128), (128, 256), (256, 256))
backbone_block = ResNetBlock
backbone_convolution = nn.Conv2d
backbone_normalization = nn.BatchNorm2d
backbone_activation = nn.LeakyReLU
backbone_pooling = nn.AvgPool2d
backbone = Backbone(channels=backbone_channels, block=backbone_block, convolution=backbone_convolution,
                                 normalization=backbone_normalization, activation=backbone_activation,
                                 pooling=backbone_pooling)
convolution_mapping = nn.Conv2d(in_channels=backbone_channels[-1][-1], out_channels=hidden_features,
                                             kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
                                             

features, feature_list = backbone(x)
features = convolution_mapping(features)
height, width = features.shape[2:]
print("height width: ",height, width)
# Get batch size
batch_size = features.shape[0]
# Make positional embeddings
print("features size: ",features.shape,len(feature_list))

query_positions = nn.Parameter(
            data=torch.randn(number_of_query_positions, hidden_features, dtype=torch.float),
            requires_grad=True)

row_embedding = nn.Parameter(data=torch.randn(50, hidden_features // 2, dtype=torch.float),
                                          requires_grad=True)
column_embedding = nn.Parameter(data=torch.randn(50, hidden_features // 2, dtype=torch.float),
                                     requires_grad=True)
positional_embeddings = torch.cat([column_embedding[:height].unsqueeze(dim=0).repeat(height, 1, 1),
                                   row_embedding[:width].unsqueeze(dim=1).repeat(1, width, 1)],
                                  dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
print("query_positions size: ", query_positions.shape)
print("positional_embeddings size: ",positional_embeddings.shape)

positional_embeddings.max(), positional_embeddings.min()
Input = features+positional_embeddings
print(Input.shape)
plt.figure()
plt.subplot(1,3,1)
plt.imshow(features.detach()[0,63,:])
plt.subplot(1,3,2)
plt.imshow(positional_embeddings.detach()[0,63,:])
plt.subplot(1,3,3)
plt.imshow(Input.detach()[0,63,:])
plt.figure()
plt.subplot(1,2,1)
plt.imshow(positional_embeddings.detach()[0,63,:])
plt.subplot(1,2,2)
plt.imshow(positional_embeddings.detach()[0,64,:])



transformer = Transformer(d_model=hidden_features, nhead=transformer_attention_heads,
                                       num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                       dropout=dropout, dim_feedforward=4 * hidden_features,
                                       activation=transformer_activation)
latent_tensor, features_encoded = transformer(features, None, query_positions, positional_embeddings)
latent_tensor = latent_tensor.permute(2, 0, 1)
print(latent_tensor.shape,features_encoded.shape)



from Ours.Cell_DETR_master.segmentation import MultiHeadAttention, SegmentationHead, ResFeaturePyramidBlock, ResPACFeaturePyramidBlock
segmentation_head_channels = ((128 + 8, 128), (128, 64), (64, 32))
segmentation_head_feature_channels = (256, 128, 64)
segmentation_head_convolution = Conv2d
segmentation_head_normalization = nn.InstanceNorm2d
segmentation_head_activation = nn.LeakyReLU
segmentation_head_block = ResPACFeaturePyramidBlock
segmentation_head_final_activation = nn.Sigmoid
segmentation_attention_head = MultiHeadAttention(query_dimension=hidden_features,
                                      hidden_features=hidden_features,
                                      number_of_heads=segmentation_attention_heads,
                                      dropout=dropout)
segmentation_head = SegmentationHead(channels=segmentation_head_channels,
                          feature_channels=segmentation_head_feature_channels,
                          convolution=segmentation_head_convolution,
                          normalization=segmentation_head_normalization,
                          activation=segmentation_head_activation,
                          block=segmentation_head_block,
                          number_of_query_positions=number_of_query_positions,
                          softmax=isinstance(segmentation_head_final_activation(), nn.Softmax))



bounding_box_attention_masks = segmentation_attention_head(
latent_tensor, features_encoded.contiguous())
instance_segmentation_prediction = segmentation_head(features.contiguous(),
                                  bounding_box_attention_masks.contiguous(),
                                  feature_list[-2::-1])