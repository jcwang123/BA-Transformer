import sys, os

root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, 'lib/Cell_DETR_master'))

import torch
import torch.nn as nn

from detr_new import CellDETR
from segmentation import ResFeaturePyramidBlock, ResPACFeaturePyramidBlock
#from transformer import BoundaryAwareTransformer, Transformer
from src.transformer import BoundaryAwareTransformer, Transformer

type_of_transformer = [BoundaryAwareTransformer, Transformer]


def cell_detr_128(pretrained=False, type_index=0):

    detr = CellDETR(num_classes=2,
                    transformer_type=type_of_transformer[type_index],
                    segmentation_head_block=ResFeaturePyramidBlock,
                    segmentation_head_final_activation=nn.Softmax,
                    backbone_convolution=nn.Conv2d,
                    segmentation_head_convolution=nn.Conv2d,
                    transformer_activation=nn.LeakyReLU,
                    backbone_activation=nn.LeakyReLU,
                    bounding_box_head_activation=nn.LeakyReLU,
                    classification_head_activation=nn.LeakyReLU,
                    segmentation_head_activation=nn.LeakyReLU)
    if pretrained:
        ckpt = torch.load(
            "/home/chenfei/my_codes/TransformerCode-master/lib/Cell_DETR_master/trained_models/Cell_DETR_A/detr_99.pt"
        )
        detr.load_state_dict(ckpt, strict=False)
        print("pretrained cell_detr's transformer!")
    return detr.transformer