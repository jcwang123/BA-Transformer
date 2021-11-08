import torch
import torch.nn as nn

from timm.models.registry import register_model
import torch.nn.functional as F
import numpy as np

from models.DETR_modules.transformer import BoundaryAwareTransformer, Transformer
from models.DETR_modules.backbone import build_R50
from models.DETR_modules.detr import DETR

type_of_transformer = [BoundaryAwareTransformer, Transformer]

@register_model
def detr_base_256(pretrained=False, type_index=0):
    backbone = build_R50()
    transformer_type = type_of_transformer[type_index]
    transformer = transformer_type(d_model=256, 
                                   dropout=0.1, 
                                   nhead=8, 
                                   dim_feedforward=2048, 
                                   num_encoder_layers=3, 
                                   num_decoder_layers=2)
    model = DETR(
        backbone, transformer,
        num_classes=91, num_queries=100, aux_loss=False)
    if pretrained:
        ckpt = torch.load('/home/chenfei/my_codes/TransformerCode-master/Ours/pretrained/detr-r50-e632da11.pth')
        model.load_state_dict(ckpt['model'], strict=False)
        print("load detr's pretrained model")
    return model.transformer