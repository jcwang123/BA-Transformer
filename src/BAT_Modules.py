import torch.nn.functional as F
import torch.nn as nn
import torch

class CrossAttention(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=512,
                 dropout=0.0):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.LeakyReLU()

    
    def forward(self, tgt, src):
        "tgt shape: Batch_size, C, H, W "
        "src shape: Batch_size, 1, C    "
        
        B, C, h, w = tgt.shape
        tgt = tgt.view(B, C, h*w).permute(2,0,1)  # shape: L, B, C
        
        src = src.permute(1,0,2)  # shape: Q:1, B, C
        
        fusion_feature = self.cross_attn(query=tgt,
                                         key=src,
                                         value=src)[0]
        tgt = tgt + self.dropout1(fusion_feature)
        tgt = self.norm1(tgt)
        tgt1 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt1)
        tgt = self.norm2(tgt)
        return tgt.permute(1, 2, 0).view(B, C, h, w)

class BoundaryCrossAttention(CrossAttention):
    def __init__(self,
                 d_model,
                 nhead,
                 BAG_type='2D',
                 Atrous=True,
                 dim_feedforward=512,
                 dropout=0.0):
        super().__init__(d_model, nhead, dim_feedforward, dropout)
        
        #self.BAG = nn.Sequential(
        #    nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
        #    nn.BatchNorm2d(d_model),
        #    nn.ReLU(inplace=False),
        #    nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
        #    nn.BatchNorm2d(d_model),
        #    nn.ReLU(inplace=False),
        #    nn.Conv2d(d_model, 1, kernel_size=1))
        self.BAG_type = BAG_type
        if self.BAG_type == '1D':
            if Atrous:
                self.BAG = BoundaryWiseAttentionGateAtrous1D(d_model)
            else:
                self.BAG = BoundaryWiseAttentionGate1D(d_model)
        elif self.BAG_type == '2D':
            if Atrous:
                self.BAG = BoundaryWiseAttentionGateAtrous2D(d_model)
            else:
                self.BAG = BoundaryWiseAttentionGate2D(d_model)
    
    def forward(self, tgt, src):
        "tgt shape: Batch_size, C, H, W "
        "src shape: Batch_size, 1, C    "
        
        B, C, h, w = tgt.shape
        tgt = tgt.view(B, C, h*w).permute(2,0,1)  # shape: L, B, C
        
        src = src.permute(1,0,2)  # shape: Q:1, B, C
        
        fusion_feature = self.cross_attn(query=tgt,
                                         key=src,
                                         value=src)[0]
        tgt = tgt + self.dropout1(fusion_feature)
        tgt = self.norm1(tgt)
        tgt1 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt1)
        tgt = self.norm2(tgt)

        if self.BAG_type == '1D':
            tgt = tgt.permute(1,2,0)
            tgt, weights = self.BAG(tgt)
            tgt = tgt.view(B, C, h, w).contiguous()
            weights = weights.view(B, 1, h, w)
        elif self.BAG_type == '2D':
            tgt = tgt.permute(1,2,0).view(B, C, h, w)
            tgt, weights = self.BAG(tgt)
            tgt = tgt.contiguous()
        return tgt, weights
    
class MultiHeadAttention(nn.Module):
    """
    This class implements a multi head attention module like proposed in:
    https://arxiv.org/abs/2005.12872
    """
    def __init__(self, query_dimension: int = 64, hidden_features: int = 64, number_of_heads: int = 16,
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
        self.layer_box_embedding = nn.Linear(in_features=query_dimension, out_features=hidden_features, bias=True)
        # Init convolution layer
        self.layer_image_encoding = nn.Conv2d(in_channels=query_dimension, out_channels=hidden_features,
                                              kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        # Init normalization factor
        self.normalization_factor = torch.tensor(self.hidden_features / self.number_of_heads, dtype=torch.float).sqrt()

        # Linear
        self.linear = nn.Linear(in_features=number_of_heads, out_features=1)

    def forward(self, input_box_embeddings: torch.Tensor, input_image_encoding: torch.Tensor) -> torch.Tensor:
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
        output_box_embeddings = output_box_embeddings.view(output_box_embeddings.shape[0],
                                                           output_box_embeddings.shape[1],
                                                           self.number_of_heads,
                                                           self.hidden_features // self.number_of_heads)
        # Reshape output image encoding
        output_image_encoding = output_image_encoding.view(output_image_encoding.shape[0],
                                                           self.number_of_heads,
                                                           self.hidden_features // self.number_of_heads,
                                                           output_image_encoding.shape[-2],
                                                           output_image_encoding.shape[-1])
        # Combine tensors and normalize
        output = torch.einsum("bqnc,bnchw->bqnhw",
                              output_box_embeddings * self.normalization_factor,
                              output_image_encoding)
        # Apply softmax
        output = F.softmax(output.flatten(start_dim=2), dim=-1).view_as(output)

        # Linear: to generate one map
        b, _, _, h, w = output.shape 
        output = torch.sigmoid(self.linear(output.flatten(start_dim=3).permute(0,1,3,2))).view(b,1,h,w)

        # Perform dropout if utilized
        if self.dropout > 0.0:
            output = F.dropout(input=output, p=self.dropout, training=self.training)
#         print("MultiHead Attention",output.shape)
        return output.contiguous()

    
class BoundaryWiseAttentionGateAtrous2D(nn.Module):
    def __init__(self, in_channels, hidden_channels = None):

        super(BoundaryWiseAttentionGateAtrous2D,self).__init__()

        modules = []

        if hidden_channels == None:
            hidden_channels = in_channels // 2

        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)))

        self.convs = nn.ModuleList(modules)
        
        self.conv_out = nn.Conv2d(5 * hidden_channels, 1, 1, bias=False)
    def forward(self, x):
        " x.shape: B, C, H, W "
        " return: feature, weight (B,C,H,W) "
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        weight = torch.sigmoid(self.conv_out(res))
        x = x * weight + x
        return x, weight

class BoundaryWiseAttentionGateAtrous1D(nn.Module):
    def __init__(self, in_channels, hidden_channels = None):

        super(BoundaryWiseAttentionGateAtrous1D,self).__init__()

        modules = []

        if hidden_channels == None:
            hidden_channels = in_channels // 2

        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)))

        self.convs = nn.ModuleList(modules)
        
        self.conv_out = nn.Conv1d(5 * hidden_channels, 1, 1, bias=False)
    def forward(self, x):
        " x.shape: B, C, L "
        " return: feature, weight (B,C,L) "
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        weight = torch.sigmoid(self.conv_out(res))
        x = x * weight + x
        return x, weight

class BoundaryWiseAttentionGate2D(nn.Sequential):
    def __init__(self, in_channels, hidden_channels = None):
        super(BoundaryWiseAttentionGate2D,self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, 1, kernel_size=1))
    def forward(self, x):
        " x.shape: B, C, H, W "
        " return: feature, weight (B,C,H,W) "
        weight = torch.sigmoid(super(BoundaryWiseAttentionGate2D,self).forward(x))
        x = x * weight + x
        return x, weight

class BoundaryWiseAttentionGate1D(nn.Sequential):
    def __init__(self, in_channels, hidden_channels = None):
        super(BoundaryWiseAttentionGate1D,self).__init__(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels, 1, kernel_size=1))
    def forward(self, x):
        " x.shape: B, C, L "
        " return: feature, weight (B,C,L) "
        weight = torch.sigmoid(super(BoundaryWiseAttentionGate1D,self).forward(x))
        x = x * weight + x
        return x, weight