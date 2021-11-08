from typing import Type, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    This class implements a multi head attention module like proposed in:
    https://arxiv.org/abs/2005.12872
    """

    def __init__(self, query_dimension: int = 64, hidden_features: int = 64, number_of_heads: int = 16,
                 dropout: float = 0.0, feature_length: int = 32*32) -> None:
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
        #self.linear = nn.Conv1d(number_of_heads, 1, kernel_size=1)

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
        return output.squeeze(3).contiguous()
