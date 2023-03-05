import torch
from torch import nn as nn
from torch.nn import LayerNorm

from nemo.collections.asr.parts.utils.activations import Swish
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin
from nemo.utils import logging


from safari.src.models.sequence.h3 import H3


__all__ = ['H3ASREncoderConvolution', 'H3ASREncoderFeedForward', 'H3ASREncoderLayer']


class H3ASREncoderLayer(torch.nn.Module, AdapterModuleMixin, AccessMixin):
    """A single block of the H3ASREncoder encoder.
    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
    """

    def __init__(
        self,
        d_model,
        d_ff,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        dropout=0.1,
        d_state=64,
    ):
        super(H3ASREncoderLayer, self).__init__()

        self.fc_factor = 0.5

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = H3ASREncoderFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = H3ASREncoderConvolution(d_model=d_model, kernel_size=conv_kernel_size, norm_type=conv_norm_type)

        # multi-headed self-attention module
        self.norm_h3 = LayerNorm(d_model)
        self.h3 = H3(d_model, dropout=dropout, d_state=d_state)

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = H3ASREncoderFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(self, x, lengths):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_h3(residual)
        # pad_mask
        if lengths is not None:
            pad_mask = torch.arange(0, x.shape[-1], device=x.device).expand(
                lengths.size(0), -1
            ) < lengths.unsqueeze(-1)
            x = x.float().masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.h3(x)
        
        residual = residual + self.dropout(x)

        x = self.norm_conv(residual)
        x = self.conv(x, lengths=lengths)
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)

        return x


class H3ASREncoderConvolution(nn.Module):
    """The convolution module for the H3ASREncoder model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
    """

    def __init__(self, d_model, kernel_size, norm_type='batch_norm'):
        super(H3ASREncoderConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )
        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(d_model)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not valid!")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)

        if lengths is not None:
            pad_mask = torch.arange(0, x.shape[-1], device=x.device).expand(
                lengths.size(0), -1
            ) < lengths.unsqueeze(-1)
            x = x.float().masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x)

        if isinstance(self.batch_norm, nn.LayerNorm):
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return x


class H3ASREncoderFeedForward(nn.Module):
    """
    feed-forward module of H3ASREncoder model.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish()):
        super(H3ASREncoderFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x