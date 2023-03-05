import math
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.distributed
import torch.nn as nn

from models import H3ASREncoderLayer
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling, StackingSubsampling
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType

__all__ = ['H3ASREncoderLayer']


class H3ASREncoder(NeuralModule, Exportable):
    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        dev = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(dev)
        input_example_length = torch.randint(1, max_dim, (max_batch,)).to(dev)
        return tuple([input_example, input_example_length])

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_channels=-1,
        ff_expansion_factor=4,
        xscaling=True,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        dropout=0.1,
        dropout_emb=0.1,
        d_state=64
    ):
        super().__init__()

        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self._feat_in = feat_in
        self.scale = math.sqrt(self.d_model)

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            if subsampling == 'stacking':
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor, feat_in=feat_in, feat_out=d_model
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    activation=nn.ReLU(),
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        self._feat_out = d_model

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = H3ASREncoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                dropout=dropout,
                d_state=d_state
            )
            self.layers.append(layer)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model

    @typecheck()
    def forward(self, audio_signal, length=None):
        return self.forward_for_export(audio_signal=audio_signal, length=length)

    @typecheck()
    def forward_for_export(self, audio_signal, length):
        if length is None:
            length = audio_signal.new_full(
                audio_signal.size(0), max_audio_length, dtype=torch.int32, device=self.seq_range.device
            )

        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(audio_signal, length)


        for lth, layer in enumerate(self.layers):
            audio_signal = layer(x=audio_signal, lengths=length)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        return audio_signal, length