from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from eeg_benchmark.engine import add_common_transfer_arguments, run_torch_transfer


class _NoNorm(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class ChannelProjection(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = 20) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.projection = nn.Conv1d(
            self.input_channels,
            self.output_channels,
            kernel_size=1,
            bias=False,
        )
        with torch.no_grad():
            self.projection.weight.zero_()
            diagonal = min(self.input_channels, self.output_channels)
            indices = torch.arange(diagonal)
            self.projection.weight[indices, indices, 0] = 1.0

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.projection(signal)


class ConvEncoderBENDR(nn.Module):
    def __init__(
        self,
        input_channels: int = 20,
        encoder_h: int = 512,
        widths: tuple[int, ...] = (3, 2, 2, 2, 2, 2),
        strides: tuple[int, ...] = (3, 2, 2, 2, 2, 2),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(widths) != len(strides):
            raise ValueError('BENDR convolution widths and strides must align')
        self.encoder = nn.Sequential()
        current = int(input_channels)
        for index, (width, stride) in enumerate(zip(widths, strides)):
            kernel = int(width) if int(width) % 2 else int(width) + 1
            dropout_layer: nn.Module = (
                nn.Identity() if dropout == 0.0 else nn.Dropout1d(dropout)
            )
            self.encoder.add_module(
                f'Encoder_{index}',
                nn.Sequential(
                    nn.Conv1d(
                        current,
                        encoder_h,
                        kernel,
                        stride=int(stride),
                        padding=kernel // 2,
                    ),
                    dropout_layer,
                    nn.GroupNorm(encoder_h // 2, encoder_h),
                    nn.GELU(),
                ),
            )
            current = encoder_h

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.encoder(signal)


class BENDRContextualizer(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        hidden_feedforward: int = 3076,
        heads: int = 8,
        layers: int = 8,
        dropout: float = 0.0,
        position_encoder: int = 25,
        start_token: float = -5.0,
    ) -> None:
        super().__init__()
        transformer_dim = in_features * 3
        self.in_features = int(in_features)
        self.start_token = float(start_token)
        self.norm = nn.LayerNorm(transformer_dim)
        position = nn.Conv1d(
            in_features,
            in_features,
            position_encoder,
            padding=position_encoder // 2,
            groups=16,
        )
        nn.init.normal_(position.weight, mean=0.0, std=2.0 / transformer_dim)
        nn.init.constant_(position.bias, 0.0)
        position = nn.utils.parametrizations.weight_norm(position, dim=2)
        self.relative_position = nn.Sequential(position, nn.GELU())
        self.input_conditioning = nn.Sequential(
            nn.Identity(),
            nn.LayerNorm(in_features),
            nn.Identity(),
            nn.Linear(in_features, transformer_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=heads,
            dim_feedforward=hidden_feedforward,
            dropout=dropout,
            activation='gelu',
        )
        layer.norm1 = _NoNorm()
        layer.norm2 = _NoNorm()
        self.transformer_layers = nn.ModuleList(
            [self._copy_layer(layer) for _ in range(layers)]
        )
        self.output_layer = nn.Conv1d(transformer_dim, in_features, 1)

    @staticmethod
    def _copy_layer(layer: nn.Module) -> nn.Module:
        import copy

        return copy.deepcopy(layer)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        value = encoded + self.relative_position(encoded)
        value = value.transpose(1, 2)
        value = self.input_conditioning[1](value)
        value = self.input_conditioning[3](value)
        value = value.permute(1, 0, 2)
        start = value.new_full(
            (1, value.shape[1], value.shape[2]),
            self.start_token,
        )
        value = torch.cat([start, value], dim=0)
        for layer in self.transformer_layers:
            value = layer(value)
        return self.output_layer(value.permute(1, 2, 0))


class BENDRTransferModel(nn.Module):
    def __init__(self, contract_channels: int) -> None:
        super().__init__()
        self.contract_channels = int(contract_channels)
        self.pretrained_channels = 20
        self.channel_projection = ChannelProjection(
            self.contract_channels,
            self.pretrained_channels,
        )
        self.encoder = ConvEncoderBENDR(input_channels=self.pretrained_channels)
        self.contextualizer = BENDRContextualizer()
        self.classifier = nn.Linear(512, 1)

    def forward(self, eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 4:
            raise ValueError(
                f'BENDR expects batch by view by channel by time, got {tuple(eeg.shape)}'
            )
        batch, views, channels, points = eeg.shape
        if channel_mask.shape != (batch, channels):
            raise ValueError('BENDR channel mask does not match its signal')
        signal = eeg * channel_mask[:, None, :, None].to(eeg.dtype)
        signal = signal.reshape(batch * views, channels, points)
        signal = self.channel_projection(signal)
        encoded = self.encoder(signal)
        context = self.contextualizer(encoded)
        view_logits = self.classifier(context[:, :, -1]).reshape(batch, views)
        return view_logits.mean(dim=1)

    def model_information(self) -> dict[str, object]:
        return {
            'Architecture': 'official BENDR convolutional encoder plus contextualizer',
            'Pretrained input channels': self.pretrained_channels,
            'Contract input channels': self.contract_channels,
            'Encoder width': 512,
            'Transformer layers': 8,
            'Transformer heads': 8,
            'View length': '4 seconds at 250 Hz',
            'View aggregation': 'mean logit across deterministic views',
        }


def model_factory(contract) -> BENDRTransferModel:
    return BENDRTransferModel(len(contract.channel_keys))


def load_pretrained(
    model: BENDRTransferModel,
    path: Path,
    contract,
) -> dict[str, object]:
    state = load_file(str(path), device='cpu')
    accepted_prefixes = ('encoder.', 'contextualizer.')
    selected = {
        key: value
        for key, value in state.items()
        if key.startswith(accepted_prefixes)
    }
    result = model.load_state_dict(selected, strict=False)
    skipped = sorted(set(state) - set(selected))
    missing_backbone = sorted(
        key
        for key in result.missing_keys
        if key.startswith(accepted_prefixes)
    )
    if missing_backbone or result.unexpected_keys:
        raise ValueError(
            'BENDR pretrained backbone is incompatible: '
            f'missing={missing_backbone},unexpected={result.unexpected_keys}'
        )
    expected = len(selected)
    if expected < 90:
        raise ValueError(f'BENDR pretrained backbone is unexpectedly small: {expected}')
    return {
        'path': str(path),
        'format': 'safetensors',
        'loaded_tensor_count': expected,
        'skipped_task_head_keys': skipped,
        'missing_task_specific_keys': sorted(result.missing_keys),
        'pretrained_channel_count': 20,
        'task_channel_count': len(contract.channel_keys),
        'policy': 'strict_backbone_load_new_channel_bridge_and_binary_head',
    }


def transfer_configurator(model: BENDRTransferModel, mode: str) -> None:
    if mode not in {'SourceOnly', 'FT'}:
        raise ValueError(f'Unsupported BENDR mode: {mode}')


def forward_clip(
    model: BENDRTransferModel,
    eeg: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    return model(eeg, channel_mask)


def budget_module_resolver(model: BENDRTransferModel):
    return model.classifier, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='BENDR transfer baseline using the official pretrained backbone'
    )
    parser.add_argument('--model', choices=['BENDR'], default='BENDR')
    add_common_transfer_arguments(parser)
    parser.add_argument('--progress-update-interval', type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.use_pretrained:
        raise ValueError('BENDR benchmark requires the supplied pretrained weight')
    if args.prefetch_factor <= 0:
        raise ValueError('BENDR prefetch factor must be positive')
    if args.progress_update_interval <= 0:
        raise ValueError('BENDR progress update interval must be positive')
    run_torch_transfer(
        args,
        model_name='BENDR',
        input_spec_name='bendr',
        model_factory=model_factory,
        pretrained_loader=load_pretrained,
        transfer_configurator=transfer_configurator,
        forward_clip=forward_clip,
        budget_module_resolver=budget_module_resolver,
        model_arguments={
            'pretrained_channels': 20,
            'encoder_width': 512,
            'transformer_layers': 8,
            'transformer_heads': 8,
            'view_seconds': 4.0,
            'view_sampling_frequency': 250,
            'channel_projection': 'identity_initialized_learnable_1x1',
        },
        runtime_arguments={
            'prefetch_factor': args.prefetch_factor,
            'progress_update_interval': args.progress_update_interval,
            'evaluation_context': 'torch_inference_mode',
        },
    )


if __name__ == '__main__':
    main()
