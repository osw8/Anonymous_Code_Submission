from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
SCATTERFORMER_ROOT = Path(__file__).resolve().parent
for value in (BENCHMARK_ROOT, SCATTERFORMER_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from eeg_benchmark.engine import add_common_transfer_arguments, run_torch_transfer


try:
    from scatter_former import ScatterFormer as OriginalScatterFormer
    ORIGINAL_IMPORT_ERROR = None
except Exception as error:  
    OriginalScatterFormer = None
    ORIGINAL_IMPORT_ERROR = repr(error)


class CompactScatterFormer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(image).flatten(1))


class BenchmarkScatterFormer(nn.Module):
    def __init__(self, channel_count: int) -> None:
        super().__init__()
        self.channel_count = int(channel_count)
        self.backbone = (
            OriginalScatterFormer(mode='train')
            if OriginalScatterFormer is not None
            else CompactScatterFormer()
        )
        self.head = (
            self.backbone.proj
            if hasattr(self.backbone, 'proj')
            else getattr(self.backbone, 'head', self.backbone)
        )

    def model_information(self) -> dict[str, object]:
        return {
            'Baseline family': 'ScatterFormer',
            'Input adapter': 'standardized_clip_to_three_channel_time_channel_image',
            'Original implementation loaded': OriginalScatterFormer is not None,
            'Original import error': ORIGINAL_IMPORT_ERROR,
        }

    @staticmethod
    def _image(eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        signal = eeg * channel_mask[:, None, :, None].to(eeg.dtype)
        batch, views, channels, points = signal.shape
        signal = signal.permute(0, 2, 1, 3).reshape(batch, channels, views * points)
        mean = signal.mean(dim=-1, keepdim=True)
        std = signal.std(dim=-1, keepdim=True).clamp_min(1e-5)
        normalized = (signal - mean) / std
        diff = F.pad(normalized.diff(dim=-1), (1, 0))
        magnitude = normalized.abs()
        image = torch.stack((normalized, diff, magnitude), dim=1)
        return F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False)

    def forward(self, eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(self._image(eeg, channel_mask))
        if isinstance(output, tuple):
            output = output[0]
        if output.ndim == 1:
            return output
        if output.shape[-1] == 1:
            return output.reshape(-1)
        if output.shape[-1] != 2:
            raise ValueError(f'ScatterFormer expected binary output, got {tuple(output.shape)}')
        values = output.float()
        if torch.all(values >= 0.0) and torch.all(values <= 1.0):
            return torch.log(values[:, 1].clamp_min(1e-6)) - torch.log(values[:, 0].clamp_min(1e-6))
        return values[:, 1] - values[:, 0]


def build_model(contract):
    return BenchmarkScatterFormer(len(contract.channel_keys))


def load_pretrained(model, path, contract):
    return {'status': 'not_applicable', 'path': str(path)}


def configure_transfer(model, mode):
    for parameter in model.parameters():
        parameter.requires_grad = True


def forward_clip(model, eeg, channel_mask):
    return model(eeg, channel_mask)


def budget_modules(model):
    return model.head, model.backbone


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='ScatterFormer benchmark adapter')
    parser.add_argument('--model', choices=['ScatterFormer'], default='ScatterFormer')
    return add_common_transfer_arguments(parser)


if __name__ == '__main__':
    run_torch_transfer(
        build_parser().parse_args(),
        model_name='ScatterFormer',
        input_spec_name='steegformer',
        model_factory=build_model,
        pretrained_loader=load_pretrained,
        transfer_configurator=configure_transfer,
        forward_clip=forward_clip,
        budget_module_resolver=budget_modules,
        model_arguments={
            'scatterformer_input_adapter': 'steegformer_continuous_views_to_three_channel_time_channel_image',
            'scatterformer_image_size': [64, 64],
        },
    )
